import numpy as np
import pandas as pd
import os
from scipy import signal, fft
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress indicator
    def tqdm(iterable, desc="Processing"):
        total = len(iterable)
        for i, item in enumerate(iterable):
            if i % max(1, total // 20) == 0:  # Update every 5%
                print(f"\r{desc}: {i}/{total} ({100*i//total}%)", end='', flush=True)
            yield item
        print(f"\r{desc}: {total}/{total} (100%)", flush=True)

# --- CONFIGURATION ---
NPZ_FILE = "/mnt/dataset-augmentation/siamese_A01_augmented.npz"
OUTPUT_DIR = "/mnt/signal-processing-techniques/signal_processing_results"

class SpectralFeatureExtractor:
    """
    Extract frequency domain features from time series data
    """
    def __init__(self, fs=30):
        self.fs = fs  # Sampling frequency (30 Hz)
        
    def compute_fft_features(self, signal_data):
        """
        Compute FFT-based features
        Returns: frequency spectrum, power spectrum, dominant frequencies
        """
        # Apply window to reduce spectral leakage
        window = np.hanning(len(signal_data))
        windowed_signal = signal_data * window
        
        # Compute FFT
        fft_vals = fft.fft(windowed_signal)
        fft_freq = fft.fftfreq(len(signal_data), 1/self.fs)
        
        # Take only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_vals = fft_vals[positive_freq_idx]
        
        # Power spectrum
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Normalize
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        return fft_freq, fft_vals, power_spectrum
    
    def compute_spectral_centroid(self, power_spectrum, frequencies):
        """
        Compute the spectral centroid (center of mass of spectrum)
        """
        return np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
    
    def compute_spectral_spread(self, power_spectrum, frequencies, centroid):
        """
        Compute spectral spread (variance around centroid)
        """
        return np.sqrt(np.sum(((frequencies - centroid) ** 2) * power_spectrum) / np.sum(power_spectrum))
    
    def compute_spectral_entropy(self, power_spectrum):
        """
        Compute spectral entropy (measure of spectral complexity)
        """
        # Normalize to probability distribution
        prob = power_spectrum / np.sum(power_spectrum)
        # Remove zeros to avoid log(0)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
    
    def compute_spectral_rolloff(self, power_spectrum, frequencies, rolloff_percent=0.85):
        """
        Frequency below which rolloff_percent of energy is contained
        """
        cumsum = np.cumsum(power_spectrum)
        rolloff_idx = np.where(cumsum >= rolloff_percent * cumsum[-1])[0][0]
        return frequencies[rolloff_idx]
    
    def extract_dominant_frequencies(self, power_spectrum, frequencies, n_peaks=5):
        """
        Extract N dominant frequency components
        """
        # Find peaks in power spectrum
        peaks, properties = signal.find_peaks(power_spectrum, height=0)
        
        if len(peaks) == 0:
            return np.zeros(n_peaks), np.zeros(n_peaks)
        
        # Sort by height (power)
        sorted_idx = np.argsort(properties['peak_heights'])[::-1]
        top_peaks = peaks[sorted_idx[:n_peaks]]
        
        # Pad if fewer than n_peaks
        if len(top_peaks) < n_peaks:
            top_peaks = np.pad(top_peaks, (0, n_peaks - len(top_peaks)), mode='constant')
        
        dom_freqs = frequencies[top_peaks]
        dom_powers = power_spectrum[top_peaks]
        
        return dom_freqs, dom_powers
    
    def compute_mel_frequency_features(self, signal_data, n_mels=13):
        """
        Compute Mel-frequency cepstral coefficients (MFCCs)
        Adapted for motion data instead of audio
        """
        # Compute power spectrum
        _, _, power_spectrum = self.compute_fft_features(signal_data)
        
        # Simplified mel-scale binning
        n_fft = len(power_spectrum)
        mel_bins = np.linspace(0, n_fft-1, n_mels+2, dtype=int)
        
        mel_features = []
        for i in range(n_mels):
            bin_power = np.mean(power_spectrum[mel_bins[i]:mel_bins[i+2]])
            mel_features.append(bin_power)
        
        return np.array(mel_features)
    
    def extract_all_features(self, signal_data):
        """
        Extract comprehensive spectral features from signal
        """
        features = {}
        
        # FFT-based features
        frequencies, fft_vals, power_spectrum = self.compute_fft_features(signal_data)
        
        # Spectral statistics
        features['spectral_centroid'] = self.compute_spectral_centroid(power_spectrum, frequencies)
        features['spectral_spread'] = self.compute_spectral_spread(power_spectrum, frequencies, 
                                                                    features['spectral_centroid'])
        features['spectral_entropy'] = self.compute_spectral_entropy(power_spectrum)
        features['spectral_rolloff'] = self.compute_spectral_rolloff(power_spectrum, frequencies)
        
        # Dominant frequencies
        dom_freqs, dom_powers = self.extract_dominant_frequencies(power_spectrum, frequencies, n_peaks=5)
        for i, (freq, power) in enumerate(zip(dom_freqs, dom_powers)):
            features[f'dominant_freq_{i+1}'] = freq
            features[f'dominant_power_{i+1}'] = power
        
        # Mel-frequency features
        mel_features = self.compute_mel_frequency_features(signal_data)
        for i, mf in enumerate(mel_features):
            features[f'mel_feature_{i+1}'] = mf
        
        # Power in different frequency bands
        # Low: 0-1 Hz, Mid: 1-5 Hz, High: 5-15 Hz (walking gait frequencies)
        low_band = (frequencies >= 0) & (frequencies < 1)
        mid_band = (frequencies >= 1) & (frequencies < 5)
        high_band = (frequencies >= 5) & (frequencies < 15)
        
        features['power_low_band'] = np.sum(power_spectrum[low_band])
        features['power_mid_band'] = np.sum(power_spectrum[mid_band])
        features['power_high_band'] = np.sum(power_spectrum[high_band])
        
        return features, power_spectrum, frequencies


class WaveletFeatureExtractor:
    """
    Extract wavelet-based features for multi-resolution analysis
    NOTE: If PyWavelets unavailable, uses alternative multi-scale features
    """
    def __init__(self, wavelet='db4', levels=4):
        self.wavelet = wavelet
        self.levels = levels
        self.pywt_available = None
        self._warning_shown = False
    
    def compute_wavelet_features(self, signal_data):
        """
        Compute discrete wavelet transform features
        If PyWavelets unavailable, use alternative multi-scale decomposition
        """
        # Check PyWavelets availability once
        if self.pywt_available is None:
            try:
                from pywt import wavedec
                self.pywt_available = True
            except ImportError:
                self.pywt_available = False
                if not self._warning_shown:
                    print("⚠️  PyWavelets not available. Using alternative multi-scale features instead.")
                    self._warning_shown = True
        
        if self.pywt_available:
            try:
                from pywt import wavedec
                
                # Perform wavelet decomposition
                coeffs = wavedec(signal_data, self.wavelet, level=self.levels)
                
                features = {}
                
                # Extract statistics from each level
                for i, coeff in enumerate(coeffs):
                    level_name = 'approx' if i == 0 else f'detail_{i}'
                    features[f'wavelet_{level_name}_mean'] = np.mean(coeff)
                    features[f'wavelet_{level_name}_std'] = np.std(coeff)
                    features[f'wavelet_{level_name}_energy'] = np.sum(coeff ** 2)
                    features[f'wavelet_{level_name}_max'] = np.max(np.abs(coeff))
                
                return features
            except Exception as e:
                # Fallback to alternative features
                return self._compute_alternative_multiscale_features(signal_data)
        else:
            # Use alternative multi-scale features
            return self._compute_alternative_multiscale_features(signal_data)
    
    def _compute_alternative_multiscale_features(self, signal_data):
        """
        Alternative multi-scale decomposition using moving averages
        Simpler than wavelets but captures similar multi-resolution information
        """
        features = {}
        
        # Multi-scale smoothing with different window sizes
        # Approximates different wavelet decomposition levels
        window_sizes = [2, 4, 8, 16]
        
        for i, window_size in enumerate(window_sizes):
            # Simple moving average as approximation
            if len(signal_data) >= window_size:
                kernel = np.ones(window_size) / window_size
                smoothed = np.convolve(signal_data, kernel, mode='valid')
                
                # Detail coefficients: difference from smoothed version
                detail = signal_data[:len(smoothed)] - smoothed
                
                level_name = f'scale_{i+1}'
                features[f'multiscale_{level_name}_mean'] = np.mean(smoothed)
                features[f'multiscale_{level_name}_std'] = np.std(smoothed)
                features[f'multiscale_{level_name}_energy'] = np.sum(smoothed ** 2)
                features[f'multiscale_{level_name}_detail_energy'] = np.sum(detail ** 2)
        
        return features


class TimeSeriesSimilarity:
    """
    Compute various similarity measures between time series
    """
    def __init__(self):
        pass
    
    def dynamic_time_warping(self, seq1, seq2):
        """
        Compute DTW distance between two sequences
        """
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, :] = np.inf
        dtw_matrix[:, 0] = np.inf
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                              dtw_matrix[i, j-1],    # deletion
                                              dtw_matrix[i-1, j-1])  # match
        
        return dtw_matrix[n, m]
    
    def normalized_cross_correlation(self, seq1, seq2):
        """
        Compute normalized cross-correlation
        """
        # Normalize sequences
        seq1_norm = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
        seq2_norm = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)
        
        # Compute cross-correlation
        correlation = signal.correlate(seq1_norm, seq2_norm, mode='valid')
        max_corr = np.max(correlation) / len(seq1)
        
        return max_corr
    
    def autocorrelation_similarity(self, seq1, seq2, max_lag=20):
        """
        Compare autocorrelation functions of two sequences
        Captures periodicity patterns
        """
        def autocorr(x, lag):
            x_mean = np.mean(x)
            c0 = np.sum((x - x_mean) ** 2)
            c_lag = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
            return c_lag / (c0 + 1e-8)
        
        # Compute autocorrelations
        ac1 = [autocorr(seq1, lag) for lag in range(1, min(max_lag, len(seq1)//2))]
        ac2 = [autocorr(seq2, lag) for lag in range(1, min(max_lag, len(seq2)//2))]
        
        # Compare autocorrelation patterns
        min_len = min(len(ac1), len(ac2))
        if min_len > 0:
            similarity = 1 - euclidean(ac1[:min_len], ac2[:min_len]) / (min_len ** 0.5)
            return max(0, similarity)
        return 0.0
    
    def shape_based_distance(self, seq1, seq2):
        """
        Shape-based distance (insensitive to amplitude scaling)
        """
        # Normalize to unit norm
        seq1_norm = seq1 / (np.linalg.norm(seq1) + 1e-8)
        seq2_norm = seq2 / (np.linalg.norm(seq2) + 1e-8)
        
        # Compute shape distance
        return 1 - np.abs(np.dot(seq1_norm, seq2_norm))
    
    def spectral_similarity(self, power_spec1, power_spec2):
        """
        Compute similarity between power spectra
        """
        # Normalize
        ps1_norm = power_spec1 / (np.sum(power_spec1) + 1e-8)
        ps2_norm = power_spec2 / (np.sum(power_spec2) + 1e-8)
        
        # Compute various distances
        # 1. Euclidean distance
        euclidean_dist = np.sqrt(np.sum((ps1_norm - ps2_norm) ** 2))
        
        # 2. Cosine similarity
        cosine_sim = 1 - cosine(ps1_norm, ps2_norm)
        
        # 3. KL divergence (symmetrized)
        ps1_norm = np.clip(ps1_norm, 1e-10, 1)
        ps2_norm = np.clip(ps2_norm, 1e-10, 1)
        kl_div = 0.5 * (np.sum(ps1_norm * np.log(ps1_norm / ps2_norm)) + 
                        np.sum(ps2_norm * np.log(ps2_norm / ps1_norm)))
        
        return {
            'euclidean': euclidean_dist,
            'cosine': cosine_sim,
            'kl_divergence': kl_div
        }


class SignalProcessingReID:
    """
    Main class for signal processing-based person re-identification
    """
    def __init__(self):
        self.spectral_extractor = SpectralFeatureExtractor(fs=30)
        self.wavelet_extractor = WaveletFeatureExtractor()
        self.similarity_computer = TimeSeriesSimilarity()
        
    def extract_multivariate_features(self, sequence):
        """
        Extract features from multivariate time series
        sequence shape: (time_steps, features)
        """
        all_features = {}
        all_power_spectra = []
        
        # Extract features from each dimension
        for dim in range(sequence.shape[1]):
            signal_data = sequence[:, dim]
            
            # Spectral features
            spec_features, power_spectrum, frequencies = self.spectral_extractor.extract_all_features(signal_data)
            for key, val in spec_features.items():
                all_features[f'dim{dim}_{key}'] = val
            
            all_power_spectra.append(power_spectrum)
            
            # Wavelet features (if available)
            wavelet_features = self.wavelet_extractor.compute_wavelet_features(signal_data)
            for key, val in wavelet_features.items():
                all_features[f'dim{dim}_{key}'] = val
        
        # Aggregate power spectrum across dimensions
        avg_power_spectrum = np.mean(all_power_spectra, axis=0)
        
        return all_features, avg_power_spectrum, frequencies
    
    def compute_similarity_score(self, video_seq, imu_seq):
        """
        Compute comprehensive similarity score between video and IMU sequences
        Returns a similarity score (higher = more similar)
        """
        # Extract features
        video_features, video_spectrum, video_freqs = self.extract_multivariate_features(video_seq)
        imu_features, imu_spectrum, imu_freqs = self.extract_multivariate_features(imu_seq)
        
        # Convert features to vectors
        feature_keys = sorted(set(video_features.keys()) & set(imu_features.keys()))
        video_vector = np.array([video_features[k] for k in feature_keys])
        imu_vector = np.array([imu_features[k] for k in feature_keys])
        
        # Handle NaN/Inf
        video_vector = np.nan_to_num(video_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        imu_vector = np.nan_to_num(imu_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize feature vectors
        video_vector = (video_vector - np.mean(video_vector)) / (np.std(video_vector) + 1e-8)
        imu_vector = (imu_vector - np.mean(imu_vector)) / (np.std(imu_vector) + 1e-8)
        
        # Compute feature-based similarity
        feature_euclidean = euclidean(video_vector, imu_vector)
        feature_cosine = 1 - cosine(video_vector, imu_vector)
        
        # Compute spectral similarity
        spectral_sim = self.similarity_computer.spectral_similarity(video_spectrum, imu_spectrum)
        
        # Compute time-domain similarities on first principal component
        video_pc1 = np.mean(video_seq, axis=1)  # Average across features
        imu_pc1 = np.mean(imu_seq, axis=1)
        
        # DTW distance (normalize to [0,1] range)
        dtw_dist = self.similarity_computer.dynamic_time_warping(video_pc1, imu_pc1)
        dtw_dist_norm = dtw_dist / (len(video_pc1) * np.std(np.concatenate([video_pc1, imu_pc1])))
        
        # Cross-correlation
        cross_corr = self.similarity_computer.normalized_cross_correlation(video_pc1, imu_pc1)
        
        # NEW: Autocorrelation similarity (captures periodicity)
        autocorr_sim = self.similarity_computer.autocorrelation_similarity(video_pc1, imu_pc1)
        
        # NEW: Shape-based distance
        shape_dist = self.similarity_computer.shape_based_distance(video_pc1, imu_pc1)
        
        # Combine all measures into a single similarity score
        # Lower distances = higher similarity, so we invert them
        similarity_components = {
            'feature_cosine': feature_cosine,  # Higher is better
            'feature_euclidean': 1 / (1 + feature_euclidean),  # Invert to make higher better
            'spectral_cosine': spectral_sim['cosine'],  # Higher is better
            'spectral_euclidean': 1 / (1 + spectral_sim['euclidean']),  # Invert
            'kl_divergence': 1 / (1 + spectral_sim['kl_divergence']),  # Invert
            'dtw': 1 / (1 + dtw_dist_norm),  # Invert
            'cross_correlation': (cross_corr + 1) / 2,  # Map [-1,1] to [0,1]
            'autocorrelation': autocorr_sim,  # Already [0,1]
            'shape_similarity': 1 - shape_dist  # Invert distance to similarity
        }
        
        # Weighted combination (can be tuned)
        # Adjusted weights to include new features
        weights = {
            'feature_cosine': 0.12,
            'feature_euclidean': 0.12,
            'spectral_cosine': 0.18,
            'spectral_euclidean': 0.12,
            'kl_divergence': 0.08,
            'dtw': 0.13,
            'cross_correlation': 0.08,
            'autocorrelation': 0.10,
            'shape_similarity': 0.07
        }
        
        combined_score = sum(similarity_components[k] * weights[k] for k in weights.keys())
        
        return combined_score, similarity_components


def generate_synthetic_dataset(n_subjects=8, n_trials_per_subject=50, seq_length=90, n_video_features=51, n_imu_features=20):
    """
    Generate synthetic dataset for demonstration when real data unavailable
    """
    print("\n⚠️  Real dataset not found. Generating synthetic demo data...")
    print(f"   Subjects: {n_subjects}")
    print(f"   Trials per subject: {n_trials_per_subject}")
    print(f"   Sequence length: {seq_length} frames")
    
    np.random.seed(42)
    
    total_pairs = n_subjects * n_trials_per_subject * 2  # Positive + Negative
    
    video_pairs = []
    imu_pairs = []
    labels = []
    
    # Generate subject-specific patterns
    for subject_id in range(n_subjects):
        # Each subject has characteristic frequency patterns
        base_freq_video = 1.5 + subject_id * 0.1  # 1.5 to 2.2 Hz (stride frequency)
        base_freq_imu = base_freq_video + np.random.normal(0, 0.05)
        
        for trial in range(n_trials_per_subject):
            # Generate video data (pose keypoints)
            t = np.linspace(0, seq_length/30, seq_length)  # 30 fps
            video_data = np.zeros((seq_length, n_video_features))
            for feat_idx in range(n_video_features):
                # Gait-like periodic pattern + noise
                video_data[:, feat_idx] = (
                    np.sin(2 * np.pi * base_freq_video * t + feat_idx * 0.1) +
                    0.3 * np.sin(2 * np.pi * 2 * base_freq_video * t) +  # Harmonic
                    0.1 * np.random.randn(seq_length)
                )
            
            # Generate IMU data (quaternions)
            imu_data = np.zeros((seq_length, n_imu_features))
            for feat_idx in range(n_imu_features):
                imu_data[:, feat_idx] = (
                    np.sin(2 * np.pi * base_freq_imu * t + feat_idx * 0.15) +
                    0.3 * np.sin(2 * np.pi * 2 * base_freq_imu * t) +
                    0.12 * np.random.randn(seq_length)
                )
            
            # Positive pair (same subject)
            video_pairs.append(video_data)
            imu_pairs.append(imu_data)
            labels.append(1)
            
            # Negative pair (different subject)
            other_subject = (subject_id + np.random.randint(1, n_subjects)) % n_subjects
            other_freq = 1.5 + other_subject * 0.1
            
            imu_data_neg = np.zeros((seq_length, n_imu_features))
            for feat_idx in range(n_imu_features):
                imu_data_neg[:, feat_idx] = (
                    np.sin(2 * np.pi * other_freq * t + feat_idx * 0.15) +
                    0.3 * np.sin(2 * np.pi * 2 * other_freq * t) +
                    0.12 * np.random.randn(seq_length)
                )
            
            video_pairs.append(video_data)
            imu_pairs.append(imu_data_neg)
            labels.append(0)
    
    print(f"✓ Generated {len(labels)} synthetic pairs")
    
    return {
        'video_pairs': np.array(video_pairs),
        'imu_pairs': np.array(imu_pairs),
        'labels': np.array(labels)
    }


def evaluate_signal_processing_approach(npz_file=None, subset_size=2000):
    """
    Evaluate the signal processing approach and compare with ML metrics
    """
    print("=" * 80)
    print("SIGNAL PROCESSING-BASED PERSON RE-IDENTIFICATION")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    
    if npz_file and os.path.exists(npz_file):
        data = np.load(npz_file)
        print(f"   ✓ Loaded real dataset from: {npz_file}")
    else:
        if npz_file:
            print(f"   ✗ Dataset not found at: {npz_file}")
        data = generate_synthetic_dataset(n_subjects=8, n_trials_per_subject=min(25, subset_size//16))
    
    # Use subset for faster evaluation
    video_pairs = data['video_pairs'][:subset_size]
    imu_pairs = data['imu_pairs'][:subset_size]
    labels = data['labels'][:subset_size]
    
    print(f"   Loaded {len(labels)} pairs")
    print(f"   Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")
    
    # Initialize signal processing system
    print("\n[2/5] Initializing signal processing pipeline...")
    sp_reid = SignalProcessingReID()
    
    # Compute similarity scores
    print("\n[3/5] Computing similarity scores...")
    similarity_scores = []
    component_scores = {
        'feature_cosine': [],
        'feature_euclidean': [],
        'spectral_cosine': [],
        'spectral_euclidean': [],
        'kl_divergence': [],
        'dtw': [],
        'cross_correlation': []
    }
    
    for i in tqdm(range(len(labels)), desc="Processing pairs"):
        score, components = sp_reid.compute_similarity_score(video_pairs[i], imu_pairs[i])
        similarity_scores.append(score)
        
        for key in component_scores.keys():
            component_scores[key].append(components[key])
    
    similarity_scores = np.array(similarity_scores)
    
    # Convert similarity scores to distances for threshold finding
    # (Higher similarity = positive match, so we use negative for ROC)
    distances = -similarity_scores
    
    print("\n[4/5] Computing metrics...")
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (similarity_scores >= optimal_threshold).astype(int)
    accuracy = 100 * np.mean(predictions == labels)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Score statistics
    pos_scores = similarity_scores[labels == 1]
    neg_scores = similarity_scores[labels == 0]
    
    print(f"\n   === Performance Metrics (Threshold={optimal_threshold:.4f}) ===")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print(f"   Accuracy:  {accuracy:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n   === Similarity Score Statistics ===")
    print(f"   Positive pairs - Mean: {np.mean(pos_scores):.4f}, Std: {np.std(pos_scores):.4f}")
    print(f"   Negative pairs - Mean: {np.mean(neg_scores):.4f}, Std: {np.std(neg_scores):.4f}")
    print(f"   Separation: {np.mean(pos_scores) - np.mean(neg_scores):.4f}")
    
    # Analyze component contributions
    print(f"\n   === Component Contribution Analysis ===")
    for component_name, scores in component_scores.items():
        scores_array = np.array(scores)
        pos_component = scores_array[labels == 1]
        neg_component = scores_array[labels == 0]
        separation = np.mean(pos_component) - np.mean(neg_component)
        print(f"   {component_name:20s}: Separation = {separation:.4f}")
    
    print("\n[5/5] Creating visualizations...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: ROC Curve
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5,
                label=f'Optimal (TPR={tpr[optimal_idx]:.3f})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Score Distribution
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(pos_scores, bins=50, alpha=0.6, color='blue', label=f'Positive (n={len(pos_scores)})', density=True)
    ax2.hist(neg_scores, bins=50, alpha=0.6, color='red', label=f'Negative (n={len(neg_scores)})', density=True)
    ax2.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={optimal_threshold:.3f}')
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    ax3 = plt.subplot(2, 4, 3)
    im = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Negative', 'Positive'])
    ax3.set_yticklabels(['Negative', 'Positive'])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix\n(Accuracy: {accuracy:.2f}%)')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Component Contributions
    ax4 = plt.subplot(2, 4, 4)
    component_names = list(component_scores.keys())
    separations = []
    for name in component_names:
        scores_array = np.array(component_scores[name])
        pos = scores_array[labels == 1]
        neg = scores_array[labels == 0]
        separations.append(np.mean(pos) - np.mean(neg))
    
    colors_comp = ['green' if s > 0 else 'red' for s in separations]
    bars = ax4.barh(component_names, separations, color=colors_comp, alpha=0.6)
    ax4.set_xlabel('Mean Separation (Pos - Neg)')
    ax4.set_title('Component Discriminative Power')
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Precision-Recall vs Threshold
    ax5 = plt.subplot(2, 4, 5)
    test_thresholds = np.linspace(np.min(similarity_scores), np.max(similarity_scores), 100)
    precisions_list = []
    recalls_list = []
    for thresh in test_thresholds:
        preds = (similarity_scores >= thresh).astype(int)
        cm_temp = confusion_matrix(labels, preds)
        tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
        prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precisions_list.append(prec)
        recalls_list.append(rec)
    
    ax5.plot(test_thresholds, precisions_list, label='Precision', color='blue')
    ax5.plot(test_thresholds, recalls_list, label='Recall', color='red')
    ax5.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label='Optimal')
    ax5.set_xlabel('Threshold')
    ax5.set_ylabel('Score')
    ax5.set_title('Precision & Recall vs Threshold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Score vs Pair Index
    ax6 = plt.subplot(2, 4, 6)
    pos_mask = labels == 1
    neg_mask = labels == 0
    ax6.scatter(np.where(pos_mask)[0], similarity_scores[pos_mask], c='blue', alpha=0.3, s=10, label='Positive')
    ax6.scatter(np.where(neg_mask)[0], similarity_scores[neg_mask], c='red', alpha=0.3, s=10, label='Negative')
    ax6.axhline(optimal_threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax6.set_xlabel('Pair Index')
    ax6.set_ylabel('Similarity Score')
    ax6.set_title('Similarity Score vs Pair Index')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Component Box Plot
    ax7 = plt.subplot(2, 4, 7)
    component_data_pos = [np.array(component_scores[name])[labels == 1] for name in component_names]
    bp = ax7.boxplot(component_data_pos, labels=[name.replace('_', '\n') for name in component_names], 
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax7.set_ylabel('Score Value')
    ax7.set_title('Component Scores (Positive Pairs)')
    ax7.tick_params(axis='x', rotation=45, labelsize=8)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Method Comparison Summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    summary_text = f"""
    SIGNAL PROCESSING APPROACH
    ═══════════════════════════
    
    ROC-AUC:        {roc_auc:.4f}
    Accuracy:       {accuracy:.2f}%
    Precision:      {precision:.4f}
    Recall:         {recall:.4f}
    F1-Score:       {f1:.4f}
    
    Optimal Threshold: {optimal_threshold:.4f}
    
    True Positives:  {tp}
    True Negatives:  {tn}
    False Positives: {fp}
    False Negatives: {fn}
    
    ═══════════════════════════
    Features Used:
    • Spectral Centroid
    • Spectral Spread/Entropy
    • Dominant Frequencies
    • Power Spectrum Analysis
    • DTW Distance
    • Cross-Correlation
    • Wavelet Features
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace')
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/signal_processing_evaluation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")
    
    # Save detailed results
    results = {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm.tolist(),
        'component_separations': {name: float(np.mean(np.array(component_scores[name])[labels == 1]) - 
                                               np.mean(np.array(component_scores[name])[labels == 0]))
                                  for name in component_names}
    }
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_csv_path = f"{OUTPUT_DIR}/signal_processing_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"   Saved results to: {results_csv_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if real dataset exists
    if os.path.exists(NPZ_FILE):
        print(f"\n✓ Found real dataset: {NPZ_FILE}")
        results = evaluate_signal_processing_approach(NPZ_FILE, subset_size=2000)
    else:
        print(f"\n⚠️  Real dataset not found at: {NPZ_FILE}")
        print("   Running in DEMO MODE with synthetic data...")
        results = evaluate_signal_processing_approach(None, subset_size=800)