import numpy as np
import pandas as pd
import glob
import os
from scipy.fft import fft, fftfreq
from tqdm import tqdm
import time

# --- CONFIGURATION ---
DATASET_DIR = "/mnt/dataset-synchronization/A01-synced-dataset"
FPS = 30.0
MAX_FREQ_HZ = 10.0
COMMON_GRID_SIZE = 100 

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        is_header = df.columns.astype(str).str.contains('[a-zA-Z]').any()
        if is_header:
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df.values
        else:
            df = pd.read_csv(filepath, header=None)
            return df.values
    except:
        return None

def get_spectral_fingerprint(data):
    try:
        data = np.array(data, dtype=float)
        if len(data.shape) > 1:
            velocity = np.diff(data, axis=0)
            energy = np.linalg.norm(velocity, axis=1)
        else:
            energy = np.diff(data)

        window = np.hanning(len(energy))
        energy = energy * window
        
        N = len(energy)
        if N < 10: return np.zeros(COMMON_GRID_SIZE)
        
        spectrum = np.abs(fft(energy)[0:N//2])
        freqs = fftfreq(N, 1/FPS)[0:N//2]
        
        common_freqs = np.linspace(0, MAX_FREQ_HZ, COMMON_GRID_SIZE)
        fixed_spectrum = np.interp(common_freqs, freqs, spectrum)
        
        norm = np.linalg.norm(fixed_spectrum)
        if norm > 0: fixed_spectrum = fixed_spectrum / norm
            
        return fixed_spectrum
    except:
        return np.zeros(COMMON_GRID_SIZE)

def build_database():
    print(f"--- 🎵 Building Spectral Database ---")
    
    subject_dirs = sorted([d for d in glob.glob(os.path.join(DATASET_DIR, "S*")) if os.path.isdir(d)])
    
    video_fps = []
    imu_fps = []
    labels = []
    filenames = []
    
    for subj_dir in tqdm(subject_dirs):
        subj_id = os.path.basename(subj_dir)
        csv_files = glob.glob(os.path.join(subj_dir, "*.csv"))
        
        for vid_path in csv_files:
            filename = os.path.basename(vid_path)
            base_name = os.path.splitext(filename)[0]
            
            # Find matching IMU
            imu_path = os.path.join(subj_dir, base_name + ".raw")
            if not os.path.exists(imu_path):
                imu_path = os.path.join(subj_dir, base_name + ".txt")
                if not os.path.exists(imu_path): continue 
            
            vid_data = load_data(vid_path)
            imu_data = load_data(imu_path)
            
            if vid_data is None or imu_data is None: continue
            
            vid_fp = get_spectral_fingerprint(vid_data)
            imu_fp = get_spectral_fingerprint(imu_data)
            
            if np.all(vid_fp == 0) or np.all(imu_fp == 0): continue
            
            video_fps.append(vid_fp)
            imu_fps.append(imu_fp)
            labels.append(subj_id)
            filenames.append(filename)

    return np.array(video_fps), np.array(imu_fps), np.array(labels), np.array(filenames)

def main():
    # 1. Build Database
    X_video, X_imu, y_labels, filenames = build_database()
    
    if len(X_video) == 0:
        print("❌ Database empty. Check dataset path.")
        return

    print(f"\nDatabase Ready: {len(X_video)} pairs indexed.")

    # 2. Evaluate Global Accuracy
    dists = np.linalg.norm(X_video[:, None, :] - X_imu[None, :, :], axis=2)
    rank1 = 0
    for i in range(len(y_labels)):
        # Distance from this IMU (i) to all Videos
        # Note: We match IMU query against Video Database
        # dists[row, col] -> row=video, col=imu
        # So for IMU i, we look at column i across all rows
        query_dists = dists[:, i]
        closest_vid_idx = np.argmin(query_dists)
        if y_labels[closest_vid_idx] == y_labels[i]:
            rank1 += 1
            
    print(f"Global Rank-1 Accuracy: {(rank1/len(y_labels))*100:.2f}%\n")

    # 3. Interactive Loop
    while True:
        print("-" * 50)
        print("Enter Subject ID (e.g., S40) to test.")
        print("Type 'q' to quit.")
        user_input = input(">> ").strip()
        
        if user_input.lower() == 'q':
            break
            
        # Find all IMU samples for this subject
        indices = [i for i, label in enumerate(y_labels) if label == user_input]
        
        if not indices:
            print(f"❌ Subject '{user_input}' not found in database.")
            print(f"Available subjects: {np.unique(y_labels)}")
            continue
            
        print(f"\nFound {len(indices)} IMU recordings for {user_input}.")
        
        for idx in indices:
            print(f"\n📝 Testing Trial: {filenames[idx]}")
            
            # Get the IMU fingerprint for this specific trial
            query_imu = X_imu[idx]
            
            # Compare against ALL videos in the database
            # Calculate distance between this IMU and every Video
            # X_video shape: (N, 100), query_imu shape: (100,)
            distances = np.linalg.norm(X_video - query_imu, axis=1)
            
            # Find closest match
            best_match_idx = np.argmin(distances)
            
            predicted_subject = y_labels[best_match_idx]
            predicted_file = filenames[best_match_idx]
            match_dist = distances[best_match_idx]
            
            print(f"   • Prediction:  {predicted_subject}  (File: {predicted_file})")
            print(f"   • Ground Truth: {user_input}")
            print(f"   • Distance: {match_dist:.4f}")
            
            if predicted_subject == user_input:
                print("   ✅ MATCH SUCCESS")
            else:
                print("   ❌ MATCH FAILED")

if __name__ == "__main__":
    main()