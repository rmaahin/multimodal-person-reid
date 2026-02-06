import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import torch.nn.functional as F
import math
import seaborn as sns

# --- CONFIGURATION ---
NPZ_FILE = "/mnt/multimodal-person-reid/dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "/mnt/multimodal-person-reid/trial-3/best_model_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.2

print(f"Using device: {DEVICE}")

# --- MODEL DEFINITION (Must match training) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SOTATransformerBranch(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=DROPOUT, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        return x

class CrossModalNetwork(nn.Module):
    def __init__(self, video_dim, imu_dim):
        super().__init__()
        
        transformer_dim = 256
        embedding_dim = 128
        
        self.video_branch = SOTATransformerBranch(video_dim, model_dim=transformer_dim)
        self.imu_branch = SOTATransformerBranch(imu_dim, model_dim=transformer_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, video_seq, imu_seq):
        v_feat = self.video_branch(video_seq)
        i_feat = self.imu_branch(imu_seq)
        
        v_emb = self.projector(v_feat)
        i_emb = self.projector(i_feat)
        
        return F.normalize(v_emb, p=2, dim=1), F.normalize(i_emb, p=2, dim=1)

def compute_derivatives(sequence):
    if isinstance(sequence, np.ndarray): 
        sequence = torch.from_numpy(sequence).float()
    vel = torch.zeros_like(sequence)
    vel[1:] = sequence[1:] - sequence[:-1]
    acc = torch.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return torch.cat([sequence, vel, acc], dim=1)

# --- EVALUATION FUNCTION ---
def evaluate():
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading dataset...")
    data = np.load(NPZ_FILE)
    
    # Use subset for faster evaluation (adjust as needed)
    SUBSET_SIZE = 2000
    vid_pairs = data['video_pairs'][:SUBSET_SIZE]
    imu_pairs = data['imu_pairs'][:SUBSET_SIZE]
    labels = data['labels'][:SUBSET_SIZE]

    print(f"   Loaded {len(labels)} pairs")
    print(f"   Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")

    # Process features
    print("\n[2/6] Computing derivatives (velocity & acceleration)...")
    proc_vid = torch.stack([compute_derivatives(v) for v in vid_pairs])
    proc_imu = torch.stack([compute_derivatives(i) for i in imu_pairs])
    labels_tensor = torch.from_numpy(labels)

    # Load model
    print("\n[3/6] Loading model...")
    model = CrossModalNetwork(proc_vid.shape[2], proc_imu.shape[2]).to(DEVICE)
    
    # FIXED: PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_threshold = checkpoint.get('threshold', None)
        saved_auc = checkpoint.get('auc', None)
        saved_margin = checkpoint.get('margin', None)
        print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if saved_threshold:
            print(f"   Saved threshold: {saved_threshold:.4f}")
        if saved_auc:
            print(f"   Saved AUC: {saved_auc:.4f}")
        if saved_margin:
            print(f"   Training margin: {saved_margin:.2f}")
    else:
        model.load_state_dict(checkpoint)
        saved_threshold = None
    
    model.eval()

    # Run inference
    print("\n[4/6] Running inference...")
    with torch.no_grad():
        v_emb, i_emb = model(proc_vid.to(DEVICE), proc_imu.to(DEVICE))
        dists = F.pairwise_distance(v_emb, i_emb).cpu().numpy()

    v_emb_np = v_emb.cpu().numpy()
    i_emb_np = i_emb.cpu().numpy()

    # --- METRIC 1: ROC CURVE & OPTIMAL THRESHOLD ---
    print("\n[5/6] Computing metrics...")
    
    # FIXED: Use negative distances as scores (lower distance = positive class)
    scores = -dists
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]  # Convert back to distance
    
    print(f"\n   ROC-AUC Score: {roc_auc:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    if saved_threshold:
        print(f"   Saved Threshold: {saved_threshold:.4f}")
        print(f"   Threshold Difference: {abs(optimal_threshold - saved_threshold):.4f}")
    
    # Calculate metrics at optimal threshold
    predictions = (dists < optimal_threshold).astype(int)
    accuracy = 100 * np.mean(predictions == labels)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   === Performance Metrics (Threshold={optimal_threshold:.4f}) ===")
    print(f"   Accuracy:  {accuracy:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Distance statistics
    pos_dists = dists[labels == 1]
    neg_dists = dists[labels == 0]
    
    print(f"\n   === Distance Statistics ===")
    print(f"   Positive pairs - Mean: {np.mean(pos_dists):.4f}, Std: {np.std(pos_dists):.4f}")
    print(f"   Negative pairs - Mean: {np.mean(neg_dists):.4f}, Std: {np.std(neg_dists):.4f}")
    print(f"   Separation: {np.mean(neg_dists) - np.mean(pos_dists):.4f}")
    
    # --- VISUALIZATION ---
    print("\n[6/6] Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5, 
                label=f'Optimal (TPR={tpr[optimal_idx]:.3f})')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(pos_dists, bins=50, alpha=0.6, color='blue', label=f'Positive (n={len(pos_dists)})', density=True)
    ax2.hist(neg_dists, bins=50, alpha=0.6, color='red', label=f'Negative (n={len(neg_dists)})', density=True)
    ax2.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={optimal_threshold:.3f}')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title(f'Confusion Matrix\n(Accuracy: {accuracy:.2f}%)')
    
    # Plot 4: Precision-Recall-Threshold Curve
    ax4 = plt.subplot(2, 3, 4)
    # Calculate precision and recall for different thresholds
    precisions = []
    recalls = []
    test_thresholds = np.linspace(0, 2, 100)
    for thresh in test_thresholds:
        preds = (dists < thresh).astype(int)
        cm_temp = confusion_matrix(labels, preds)
        tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
        prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
    
    ax4.plot(test_thresholds, precisions, label='Precision', color='blue')
    ax4.plot(test_thresholds, recalls, label='Recall', color='red')
    ax4.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label='Optimal')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Precision & Recall vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: t-SNE Visualization (2D)
    ax5 = plt.subplot(2, 3, 5)
    print("   Computing 2D t-SNE...")
    
    # Use positive pairs for visualization
    pos_indices = np.where(labels == 1)[0][:100]  # First 100 positive pairs
    
    v_subset = v_emb_np[pos_indices]
    i_subset = i_emb_np[pos_indices]
    
    combined = np.concatenate([v_subset, i_subset], axis=0)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
    results_2d = tsne_2d.fit_transform(combined)
    
    v_tsne = results_2d[:len(v_subset)]
    i_tsne = results_2d[len(v_subset):]
    
    ax5.scatter(v_tsne[:, 0], v_tsne[:, 1], c='blue', alpha=0.6, s=50, label='Video', marker='o')
    ax5.scatter(i_tsne[:, 0], i_tsne[:, 1], c='red', alpha=0.6, s=50, label='IMU', marker='^')
    
    # Draw connecting lines
    for j in range(len(v_subset)):
        ax5.plot([v_tsne[j, 0], i_tsne[j, 0]], 
                [v_tsne[j, 1], i_tsne[j, 1]], 
                'gray', alpha=0.2, linewidth=0.5)
    
    ax5.set_xlabel('t-SNE Dimension 1')
    ax5.set_ylabel('t-SNE Dimension 2')
    ax5.set_title('t-SNE Embedding (100 Positive Pairs)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Distance vs Pair Index
    ax6 = plt.subplot(2, 3, 6)
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    ax6.scatter(np.where(pos_mask)[0], dists[pos_mask], c='blue', alpha=0.3, s=10, label='Positive')
    ax6.scatter(np.where(neg_mask)[0], dists[neg_mask], c='red', alpha=0.3, s=10, label='Negative')
    ax6.axhline(optimal_threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax6.set_xlabel('Pair Index')
    ax6.set_ylabel('Distance')
    ax6.set_title('Distance vs Pair Index')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/mnt/multimodal-person-reid/trial-3/evaluation_comprehensive_corrected.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: evaluation_comprehensive_corrected.png")
    
    # --- SUMMARY REPORT ---
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"True Positives:    {tp}")
    print(f"True Negatives:    {tn}")
    print(f"False Positives:   {fp}")
    print(f"False Negatives:   {fn}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()