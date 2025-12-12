import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch.nn.functional as F
import math

# --- CONFIGURATION ---
NPZ_FILE = "dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "best-model.pth"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RE-DEFINE MODEL (Must match training script exactly) ---
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
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=6, output_dim=128):
        super().__init__()
        
        # Hardcoded dropout to match training config
        dropout = 0.2 
        
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- THE FIX IS HERE ---
        # We added nn.Dropout(dropout) to match the training script
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),       # <--- This layer was missing
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        embedding = self.fc(x)
        return F.normalize(embedding, p=2, dim=1)

class CrossModalNetwork(nn.Module):
    def __init__(self, video_dim, imu_dim):
        super().__init__()
        self.video_branch = SOTATransformerBranch(video_dim, output_dim=128)
        self.imu_branch = SOTATransformerBranch(imu_dim, output_dim=128)
    def forward(self, video_seq, imu_seq):
        return self.video_branch(video_seq), self.imu_branch(imu_seq)

def compute_derivatives(sequence):
    if isinstance(sequence, np.ndarray): sequence = torch.from_numpy(sequence).float()
    vel = torch.zeros_like(sequence); vel[1:] = sequence[1:] - sequence[:-1]
    acc = torch.zeros_like(vel); acc[1:] = vel[1:] - vel[:-1]
    return torch.cat([sequence, vel, acc], dim=1)

# --- MAIN EVALUATION ---
def evaluate():
    print("Loading data and model...")
    data = np.load(NPZ_FILE)
    
    # Use only a subset for visualization (too many points makes t-SNE messy)
    # We take the first 1000 pairs
    SUBSET_SIZE = 1000
    vid_pairs = data['video_pairs'][:SUBSET_SIZE]
    imu_pairs = data['imu_pairs'][:SUBSET_SIZE]
    labels = data['labels'][:SUBSET_SIZE]

    # Process Features
    proc_vid = torch.stack([compute_derivatives(v) for v in vid_pairs])
    proc_imu = torch.stack([compute_derivatives(i) for i in imu_pairs])
    labels = torch.from_numpy(labels)

    # Initialize Model
    model = CrossModalNetwork(proc_vid.shape[2], proc_imu.shape[2]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Running Inference...")
    with torch.no_grad():
        v_emb, i_emb = model(proc_vid.to(DEVICE), proc_imu.to(DEVICE))
        
        # Calculate Euclidean Distances
        dists = F.pairwise_distance(v_emb, i_emb).cpu().numpy()

    # --- METRIC 1: ROC CURVE ---
    # Invert distances because ROC expects "Higher Score = Positive"
    # Currently Dist 0 = Positive. So we use (2.0 - Distance) as the score.
    scores = 2.0 - dists
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # --- METRIC 2: t-SNE VISUALIZATION ---
    print("Computing t-SNE (this takes a moment)...")
    
    # We want to plot ONLY the Positive pairs to see if Video and IMU align
    pos_indices = np.where(labels == 1)[0]
    
    # We pick the first 50 positive pairs to plot (100 dots total)
    # Ideally, Dot i (Video) should be close to Dot i (IMU)
    limit = 50
    subset_idx = pos_indices[:limit]
    
    v_emb_subset = v_emb.cpu().numpy()[subset_idx]
    i_emb_subset = i_emb.cpu().numpy()[subset_idx]
    
    # Combine for t-SNE
    combined = np.concatenate([v_emb_subset, i_emb_subset], axis=0)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    results = tsne.fit_transform(combined)
    
    v_tsne = results[:limit]
    i_tsne = results[limit:]
    
    plt.subplot(1, 2, 2)
    plt.title(f"t-SNE of {limit} Positive Pairs")
    
    # Plot dots
    plt.scatter(v_tsne[:,0], v_tsne[:,1], c='blue', label='Video', alpha=0.6)
    plt.scatter(i_tsne[:,0], i_tsne[:,1], c='red', label='IMU', alpha=0.6)
    
    # Draw lines connecting the pairs
    for i in range(limit):
        plt.plot([v_tsne[i,0], i_tsne[i,0]], [v_tsne[i,1], i_tsne[i,1]], 'k-', alpha=0.2)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plots.png")
    print("Plots saved to evaluation_plots.png")
    print(f"Final ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate()