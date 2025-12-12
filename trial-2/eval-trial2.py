import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import math

# --- CONFIGURATION ---
NPZ_FILE = "/person-reid/multimodal-person-reid/dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "best_sota_model_v2.pth" # Ensure this is your V2 model file
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DROPOUT = 0.2 

# --- RE-DEFINE MODEL (Matches train_sota_v2.py) ---

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
        
        # NOTE: In V2, the individual branch has NO final FC layer.
        # It outputs raw transformer features.

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        return x

class CrossModalNetwork(nn.Module):
    def __init__(self, video_dim, imu_dim):
        super().__init__()
        
        # Internal dimensions matching training script
        transformer_dim = 256
        embedding_dim = 128
        
        self.video_branch = SOTATransformerBranch(video_dim, model_dim=transformer_dim)
        self.imu_branch = SOTATransformerBranch(imu_dim, model_dim=transformer_dim)
        
        # SHARED PROJECTOR (The key fix for V2)
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
    if isinstance(sequence, np.ndarray): sequence = torch.from_numpy(sequence).float()
    vel = torch.zeros_like(sequence); vel[1:] = sequence[1:] - sequence[:-1]
    acc = torch.zeros_like(vel); acc[1:] = vel[1:] - vel[:-1]
    return torch.cat([sequence, vel, acc], dim=1)

# --- MAIN EVALUATION ---
def evaluate():
    print("Loading data and model...")
    data = np.load(NPZ_FILE)
    
    # Use only a subset for visualization
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running Inference...")
    with torch.no_grad():
        v_emb, i_emb = model(proc_vid.to(DEVICE), proc_imu.to(DEVICE))
        dists = F.pairwise_distance(v_emb, i_emb).cpu().numpy()

    # --- METRIC 1: ROC CURVE ---
    # With Margin 2.0, distances can range from 0 to >2.0.
    # We use 4.0 as an arbitrary max to invert the score.
    scores = 4.0 - dists
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # --- METRIC 2: t-SNE VISUALIZATION ---
    print("Computing t-SNE (this takes a moment)...")
    
    pos_indices = np.where(labels == 1)[0]
    limit = 50
    subset_idx = pos_indices[:limit]
    
    v_emb_subset = v_emb.cpu().numpy()[subset_idx]
    i_emb_subset = i_emb.cpu().numpy()[subset_idx]
    
    combined = np.concatenate([v_emb_subset, i_emb_subset], axis=0)
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    results = tsne.fit_transform(combined)
    
    v_tsne = results[:limit]
    i_tsne = results[limit:]
    
    plt.subplot(1, 2, 2)
    plt.title(f"t-SNE of {limit} Positive Pairs")
    
    plt.scatter(v_tsne[:,0], v_tsne[:,1], c='blue', label='Video', alpha=0.6)
    plt.scatter(i_tsne[:,0], i_tsne[:,1], c='red', label='IMU', alpha=0.6)
    
    for i in range(limit):
        plt.plot([v_tsne[i,0], i_tsne[i,0]], [v_tsne[i,1], i_tsne[i,1]], 'k-', alpha=0.2)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plots_v2.png")
    print("Plots saved to evaluation_plots_v2.png")
    print(f"Final ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate()