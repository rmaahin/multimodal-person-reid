import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import math

# --- CONFIGURATION ---
NPZ_FILE = "/person-reid/multimodal-person-reid/dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "best_sota_model_v2.pth"  # Make sure this matches your saved file name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_TO_PLOT = 600
DROPOUT = 0.2  # Must match training parameter for architecture definition

# --- RE-DEFINE MODEL CLASSES (MATCHING train_sota_v2.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :]

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
        
        # NOTE: Final projection removed from branch, just like in training script

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
        
        # SHARED PROJECTOR (This was missing in your old script)
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

# --- MAIN VISUALIZATION ---
def visualize_3d():
    print("Loading data...")
    data = np.load(NPZ_FILE)
    
    indices = np.where(data['labels'] == 1)[0]
    if len(indices) > SAMPLES_TO_PLOT:
        subset_indices = np.linspace(0, len(indices)-1, SAMPLES_TO_PLOT, dtype=int)
        indices = indices[subset_indices]
    
    vid_pairs = data['video_pairs'][indices]
    imu_pairs = data['imu_pairs'][indices]
    
    proc_vid = torch.stack([compute_derivatives(v) for v in vid_pairs])
    proc_imu = torch.stack([compute_derivatives(i) for i in imu_pairs])

    print("Loading Model...")
    model = CrossModalNetwork(proc_vid.shape[2], proc_imu.shape[2]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Generating Embeddings...")
    with torch.no_grad():
        v_emb, i_emb = model(proc_vid.to(DEVICE), proc_imu.to(DEVICE))
        v_emb = v_emb.cpu().numpy()
        i_emb = i_emb.cpu().numpy()

    print("Running 3D t-SNE...")
    combined_emb = np.concatenate([v_emb, i_emb], axis=0)
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(combined_emb)
    
    v_tsne = tsne_results[:len(v_emb)]
    i_tsne = tsne_results[len(v_emb):]

    print("Plotting...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = np.linspace(0, 1, len(v_tsne))
    
    scatter_v = ax.scatter(v_tsne[:,0], v_tsne[:,1], v_tsne[:,2], c=colors, cmap='jet', marker='o', s=40, alpha=0.6, label='Video')
    scatter_i = ax.scatter(i_tsne[:,0], i_tsne[:,1], i_tsne[:,2], c=colors, cmap='jet', marker='^', s=40, alpha=0.6, label='IMU')

    for k in range(len(v_tsne)):
        ax.plot([v_tsne[k,0], i_tsne[k,0]], [v_tsne[k,1], i_tsne[k,1]], [v_tsne[k,2], i_tsne[k,2]], color='gray', alpha=0.2, linewidth=0.5)

    ax.set_title(f"3D Visualization (Shared Projector Model)", fontsize=14)
    plt.legend()
    plt.savefig("3d_clustering_v2.png", dpi=300)
    print("✅ Saved to 3d_clustering_v2.png")

if __name__ == "__main__":
    visualize_3d()