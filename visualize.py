import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import math

# --- CONFIGURATION ---
NPZ_FILE = "dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "/person-reid/multimodal-person-reid/trial-2/best_sota_model_v2.pth"
TRIAL_DIR = "trial-2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_TO_PLOT = 600  # Number of positive pairs to visualize (Don't set too high or plot gets messy)

# --- RE-DEFINE MODEL CLASSES (Must match training exactly) ---
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

# --- MAIN VISUALIZATION ---
def visualize_3d():
    print("Loading data...")
    data = np.load(NPZ_FILE)
    
    # Filter for POSITIVE pairs only (Label = 1)
    # These represent (Subject X Video, Subject X IMU)
    # Since the file is ordered by subject, indices correlate to Subject Identity
    indices = np.where(data['labels'] == 1)[0]
    
    # Subsample to keep the plot readable
    if len(indices) > SAMPLES_TO_PLOT:
        # Take evenly spaced samples to cover all subjects
        subset_indices = np.linspace(0, len(indices)-1, SAMPLES_TO_PLOT, dtype=int)
        indices = indices[subset_indices]
    
    vid_pairs = data['video_pairs'][indices]
    imu_pairs = data['imu_pairs'][indices]
    
    print(f"Processing {len(indices)} positive pairs...")
    
    # Feature Engineering
    proc_vid = torch.stack([compute_derivatives(v) for v in vid_pairs])
    proc_imu = torch.stack([compute_derivatives(i) for i in imu_pairs])

    # Load Model
    model = CrossModalNetwork(proc_vid.shape[2], proc_imu.shape[2]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Generating Embeddings...")
    with torch.no_grad():
        v_emb, i_emb = model(proc_vid.to(DEVICE), proc_imu.to(DEVICE))
        v_emb = v_emb.cpu().numpy()
        i_emb = i_emb.cpu().numpy()

    print("Running 3D t-SNE (this may take a minute)...")
    # Concatenate to run t-SNE on the shared space
    combined_emb = np.concatenate([v_emb, i_emb], axis=0)
    
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(combined_emb)
    
    # Split back
    v_tsne = tsne_results[:len(v_emb)]
    i_tsne = tsne_results[len(v_emb):]

    # --- PLOTTING ---
    print("Plotting 3D Graph...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a color map based on index (proxy for Subject ID)
    # S40 will be one color, S57 will be another
    colors = np.linspace(0, 1, len(v_tsne))
    
    # Plot Video Embeddings (Circles)
    scatter_v = ax.scatter(v_tsne[:,0], v_tsne[:,1], v_tsne[:,2], 
                           c=colors, cmap='jet', marker='o', s=40, alpha=0.6, label='Video')
    
    # Plot IMU Embeddings (Triangles)
    scatter_i = ax.scatter(i_tsne[:,0], i_tsne[:,1], i_tsne[:,2], 
                           c=colors, cmap='jet', marker='^', s=40, alpha=0.6, label='IMU')

    # Draw lines connecting the Positive Pairs
    # If the model works, these lines should be SHORT
    for k in range(len(v_tsne)):
        ax.plot([v_tsne[k,0], i_tsne[k,0]], 
                [v_tsne[k,1], i_tsne[k,1]], 
                [v_tsne[k,2], i_tsne[k,2]], 
                color='gray', alpha=0.2, linewidth=0.5)

    ax.set_title(f"3D Visualization of {len(indices)} Positive Pairs\n(Color = Subject Identity)", fontsize=14)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    
    # Add a color bar to indicate Subject progression
    cbar = plt.colorbar(scatter_v, ax=ax, pad=0.1)
    cbar.set_label('Subject ID (Start -> End)')
    
    plt.legend()
    plt.tight_layout()
    
    output_file = os.path.join(TRIAL_PATH, "3d_clustering_visualization.png")
    plt.savefig(output_file, dpi=300)
    print(f"✅ 3D Plot saved to {output_file}")

if __name__ == "__main__":
    visualize_3d()