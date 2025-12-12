import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
NPZ_FILE = "dataset-augmentation/siamese_A01_augmented.npz"

# HYPERPARAMETERS FOR RTX 4090
BATCH_SIZE = 512          # Massive batch size for effective Triplet Mining
LEARNING_RATE = 1e-4      # Lower LR for stability with large batches
EPOCHS = 100              # Transformers need time to converge
EMBEDDING_DIM = 128       # Larger embedding space for richer features
DROPOUT = 0.2
WARMUP_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on {DEVICE} with Batch Size {BATCH_SIZE}")

# --- 2. ADVANCED FEATURE ENGINEERING ---
def compute_derivatives(sequence):
    """
    Computes Velocity (1st deriv) and Acceleration (2nd deriv).
    Input: (Seq_Len, Features) -> Output: (Seq_Len, Features * 3)
    """
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).float()
    
    velocity = torch.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    
    acceleration = torch.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    return torch.cat([sequence, velocity, acceleration], dim=1)

# --- 3. DATASET CLASS ---
class VidimuSOTADataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading dataset...")
        data = np.load(npz_path)
        
        self.video_pairs = data['video_pairs']
        self.imu_pairs = data['imu_pairs']
        self.labels = torch.from_numpy(data['labels']).float()
        
        print("Feature Engineering: Computing Velocity & Acceleration...")
        
        self.processed_video = []
        self.processed_imu = []
        
        for i in range(len(self.labels)):
            self.processed_video.append(compute_derivatives(self.video_pairs[i]))
            self.processed_imu.append(compute_derivatives(self.imu_pairs[i]))
            
        self.processed_video = torch.stack(self.processed_video)
        self.processed_imu = torch.stack(self.processed_imu)
        
        print(f"Dataset Ready. Shape: {self.processed_video.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # We return the indices too, which can be useful for debugging
        return self.processed_video[idx], self.processed_imu[idx], self.labels[idx]

# --- 4. MODEL: DEEP TRANSFORMER ENCODER ---
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
        
        # Robust Input Normalization
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # Deeper Transformer (6 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=DROPOUT, 
            batch_first=True,
            norm_first=True # Pre-Norm is generally more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Aggregation & Projection
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        
        # BatchNorm on Input Features
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        # Final Projection
        embedding = self.fc(x)
        
        # L2 Normalize embeddings (Critical for Triplet/Contrastive loss)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

class CrossModalNetwork(nn.Module):
    def __init__(self, video_dim, imu_dim):
        super().__init__()
        self.video_branch = SOTATransformerBranch(video_dim, output_dim=EMBEDDING_DIM)
        self.imu_branch = SOTATransformerBranch(imu_dim, output_dim=EMBEDDING_DIM)

    def forward(self, video_seq, imu_seq):
        v_emb = self.video_branch(video_seq)
        i_emb = self.imu_branch(imu_seq)
        return v_emb, i_emb

# --- 5. LOSS: CONTRASTIVE LOSS (With Hard Mining Support) ---
# Standard Contrastive Loss is safer for 'pairs'. 
# Triplet requires (Anchor, Pos, Neg) structure which our DataLoader doesn't inherently enforce perfectly unless custom sampled.
# Given the dataset structure (Pairs), we stick to a robust Contrastive Loss but with high margin.

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Loss
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 6. TRAINING ENGINE ---
def train():
    # 1. Load Data
    full_dataset = VidimuSOTADataset(NPZ_FILE)
    
    # 90/10 Split (More training data)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Num_workers=4 to feed the GPU fast enough
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Setup Model
    sample_v, sample_i, _ = full_dataset[0]
    model = CrossModalNetwork(sample_v.shape[1], sample_i.shape[1]).to(DEVICE)
    
    # 3. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=1.2) # Higher margin pushes negatives further apart
    
    # Warmup + Cosine Decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.1 # 10% Warmup
    )

    print("\n🔥 Starting High-Performance Training...")
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_v, batch_i, batch_lbl in train_loader:
            batch_v, batch_i, batch_lbl = batch_v.to(DEVICE), batch_i.to(DEVICE), batch_lbl.to(DEVICE)
            
            optimizer.zero_grad()
            v_emb, i_emb = model(batch_v, batch_i)
            
            loss = criterion(v_emb, i_emb, batch_lbl)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()

        # Validation Step (Compute Accuracy)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Distance arrays for stats
        pos_dists = []
        neg_dists = []
        
        with torch.no_grad():
            for batch_v, batch_i, batch_lbl in val_loader:
                batch_v, batch_i, batch_lbl = batch_v.to(DEVICE), batch_i.to(DEVICE), batch_lbl.to(DEVICE)
                v_emb, i_emb = model(batch_v, batch_i)
                
                loss = criterion(v_emb, i_emb, batch_lbl)
                val_loss += loss.item()
                
                # Calculate Accuracy
                dists = F.pairwise_distance(v_emb, i_emb)
                
                # Optimal threshold search (simple version: 0.6)
                threshold = 0.6
                predictions = (dists < threshold).float()
                correct += (predictions == batch_lbl).sum().item()
                total += batch_lbl.size(0)
                
                # Collect stats
                for d, l in zip(dists, batch_lbl):
                    if l == 1: pos_dists.append(d.item())
                    else: neg_dists.append(d.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        avg_pos = np.mean(pos_dists) if pos_dists else 0
        avg_neg = np.mean(neg_dists) if neg_dists else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {accuracy:.2f}% | "
              f"Pos Dist: {avg_pos:.3f} | Neg Dist: {avg_neg:.3f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best_sota_model.pth")
            print(f"  --> New Best Model Saved! ({accuracy:.2f}%)")

    print(f"\n🏆 Final Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()