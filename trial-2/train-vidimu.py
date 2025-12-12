import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
NPZ_FILE = "/person-reid/multimodal-person-reid/dataset-augmentation/siamese_A01_augmented.npz"

# HYPERPARAMETERS
BATCH_SIZE = 512          
LEARNING_RATE = 1e-4      
EPOCHS = 100              
EMBEDDING_DIM = 128       
DROPOUT = 0.2
MARGIN = 2.0              # INCREASED: Forces harder separation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training SOTA v2 (Shared Projector) on {DEVICE}")

# --- 2. FEATURE ENGINEERING ---
def compute_derivatives(sequence):
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
        return self.processed_video[idx], self.processed_imu[idx], self.labels[idx]

# --- 4. MODEL ARCHITECTURE ---
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
        
        # NOTE: We removed the final projection here.
        # This branch now outputs the raw Transformer features.
        
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
        
        # Define internal model dimension
        transformer_dim = 256
        
        # 1. Independent Branches
        self.video_branch = SOTATransformerBranch(video_dim, model_dim=transformer_dim)
        self.imu_branch = SOTATransformerBranch(imu_dim, model_dim=transformer_dim)
        
        # 2. Shared Projection Head (THE FIX)
        # Both branches feed into this same MLP to force alignment
        self.projector = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, EMBEDDING_DIM) 
        )

    def forward(self, video_seq, imu_seq):
        # Get raw features from branches
        v_feat = self.video_branch(video_seq)
        i_feat = self.imu_branch(imu_seq)
        
        # Pass both through the SHARED projector
        v_emb = self.projector(v_feat)
        i_emb = self.projector(i_feat)
        
        # Normalize at the very end
        return F.normalize(v_emb, p=2, dim=1), F.normalize(i_emb, p=2, dim=1)

# --- 5. LOSS FUNCTION ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 6. TRAINING ENGINE ---
def train():
    full_dataset = VidimuSOTADataset(NPZ_FILE)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    sample_v, sample_i, _ = full_dataset[0]
    model = CrossModalNetwork(sample_v.shape[1], sample_i.shape[1]).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=MARGIN) # Uses the increased margin
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.1
    )

    print("\n🔥 Starting Training (v2 - Shared Projector)...")
    
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

        model.eval()
        correct = 0
        total = 0
        pos_dists = []
        neg_dists = []
        
        with torch.no_grad():
            for batch_v, batch_i, batch_lbl in val_loader:
                batch_v, batch_i, batch_lbl = batch_v.to(DEVICE), batch_i.to(DEVICE), batch_lbl.to(DEVICE)
                v_emb, i_emb = model(batch_v, batch_i)
                
                dists = F.pairwise_distance(v_emb, i_emb)
                
                # Dynamic Threshold Calculation for Accuracy
                # We assume Positive < 1.0 (since margin is 2.0)
                threshold = 1.0 
                predictions = (dists < threshold).float()
                correct += (predictions == batch_lbl).sum().item()
                total += batch_lbl.size(0)
                
                for d, l in zip(dists, batch_lbl):
                    if l == 1: pos_dists.append(d.item())
                    else: neg_dists.append(d.item())

        avg_train_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        avg_pos = np.mean(pos_dists) if pos_dists else 0
        avg_neg = np.mean(neg_dists) if neg_dists else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {accuracy:.2f}% | "
              f"Pos: {avg_pos:.3f} | Neg: {avg_neg:.3f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best_sota_model_v2.pth")
            print(f"  --> Saved v2 Model ({accuracy:.2f}%)")

    print(f"\n🏆 Final Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()