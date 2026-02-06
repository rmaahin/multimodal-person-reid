import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import gc

# --- 1. CONFIGURATION ---
NPZ_FILE = "/mnt/multimodal-person-reid/dataset-augmentation/siamese_A01_augmented.npz"

# HYPERPARAMETERS - MEMORY OPTIMIZED
BATCH_SIZE = 256          # REDUCED from 512 to save memory
LEARNING_RATE = 1e-4      
EPOCHS = 100              
EMBEDDING_DIM = 128       
DROPOUT = 0.2
MARGIN = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Corrected SOTA v2 (Memory Optimized) on {DEVICE}")

# --- 2. FEATURE ENGINEERING ---
def compute_derivatives(sequence):
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).float()
    
    velocity = torch.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    
    acceleration = torch.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    return torch.cat([sequence, velocity, acceleration], dim=1)

# --- 3. LAZY LOADING DATASET (MEMORY EFFICIENT) ---
class VidimuSOTADatasetLazy(Dataset):
    """
    Lazy loading dataset - computes derivatives on-the-fly instead of pre-loading.
    Trades CPU for RAM - slower but much more memory efficient.
    """
    def __init__(self, npz_path, indices=None):
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        # Store raw data
        if indices is not None:
            self.video_pairs = data['video_pairs'][indices]
            self.imu_pairs = data['imu_pairs'][indices]
            self.labels = data['labels'][indices]
        else:
            self.video_pairs = data['video_pairs']
            self.imu_pairs = data['imu_pairs']
            self.labels = data['labels']
        
        print(f"Dataset Ready. Size: {len(self.labels)} pairs")
        print(f"  Positive pairs: {np.sum(self.labels == 1)}")
        print(f"  Negative pairs: {np.sum(self.labels == 0)}")
        print(f"  Memory mode: Lazy (compute derivatives on-the-fly)")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Compute derivatives on-the-fly
        video = compute_derivatives(self.video_pairs[idx])
        imu = compute_derivatives(self.imu_pairs[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return video, imu, label

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
        
        self.video_branch = SOTATransformerBranch(video_dim, model_dim=transformer_dim)
        self.imu_branch = SOTATransformerBranch(imu_dim, model_dim=transformer_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, EMBEDDING_DIM) 
        )

    def forward(self, video_seq, imu_seq):
        v_feat = self.video_branch(video_seq)
        i_feat = self.imu_branch(imu_seq)
        
        v_emb = self.projector(v_feat)
        i_emb = self.projector(i_feat)
        
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

# --- 6. OPTIMAL THRESHOLD FINDER ---
def find_optimal_threshold(distances, labels):
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]
    return optimal_threshold, auc(fpr, tpr)

# --- 7. TRAINING ENGINE ---
def train():
    print("Analyzing dataset structure...")
    data = np.load(NPZ_FILE)
    total_pairs = len(data['labels'])
    
    # Create train/val split
    all_indices = np.arange(total_pairs)
    train_indices, val_indices = train_test_split(
        all_indices, 
        test_size=0.1, 
        random_state=42,
        stratify=data['labels']
    )
    
    print(f"Train set: {len(train_indices)} pairs")
    print(f"Val set: {len(val_indices)} pairs")
    
    # MEMORY OPTIMIZATION: Use lazy loading
    train_dataset = VidimuSOTADatasetLazy(NPZ_FILE, indices=train_indices)
    val_dataset = VidimuSOTADatasetLazy(NPZ_FILE, indices=val_indices)
    
    # MEMORY OPTIMIZATION: num_workers=0 to avoid shared memory issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid shared memory errors
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    sample_v, sample_i, _ = train_dataset[0]
    model = CrossModalNetwork(sample_v.shape[1], sample_i.shape[1]).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=MARGIN)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.1
    )

    print(f"\n🔥 Starting Training (Memory Optimized)...")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Margin: {MARGIN}")
    print(f"   Workers: 0 (to avoid shared memory issues)")
    
    best_auc = 0.0
    best_threshold = 0.5
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (batch_v, batch_i, batch_lbl) in enumerate(train_loader):
            batch_v = batch_v.to(DEVICE)
            batch_i = batch_i.to(DEVICE)
            batch_lbl = batch_lbl.to(DEVICE)
            
            optimizer.zero_grad()
            v_emb, i_emb = model(batch_v, batch_i)
            
            loss = criterion(v_emb, i_emb, batch_lbl)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # MEMORY OPTIMIZATION: Clear cache periodically
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_dists = []
        all_labels = []
        pos_dists = []
        neg_dists = []
        
        with torch.no_grad():
            for batch_v, batch_i, batch_lbl in val_loader:
                batch_v = batch_v.to(DEVICE)
                batch_i = batch_i.to(DEVICE)
                batch_lbl = batch_lbl.to(DEVICE)
                
                v_emb, i_emb = model(batch_v, batch_i)
                
                loss = criterion(v_emb, i_emb, batch_lbl)
                val_loss += loss.item()
                
                dists = F.pairwise_distance(v_emb, i_emb).cpu().numpy()
                labels_np = batch_lbl.cpu().numpy()
                
                all_dists.extend(dists)
                all_labels.extend(labels_np)
                
                for d, l in zip(dists, labels_np):
                    if l == 1: 
                        pos_dists.append(d)
                    else: 
                        neg_dists.append(d)

        # Convert to numpy
        all_dists = np.array(all_dists)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold and calculate AUC
        optimal_threshold, roc_auc = find_optimal_threshold(all_dists, all_labels)
        
        # Calculate accuracy
        predictions = (all_dists < optimal_threshold).astype(float)
        accuracy = 100 * np.mean(predictions == all_labels)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_pos = np.mean(pos_dists) if pos_dists else 0
        avg_neg = np.mean(neg_dists) if neg_dists else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"TrLoss: {avg_train_loss:.4f} | "
              f"VLoss: {avg_val_loss:.4f} | "
              f"Acc: {accuracy:.2f}% | "
              f"AUC: {roc_auc:.4f} | "
              f"Thr: {optimal_threshold:.3f} | "
              f"Pos: {avg_pos:.3f} | Neg: {avg_neg:.3f}")
        
        # Save best model
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_threshold = optimal_threshold
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': roc_auc,
                'threshold': optimal_threshold,
                'margin': MARGIN
            }, "/mnt/multimodal-person-reid/trial-3/best_model_v3.pth")
            print(f"  ✅ New Best Model Saved! (AUC: {roc_auc:.4f})")
        
        # MEMORY OPTIMIZATION: Periodic cleanup
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n🏆 Training Complete!")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Best Threshold: {best_threshold:.3f}")
    print(f"   Model saved to: best_model_v3.pth")

if __name__ == "__main__":
    train()