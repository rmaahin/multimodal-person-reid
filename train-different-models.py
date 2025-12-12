import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# --- 1. CONFIGURATION ---
NPZ_FILE = "siamese_A01_augmented.npz"
BATCH_SIZE = 64          # Increased batch size for stable gradients
LEARNING_RATE = 0.0001   # Lower LR for Transformer stability
EPOCHS = 30
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- 2. ADVANCED FEATURE ENGINEERING ---
def compute_derivatives(sequence):
    """
    Computes 1st derivative (Velocity) and 2nd derivative (Acceleration)
    and concatenates them to the original sequence.
    Input: (Seq_Len, Features)
    Output: (Seq_Len, Features * 3)
    """
    # Convert to tensor if numpy
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).float()
    
    # Pad to keep sequence length same after diff
    # 1st Derivative (Velocity)
    velocity = torch.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    
    # 2nd Derivative (Acceleration)
    acceleration = torch.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Concatenate: [Position, Velocity, Acceleration]
    return torch.cat([sequence, velocity, acceleration], dim=1)

# --- 3. DATASET CLASS ---
class VidimuAdvancedDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path)
        
        self.video_pairs = data['video_pairs']
        self.imu_pairs = data['imu_pairs']
        self.labels = torch.from_numpy(data['labels']).float()
        
        print(f"Computing derivatives for {len(self.labels)} pairs...")
        
        # Pre-compute derivatives to save time during training
        self.processed_video = []
        self.processed_imu = []
        
        for i in range(len(self.labels)):
            # Process Video
            vid_seq = self.video_pairs[i]
            vid_aug = compute_derivatives(vid_seq)
            self.processed_video.append(vid_aug)
            
            # Process IMU
            imu_seq = self.imu_pairs[i]
            imu_aug = compute_derivatives(imu_seq)
            self.processed_imu.append(imu_aug)
            
        # Stack into single tensors
        self.processed_video = torch.stack(self.processed_video)
        self.processed_imu = torch.stack(self.processed_imu)
        
        print(f"Feature Engineering Complete.")
        print(f"New Video Shape: {self.processed_video.shape}") # Features * 3
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.processed_video[idx], self.processed_imu[idx], self.labels[idx]

# --- 4. TRANSFORMER ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq, d_model]
        # Add positional encoding to input embeddings
        x = x + self.pe[:x.size(1), :]
        return x

class TransformerBranch(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=3, output_dim=32):
        super(TransformerBranch, self).__init__()
        
        # 1. Input Projection + Batch Norm
        # We use BatchNorm to handle scale differences (Video vs IMU)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(model_dim)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                   dim_feedforward=128, dropout=DROPOUT, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Projection
        self.fc = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Final embedding
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Features)
        
        # Normalize features
        x = x.permute(0, 2, 1) # (Batch, Features, Seq) for BatchNorm
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1) # Back to (Batch, Seq, Features)
        
        # Project to model dimension & Add Position Info
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Run Transformer
        x = self.transformer(x)
        
        # Global Average Pooling (Aggregating over time)
        # We take the mean of all time steps to get one vector
        x = x.mean(dim=1) 
        
        return self.fc(x)

class SiameseTransformer(nn.Module):
    def __init__(self, video_input_dim, imu_input_dim):
        super(SiameseTransformer, self).__init__()
        # Two independent Transformer towers
        self.video_branch = TransformerBranch(video_input_dim)
        self.imu_branch = TransformerBranch(imu_input_dim)

    def forward(self, video_seq, imu_seq):
        video_emb = self.video_branch(video_seq)
        imu_emb = self.imu_branch(imu_seq)
        return video_emb, imu_emb

# --- 5. LOSS FUNCTION ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 6. TRAINING LOOP ---
def train():
    # Load Dataset
    dataset = VidimuAdvancedDataset(NPZ_FILE)
    
    # Split Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get input dimensions (which now include derivatives!)
    sample_video, sample_imu, _ = dataset[0]
    video_dim = sample_video.shape[1] 
    imu_dim = sample_imu.shape[1]
    
    print(f"Enhanced Input Dimensions -> Video: {video_dim}, IMU: {imu_dim}")

    # Initialize Model & Optimizer
    model = SiameseTransformer(video_dim, imu_dim).to(DEVICE)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW helps regularization
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Starting training...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_video, batch_imu, batch_label in train_loader:
            batch_video, batch_imu, batch_label = batch_video.to(DEVICE), batch_imu.to(DEVICE), batch_label.to(DEVICE)
            
            optimizer.zero_grad()
            video_emb, imu_emb = model(batch_video, batch_imu)
            loss = criterion(video_emb, imu_emb, batch_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_video, batch_imu, batch_label in val_loader:
                batch_video, batch_imu, batch_label = batch_video.to(DEVICE), batch_imu.to(DEVICE), batch_label.to(DEVICE)
                v_emb, i_emb = model(batch_video, batch_imu)
                loss = criterion(v_emb, i_emb, batch_label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_siamese_transformer.pth")

    print("Training complete. Best model saved to 'best_siamese_transformer.pth'")

if __name__ == "__main__":
    train()