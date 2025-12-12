import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# --- 1. CONFIGURATION ---
NPZ_FILE = "/mnt/dataset-augmentation/siamese_A01_augmented.npz"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- 2. CUSTOM DATASET CLASS ---
class VidimuSiameseDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path)
        
        # Load the arrays
        # Shape: (NumPairs, SequenceLength, Features)
        self.video_pairs = torch.from_numpy(data['video_pairs']).float()
        self.imu_pairs = torch.from_numpy(data['imu_pairs']).float()
        self.labels = torch.from_numpy(data['labels']).float()
        
        print(f"Data loaded. {len(self.labels)} pairs.")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Returns: (video_sequence, imu_sequence, label)
        return self.video_pairs[idx], self.imu_pairs[idx], self.labels[idx]

# --- 3. MODEL DEFINITION ---
class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size=64, embedding_size=32):
        super(LSTMBranch, self).__init__()
        
        # --- FIX 1: Batch Normalization ---
        # This forces the input features (degrees or quaternions) 
        # to have mean=0 and variance=1.
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # --- FIX 2: Bidirectional LSTM ---
        # Allows the network to see the future and past context.
        # Note: We assume batch_first=True.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Since it's bidirectional, the hidden output is doubled (hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Add dropout to prevent overfitting
            nn.Linear(64, embedding_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # BatchNorm expects (batch, features, seq_len) for 1D
        # So we swap dimensions, normalize, then swap back.
        x = x.permute(0, 2, 1) 
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        
        # Run LSTM
        # output shape: (batch, seq_len, hidden_size * 2)
        output, (hidden, cell) = self.lstm(x)
        
        # Take the embedding of the FINAL time step
        final_encoding = output[:, -1, :]
        
        # Pass through fully connected layers
        embedding = self.fc(final_encoding)
        return embedding

class SiameseNetwork(nn.Module):
    def __init__(self, video_input_dim, imu_input_dim, hidden_size=64):
        super(SiameseNetwork, self).__init__()
        
        # Two separate "towers" because the data types (video vs IMU) are different
        # but we map them to the SAME size embedding space (32)
        self.video_branch = LSTMBranch(video_input_dim, hidden_size)
        self.imu_branch = LSTMBranch(imu_input_dim, hidden_size)

    def forward(self, video_seq, imu_seq):
        # Get embeddings for both
        video_emb = self.video_branch(video_seq)
        imu_emb = self.imu_branch(imu_seq)
        return video_emb, imu_emb

# --- 4. LOSS FUNCTION (CONTRASTIVE LOSS) ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between the two embeddings
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        # Contrastive Loss Formula:
        # If Label=1 (Positive): Loss = distance^2
        # If Label=0 (Negative): Loss = max(0, margin - distance)^2
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

# --- 5. TRAINING LOOP ---
def train():
    # Load Dataset
    dataset = VidimuSiameseDataset(NPZ_FILE)
    
    # Split into Train (80%) and Validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input dimensions dynamically from the first batch
    sample_video, sample_imu, _ = dataset[0]
    video_dim = sample_video.shape[1] # Features dimension
    imu_dim = sample_imu.shape[1]     # Features dimension
    
    print(f"Input Dimensions -> Video: {video_dim}, IMU: {imu_dim}")

    # Initialize Model
    model = SiameseNetwork(video_dim, imu_dim).to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_video, batch_imu, batch_label in train_loader:
            batch_video, batch_imu, batch_label = batch_video.to(DEVICE), batch_imu.to(DEVICE), batch_label.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            video_emb, imu_emb = model(batch_video, batch_imu)
            
            # Compute loss
            loss = criterion(video_emb, imu_emb, batch_label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # --- SAVE THE MODEL ---
    torch.save(model.state_dict(), "siamese_model.pth")
    print("Training complete. Model saved to 'siamese_model.pth'")

if __name__ == "__main__":
    train()