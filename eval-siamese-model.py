import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Import classes from your training script
# (Make sure train_siamese_network.py is in the same folder)
from train_siamese_network import SiameseNetwork, VidimuSiameseDataset

# --- CONFIGURATION ---
NPZ_FILE = "/mnt/dataset-augmentation/siamese_A01_augmented.npz"
MODEL_PATH = "siamese_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Loading data from {NPZ_FILE}...")
    dataset = VidimuSiameseDataset(NPZ_FILE)
    
    # Ideally, you should use a separate Test set. 
    # For now, we will evaluate on the full dataset to see how well it learned.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input dimensions from first sample
    sample_video, sample_imu, _ = dataset[0]
    video_dim = sample_video.shape[1]
    imu_dim = sample_imu.shape[1]

    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = SiameseNetwork(video_dim, imu_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running inference...")
    
    distances = []
    labels = []

    with torch.no_grad():
        for batch_video, batch_imu, batch_label in dataloader:
            batch_video = batch_video.to(DEVICE)
            batch_imu = batch_imu.to(DEVICE)
            
            # Get embeddings
            video_emb, imu_emb = model(batch_video, batch_imu)
            
            # Calculate Euclidean distance
            dist = nn.functional.pairwise_distance(video_emb, imu_emb)
            
            distances.extend(dist.cpu().numpy())
            labels.extend(batch_label.cpu().numpy())

    distances = np.array(distances)
    labels = np.array(labels)

    # --- ANALYSIS ---
    
    # Separate distances by label
    pos_distances = distances[labels == 1]
    neg_distances = distances[labels == 0]

    print(f"\nAverage Distance (Positive Pairs): {np.mean(pos_distances):.4f} (Should be low)")
    print(f"Average Distance (Negative Pairs): {np.mean(neg_distances):.4f} (Should be high)")

    # --- PLOT HISTOGRAM ---
    plt.figure(figsize=(10, 6))
    plt.hist(pos_distances, bins=30, alpha=0.7, label='Positive (Same Person)', color='green')
    plt.hist(neg_distances, bins=30, alpha=0.7, label='Negative (Diff Person)', color='red')
    plt.title("Distribution of Distances for Positive vs Negative Pairs")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "evaluation_histogram.png"
    plt.savefig(save_path)
    print(f"Histogram saved to {save_path}")
    plt.show()

    # --- CALCULATE ACCURACY ---
    # We need to find a 'threshold' to decide if it's a match or not.
    # A simple way is to check the ROC curve.
    fpr, tpr, thresholds = roc_curve(labels, -distances) # Negative because lower dist = higher score
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold (closest to top-left corner of ROC)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]
    
    print(f"\nOptimal Distance Threshold: {optimal_threshold:.4f}")
    
    # Compute accuracy at this threshold
    predictions = (distances < optimal_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate()
