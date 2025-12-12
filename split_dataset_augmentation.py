import os
import pandas as pd
import numpy as np
import random

# --- --- --- --- --- --- --- --- --- ---
# --- 1. CONFIGURATION ---
# --- --- --- --- --- --- --- --- --- ---

# Input Path (Synced Dataset)
DATA_DIRECTORY = "dataset-synchronization/A01-synced-dataset"

# Output Path (For the individual window files)
AUGMENTED_OUTPUT_DIR = "dataset-augmentation/A01-augmented-dataset"

# Output Path (For the compiled NPZ file)
OUTPUT_NPZ_FILE = "dataset-augmentation/siamese_A01_augmented.npz"

# Window Size: 90 frames = 3 seconds (at 30fps)
WINDOW_SIZE = 90 

# Stride: 10 frames
STRIDE = 10 

# --- --- --- --- --- --- --- --- --- ---
# --- 2. DATA LOADING FUNCTIONS ---
# --- --- --- --- --- --- --- --- --- ---

def load_video_data(file_path):
    """Loads video keypoints from CSV."""
    try:
        df = pd.read_csv(file_path)
        # Handle potential missing headers from previous sync steps
        if 'pelvis_x' not in df.columns[0] and not isinstance(df.columns[0], str):
             df = pd.read_csv(file_path, header=None)
        return df.values
    except Exception as e:
        print(f"  [Error] Loading video {os.path.basename(file_path)}: {e}")
        return None

def load_imu_data(file_path):
    """Loads IMU quaternion data and reshapes it."""
    try:
        df = pd.read_csv(file_path, sep=',') 
        if 'QUAT' not in df.columns:
            df.columns = ['QUAT', 'w', 'x', 'y', 'z']
            
        sensor_names = df['QUAT'].drop_duplicates().tolist()
        num_sensors = len(sensor_names)
        if num_sensors == 0: return None

        quat_data = df[['w', 'x', 'y', 'z']].values
        num_frames = len(quat_data) // num_sensors
        
        # Truncate and reshape
        quat_data = quat_data[:num_frames * num_sensors]
        reshaped_data = quat_data.reshape(num_frames, num_sensors * 4)
        
        return reshaped_data
    except Exception as e:
        print(f"  [Error] Loading IMU {os.path.basename(file_path)}: {e}")
        return None

# --- --- --- --- --- --- --- --- --- ---
# --- 3. AUGMENTATION & SAVING ---
# --- --- --- --- --- --- --- --- --- ---

def create_augmented_dataset():
    if not os.path.exists(DATA_DIRECTORY):
        print(f"[ERROR] Directory not found: {DATA_DIRECTORY}")
        return

    print(f"Starting augmentation from: {DATA_DIRECTORY}")
    print(f"Saving individual files to: {AUGMENTED_OUTPUT_DIR}")
    
    # Create the root augmented directory
    if not os.path.exists(AUGMENTED_OUTPUT_DIR):
        os.makedirs(AUGMENTED_OUTPUT_DIR)

    all_segments = []
    
    # Get all subject folders
    subject_folders = [f for f in os.listdir(DATA_DIRECTORY) if f.startswith('S') and os.path.isdir(os.path.join(DATA_DIRECTORY, f))]
    subject_folders.sort()
    
    if not subject_folders:
        print("[ERROR] No subject folders found.")
        return

    # --- Step 1: Slice, Save Files, and Store in Memory ---
    for subject in subject_folders:
        subject_path = os.path.join(DATA_DIRECTORY, subject)
        
        # Robust file finding
        files = os.listdir(subject_path)
        video_filename = next((f for f in files if f.endswith('.csv') and 'A01' in f), None)
        imu_filename = next((f for f in files if f.endswith('.raw') and 'A01' in f), None)

        if not video_filename or not imu_filename:
            continue

        # Extract Trial ID (e.g., "T01" from S40_A01_T01.csv) for naming
        trial_id = video_filename.split('_')[2].split('.')[0]

        # Prepare Subject Output Directories
        subj_out_dir_video = os.path.join(AUGMENTED_OUTPUT_DIR, subject, "video")
        subj_out_dir_imu = os.path.join(AUGMENTED_OUTPUT_DIR, subject, "imu")
        
        if not os.path.exists(subj_out_dir_video): os.makedirs(subj_out_dir_video)
        if not os.path.exists(subj_out_dir_imu): os.makedirs(subj_out_dir_imu)

        # Load Data
        video_full = load_video_data(os.path.join(subject_path, video_filename))
        imu_full = load_imu_data(os.path.join(subject_path, imu_filename))
        
        if video_full is None or imu_full is None:
            continue

        min_len = min(len(video_full), len(imu_full))
        video_full = video_full[:min_len]
        imu_full = imu_full[:min_len]
        
        if min_len < WINDOW_SIZE:
            continue
        
        # --- SLIDING WINDOW LOOP ---
        count = 0
        for i in range(0, min_len - WINDOW_SIZE + 1, STRIDE):
            # Extract window
            vid_window = video_full[i : i + WINDOW_SIZE]
            imu_window = imu_full[i : i + WINDOW_SIZE]
            
            # --- SAVE INDIVIDUAL FILES ---
            window_id = f"{subject}_A01_{trial_id}_win{count:03d}"
            
            # Save Video Window
            vid_out_path = os.path.join(subj_out_dir_video, f"{window_id}.csv")
            pd.DataFrame(vid_window).to_csv(vid_out_path, index=False, header=False)
            
            # Save IMU Window
            imu_out_path = os.path.join(subj_out_dir_imu, f"{window_id}.csv")
            pd.DataFrame(imu_window).to_csv(imu_out_path, index=False, header=False)

            # --- STORE FOR NPZ ---
            all_segments.append({
                'video': vid_window,
                'imu': imu_window,
                'subject': subject
            })
            count += 1
            
        print(f"  {subject}: Processed {count} windows. Saved to disk.")

    if not all_segments:
        print("[ERROR] No segments generated.")
        return

    print(f"\nTotal Segments Generated: {len(all_segments)}")
    
    # --- Step 2: Generate Pairs for NPZ ---
    print("Generating pairs for .npz file...")
    
    pairs_video = []
    pairs_imu = []
    labels = []
    
    segments_by_subject = {}
    for seg in all_segments:
        subj = seg['subject']
        if subj not in segments_by_subject:
            segments_by_subject[subj] = []
        segments_by_subject[subj].append(seg)
        
    subject_list = list(segments_by_subject.keys())

    for anchor_seg in all_segments:
        anchor_subj = anchor_seg['subject']
        
        # Positive Pair
        pairs_video.append(anchor_seg['video'])
        pairs_imu.append(anchor_seg['imu'])
        labels.append(1)
        
        # Negative Pair
        neg_subj = anchor_subj
        if len(subject_list) > 1:
            while neg_subj == anchor_subj:
                neg_subj = random.choice(subject_list)
        else:
            neg_subj = anchor_subj 

        neg_seg = random.choice(segments_by_subject[neg_subj])
        
        pairs_video.append(anchor_seg['video'])
        pairs_imu.append(neg_seg['imu'])
        labels.append(0)

    # --- Step 3: Save NPZ ---
    pairs_video_np = np.array(pairs_video)
    pairs_imu_np = np.array(pairs_imu)
    labels_np = np.array(labels)

    print(f"\nFinal NPZ Shapes:")
    print(f"  Video: {pairs_video_np.shape}")
    print(f"  IMU:   {pairs_imu_np.shape}")
    print(f"  Labels: {labels_np.shape}")

    np.savez_compressed(
        OUTPUT_NPZ_FILE,
        video_pairs=pairs_video_np,
        imu_pairs=pairs_imu_np,
        labels=labels_np
    )
    print(f"NPZ Saved to {OUTPUT_NPZ_FILE}")
    print(f"Individual files saved to {AUGMENTED_OUTPUT_DIR}")

if __name__ == "__main__":
    create_augmented_dataset()