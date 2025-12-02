import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.signal import resample as scipy_resample

def calculate_3d_angle(a, b, c):
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0:
        return np.nan
    cos_theta = np.clip(dot_product / (mag_ba * mag_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def load_video_angle_signal(video_csv_path, joint_names):
    df = pd.read_csv(video_csv_path)
    j1_name, j2_name, j3_name = joint_names
    p_a = df[[f'{j1_name}_x', f'{j1_name}_y', f'{j1_name}_z']].values
    p_b = df[[f'{j2_name}_x', f'{j2_name}_y', f'{j2_name}_z']].values
    p_c = df[[f'{j3_name}_x', f'{j3_name}_y', f'{j3_name}_z']].values
    angles = [calculate_3d_angle(p_a[i], p_b[i], p_c[i]) for i in range(len(p_a))]
    return np.array(angles)

def load_imu_angle_signal(imu_mot_path, angle_column_name):
    with open(imu_mot_path, 'r') as f:
        lines = f.readlines()
    header_line_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('time'):
            header_line_index = i
            break
    if header_line_index == -1:
        raise ValueError("Could not find header row in .mot file.")
    df = pd.read_csv(imu_mot_path, sep=r'\s+', skiprows=header_line_index + 1, header=None, engine='python')
    col_names = re.split(r'\s+', lines[header_line_index].strip())
    df.columns = col_names
    return df[angle_column_name].values

def smooth_signal(signal, window_size=5):
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, window, mode='valid')
    pad_front = (window_size - 1) // 2
    pad_end = window_size - 1 - pad_front
    return np.pad(smoothed, (pad_front, pad_end), mode='edge')

def resample_signal(signal, in_hz=50, out_hz=30):
    new_num_samples = int(len(signal) * (out_hz / in_hz))
    return scipy_resample(signal, new_num_samples)

if __name__ == "__main__":
    
    VIDEO_CSV_PATH = "/mnt/dataset-synchronization/synchronization_trials/S40_A01_T01_synced.csv"
    IMU_MOT_PATH = "/mnt/dataset-synchronization/synchronization_trials/S40_A01_T01_synced.mot"
    VIDEO_JOINT_NAMES = ('left_hip', 'left_knee', 'left_ankle')
    IMU_ANGLE_COLUMN = 'knee_angle_l'

    CUT_VIDEO_SAMPLES = 0
    CUT_IMU_SAMPLES = 3

    video_signal = smooth_signal(load_video_angle_signal(VIDEO_CSV_PATH, VIDEO_JOINT_NAMES))
    imu_signal = resample_signal(load_imu_angle_signal(IMU_MOT_PATH, IMU_ANGLE_COLUMN))

    video_orig = video_signal[:180]
    imu_orig = imu_signal[:180]

    video_synced = video_signal[CUT_VIDEO_SAMPLES:]
    imu_synced = imu_signal[CUT_IMU_SAMPLES:]
    
    min_len = min(len(video_synced), len(imu_synced))
    video_synced = video_synced[:min_len]
    imu_synced = imu_synced[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.set_title("BEFORE Synchronization (First 6 Seconds)", fontsize=16)
    ax1.plot(video_orig, label="Video Angle (Smoothed)", color="blue", alpha=0.7)
    ax1.plot(imu_orig, label="IMU Angle (Resampled)", color="red", linestyle="--")
    ax1.set_xlabel("Samples (at 30Hz)")
    ax1.set_ylabel("Angle (Degrees)")
    ax1.legend()
    
    ax2.set_title("AFTER Synchronization (Full Signals, Aligned)", fontsize=16)
    ax2.plot(video_synced, label="Video Angle (Synced)", color="blue", alpha=0.7)
    ax2.plot(imu_synced, label="IMU Angle (Synced)", color="red", linestyle="--")
    ax2.set_xlabel("Samples (at 30Hz)")
    ax2.set_ylabel("Angle (Degrees)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("/mnt/dataset-synchronization/synchronization_trials/synchronization_verification.png")
    print("Saved verification plot to 'synchronization_verification.png'")