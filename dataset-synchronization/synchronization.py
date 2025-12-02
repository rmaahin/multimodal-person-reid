import os
import pandas as pd
import utils.fileProcessing as fileutil
import utils.syncUtilities as syncutil
import shutil

def run_synchronization():
    """
    1. Calculates optimal shifts and saves them to 'infoToSync.csv'.
    2. Creates a new synchronized dataset based on those shifts.
    """
    
    FULDATASETPATH = '/mnt'
    SYNCHRONIZATION_PATH = '/mnt/dataset-synchronization'

    print(f"Using dataset path: {FULDATASETPATH}\n")

    print("--- Estimating Synchronization ---")
    inpath_estimate = os.path.join(FULDATASETPATH, 'dataset', 'videoandimus')
    outpath_estimate = os.path.join(SYNCHRONIZATION_PATH, 'synchronization_trials')
    if not os.path.exists(outpath_estimate):
        os.makedirs(outpath_estimate)

    subjects = ["S40", "S41", "S42", "S44", "S46", "S47", "S48", "S49",
                "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57"]
    
    dataset_activities = ["A01"]
    activities_legend = ["walk_forward"]

    RMSE_SAMPLES = 180
    FINAL_LENGTH = 180
    MAX_SYNC_OVERLAP = 15

    csvlog = 'infoToSync.csv'
    csvlogfile = os.path.join(outpath_estimate, csvlog)
    if os.path.exists(csvlogfile):
        print(f"Removing old log file: {csvlogfile}")
        os.remove(csvlogfile)

    for activity, legend in zip(dataset_activities, activities_legend):
        print(f"\nProcessing Activity: {activity} ({legend})")
        
        # This one function does all the work:
        # 1. Loads video and IMU
        # 2. Processes signals (resample, filter, mean-center)
        # 3. Finds the best shift (by minimizing RMSE)
        # 4. Saves the results to the 'infoToSync.csv'
        # 5. Saves a verification plot
        syncutil.plotFramesShiftToSyncrhonizeAllSubjectsOneActivity(
            csvlog=csvlogfile,
            inpath=inpath_estimate,
            outpath=outpath_estimate,
            subjects=subjects,
            activity=activity,
            activity_legend=legend,
            outputfilename=f"{activity}_({legend})_synchronize_plot.jpg", 
            RMSE_SAMPLES=RMSE_SAMPLES,
            MAX_SYNC_OVERLAP=MAX_SYNC_OVERLAP,
            FINAL_LENGTH=FINAL_LENGTH
        )
    
    print("\n--- Estimation complete. ---")
    print(f"Log file created at: {csvlogfile}")


    print("\n--- Building Complete Synchronized Dataset ---")

    inpath_original_data = os.path.join(FULDATASETPATH, 'dataset', 'videoandimus')
    inpath_csv_log_dir = os.path.join(SYNCHRONIZATION_PATH, 'synchronization_trials')
    
    outpath_synced_dataset = os.path.join(SYNCHRONIZATION_PATH, 'A01-synced-dataset')
    
    if not os.path.exists(outpath_synced_dataset):
        os.makedirs(outpath_synced_dataset)

    csvlogfile_modify = os.path.join(inpath_csv_log_dir, csvlog)
    try:
        dfsync = pd.read_csv(csvlogfile_modify)
        print(f"Loaded {len(dfsync)} file modifications from {csvlogfile_modify}")
    except FileNotFoundError:
        print(f"[ERROR] Can't find log file: {csvlogfile_modify}")
        print("Please run Part 1 first.")
        return

    # --- LOOP THROUGH ALL SUBJECTS AND ACTIVITIES ---
    
    print(f"Building complete dataset in: {outpath_synced_dataset}")
    
    for subject in subjects:
        outpath_subject_dir = os.path.join(outpath_synced_dataset, subject)
        if not os.path.exists(outpath_subject_dir):
            os.makedirs(outpath_subject_dir)
            
        for activity in dataset_activities:
            found_trial = None
            for trial_num in ["T01", "T02", "T03", "T04", "T05"]:
                
                mot_filename = f'ik_{subject}_{activity}_{trial_num}.mot'
                inpath_mot_full = os.path.join(inpath_original_data, subject, mot_filename)
                if os.path.exists(inpath_mot_full):
                    found_trial = trial_num
                    break
            
            if not found_trial:
                print(f"Skipping {subject}/{activity}: No valid trial found.")
                continue

            print(f"Processing {subject}/{activity}/{found_trial}...")
            
            file_types_to_process = {
                'raw': {'name_template': f'{subject}_{activity}_{found_trial}.raw', 'header_rows': 2, 'lines_per_frame': 5 * (50/30)},
                'mot': {'name_template': f'ik_{subject}_{activity}_{found_trial}.mot', 'header_rows': 9, 'lines_per_frame': (50/30)},
                'csv': {'name_template': f'{subject}_{activity}_{found_trial}.csv', 'header_rows': 2, 'lines_per_frame': 1.0}
            }

            for ftype, config in file_types_to_process.items():
                
                original_file_name = config['name_template']
                original_file_path = os.path.join(inpath_original_data, subject, original_file_name)
                new_file_path = os.path.join(outpath_subject_dir, original_file_name)

                if not os.path.exists(original_file_path):
                    continue

                task = dfsync[
                    (dfsync['File'] == original_file_path) & 
                    (dfsync['Type'] == ftype)
                ]

                if task.empty:
                    shutil.copy(original_file_path, new_file_path)
                
                else:
                    frames_to_cut = int(task.iloc[0]['CutFrames'])
                    lines_to_cut = int(frames_to_cut * config['lines_per_frame'])
                    
                    fileutil.remove_insidelines_file(
                        original_file_path, 
                        config['header_rows'], 
                        lines_to_cut, 
                        new_file_path
                    )

    print(f"\n--- Complete synchronized dataset is ready at: {outpath_synced_dataset} ---")
    print("Process complete.")

if __name__ == "__main__":
    run_synchronization()