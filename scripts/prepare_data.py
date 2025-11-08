import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.data.components import KeypointProcessor

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def prepare_data(cfg: DictConfig) -> None:
    """
    This script prepares the squat form analysis dataset.
    It simulates raw keypoint data, processes it using KeypointProcessor,
    generates labels.csv, and performs subject-stratified train/val/test splits.
    """
    print("Starting data preparation...")
    print(OmegaConf.to_yaml(cfg))

    raw_data_dir = os.path.join(cfg.paths.data_dir, "raw_simulated_keypoints")
    processed_data_dir = os.path.join(cfg.paths.data_dir, "processed", "squat_form")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # --- 1. Simulate Raw Keypoint Data (Placeholder for actual pose estimation output) ---
    # This part simulates the output of a MediaPipe-like pose estimator on raw video.
    # In a real scenario, this would involve reading video files and running a pose model.
    print("Simulating raw keypoint data...")
    num_subjects = 5
    clips_per_subject = 5
    min_frames = 60
    max_frames = 120
    num_keypoints = 33 # MediaPipe BlazePose
    num_coords = 3 # x, y, z

    all_raw_labels = []
    all_subject_metadata = []
    
    skin_tones = ["Monk_1", "Monk_2", "Monk_3", "Monk_4", "Monk_5", "Monk_6"]
    body_types = ["athletic", "average", "heavy"]

    for s_idx in range(num_subjects):
        subject_id = f"subject_{s_idx:03d}"
        os.makedirs(os.path.join(raw_data_dir, subject_id), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, subject_id), exist_ok=True)

        # Generate subject metadata
        all_subject_metadata.append({
            'subject_id': subject_id,
            'skin_tone': np.random.choice(skin_tones),
            'body_type': np.random.choice(body_types),
            'age': np.random.randint(18, 60),
            'gender': np.random.choice(['male', 'female', 'non-binary'])
        })

        for c_idx in range(clips_per_subject):
            clip_id = f"clip_{s_idx:03d}_rep_{c_idx:02d}"
            num_current_frames = np.random.randint(min_frames, max_frames + 1)
            
            # Simulate raw keypoints (e.g., pixel coordinates or camera coordinates)
            raw_keypoints = np.random.rand(num_current_frames, num_keypoints, num_coords).astype(np.float32) * 1000
            raw_keypoint_path = os.path.join(raw_data_dir, subject_id, f"{clip_id}.npy")
            np.save(raw_keypoint_path, raw_keypoints)

            # Assign a random label (matching the label_map in SquatFormDatamodule)
            labels = ["good_rep", "KNEE_VALGUS", "NOT_DEEP_ENOUGH", "BUTT_WINK", "SPINAL_FLEXION"]
            random_label = np.random.choice(labels)
            all_raw_labels.append({'clip_id': clip_id, 'subject_id': subject_id, 'label': random_label})

    raw_labels_df = pd.DataFrame(all_raw_labels)
    raw_labels_path = os.path.join(raw_data_dir, "labels.csv")
    raw_labels_df.to_csv(raw_labels_path, index=False)
    print(f"Simulated {len(raw_labels_df)} raw clips for {num_subjects} subjects.")

    subject_metadata_df = pd.DataFrame(all_subject_metadata)
    subject_metadata_path = os.path.join(processed_data_dir, "subject_metadata.csv")
    subject_metadata_df.to_csv(subject_metadata_path, index=False)
    print(f"Generated subject metadata at {subject_metadata_path}.")


    # --- 2. Copy Raw Keypoints to Processed Directory ---
    # We don't pre-process keypoints here - the dataset will handle normalization and augmentation
    print("Copying raw keypoints to processed directory...")

    all_processed_labels = []
    for _, row in raw_labels_df.iterrows():
        raw_keypoint_path = os.path.join(raw_data_dir, row['subject_id'], f"{row['clip_id']}.npy")
        processed_keypoint_path = os.path.join(processed_data_dir, row['subject_id'], f"{row['clip_id']}.npy")

        raw_kps = np.load(raw_keypoint_path)
        # Save raw keypoints to processed directory - dataset will handle processing
        np.save(processed_keypoint_path, raw_kps)
        all_processed_labels.append(row.to_dict()) # Keep original label and IDs

    processed_labels_df = pd.DataFrame(all_processed_labels)
    processed_labels_path = os.path.join(processed_data_dir, "labels.csv")
    processed_labels_df.to_csv(processed_labels_path, index=False)
    print(f"Processed {len(processed_labels_df)} clips and saved to {processed_data_dir}.")

    # --- 3. Perform Subject-Stratified Train/Val/Test Split ---
    print("Performing subject-stratified split...")
    all_subjects = processed_labels_df['subject_id'].unique().tolist()
    np.random.seed(cfg.seed) # Ensure reproducibility of split
    np.random.shuffle(all_subjects)

    train_ratio = cfg.data.train_split_ratio
    val_ratio = cfg.data.val_split_ratio
    test_ratio = cfg.data.test_split_ratio

    # Adjust ratios to sum to 1 if they don't, and ensure test_ratio is derived
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        print("Warning: Train/Val/Test ratios do not sum to 1. Adjusting test ratio.")
        test_ratio = 1.0 - (train_ratio + val_ratio)
        if test_ratio < 0:
            raise ValueError("Invalid train/val ratios, sum exceeds 1.0.")

    train_subjects, temp_subjects = train_test_split(
        all_subjects, train_size=train_ratio, random_state=cfg.seed
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, train_size=val_ratio / (val_ratio + test_ratio), random_state=cfg.seed
    )

    # Save subject lists to files
    split_dir = os.path.join(cfg.paths.data_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, "train_subjects.txt"), "w") as f:
        for s in train_subjects: f.write(f"{s}\n")
    with open(os.path.join(split_dir, "val_subjects.txt"), "w") as f:
        for s in val_subjects: f.write(f"{s}\n")
    with open(os.path.join(split_dir, "test_subjects.txt"), "w") as f:
        for s in test_subjects: f.write(f"{s}\n")

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Split subject IDs saved to {split_dir}.")

    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()
