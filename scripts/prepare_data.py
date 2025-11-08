import os
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def prepare_data(cfg: DictConfig):
    """
    This script simulates the data preparation process.
    For PR 1/N, it primarily ensures the dummy data structure is in place.
    In future iterations, this will handle:
    - Loading raw video data
    - Running pose estimation (e.g., MediaPipe) to extract keypoints
    - Normalizing and augmenting keypoints
    - Generating labels.csv and keypoint .npy files
    - Performing train/val/test splits based on subject_id
    """
    print("Starting data preparation (simulated for PR 1/N)...")
    print(OmegaConf.to_yaml(cfg))

    processed_data_dir = cfg.paths.data_dir + "/processed/squat_form"
    os.makedirs(processed_data_dir, exist_ok=True)

    # Ensure dummy subject directories exist
    subjects = ["subject_001", "subject_002"]
    for subject in subjects:
        os.makedirs(os.path.join(processed_data_dir, subject), exist_ok=True)

    # Ensure dummy labels.csv exists
    labels_path = os.path.join(processed_data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        print(f"Creating dummy labels.csv at {labels_path}")
        dummy_labels = pd.DataFrame({
            'clip_id': ['clip_001_rep_01', 'clip_001_rep_02', 'clip_002_rep_01', 'clip_002_rep_02'],
            'subject_id': ['subject_001', 'subject_001', 'subject_002', 'subject_002'],
            'label': ['good_rep', 'knees_caving', 'not_deep_enough', 'good_rep']
        })
        dummy_labels.to_csv(labels_path, index=False)

    # Ensure dummy keypoint .npy files exist
    dummy_data_specs = {
        'subject_001': {
            'clip_001_rep_01': 80,
            'clip_001_rep_02': 90,
        },
        'subject_002': {
            'clip_002_rep_01': 75,
            'clip_002_rep_02': 85,
        }
    }

    for subject, clips in dummy_data_specs.items():
        for clip_id, num_frames in clips.items():
            keypoint_file_path = os.path.join(processed_data_dir, subject, f"{clip_id}.npy")
            if not os.path.exists(keypoint_file_path):
                print(f"Creating dummy keypoint file: {keypoint_file_path}")
                dummy_keypoints = np.random.rand(num_frames, 33, 3).astype(np.float32)
                np.save(keypoint_file_path, dummy_keypoints)

    print("Data preparation simulation complete.")

if __name__ == "__main__":
    prepare_data()
