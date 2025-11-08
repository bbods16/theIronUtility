import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict

from src.data.components import KeypointProcessor

class SquatFormDataset(Dataset):
    """
    Dataset for squat form classification using keypoint sequences.
    Loads pre-processed keypoint data and applies normalization/augmentation.
    """
    def __init__(self,
                 data_dir: str,
                 label_map: Dict[str, int],
                 keypoint_processor_config: Dict,
                 is_train: bool = True,
                 sequence_length: int = 100, # Fixed sequence length for now, will pad/truncate
                 split_subjects: List[str] = None):
        """
        Args:
            data_dir (str): Path to the directory containing processed keypoint data.
                            Expected structure: data_dir/subject_id/clip_id.npy (keypoints)
                                                data_dir/labels.csv (clip_id, subject_id, label)
            label_map (Dict[str, int]): Mapping from string labels to integer IDs.
            keypoint_processor_config (Dict): Configuration for KeypointProcessor.
            is_train (bool): Whether this is a training dataset (enables augmentation).
            sequence_length (int): Desired fixed length for all sequences.
            split_subjects (List[str]): List of subject IDs to include in this dataset split.
        """
        self.data_dir = data_dir
        self.label_map = label_map
        self.is_train = is_train
        self.sequence_length = sequence_length
        self.keypoint_processor = KeypointProcessor(keypoint_processor_config)

        self.metadata = self._load_metadata(split_subjects)
        self.class_weights = self._calculate_class_weights()

    def _load_metadata(self, split_subjects: List[str]) -> pd.DataFrame:
        """Loads clip metadata and filters by subject IDs for the current split."""
        labels_path = os.path.join(self.data_dir, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.csv not found at {labels_path}")

        all_labels = pd.read_csv(labels_path)

        if split_subjects:
            metadata = all_labels[all_labels['subject_id'].isin(split_subjects)].copy()
        else:
            metadata = all_labels.copy()

        if metadata.empty:
            raise ValueError(f"No data found for subjects: {split_subjects} in {self.data_dir}")

        # Add full path to keypoint files
        metadata['keypoint_path'] = metadata.apply(
            lambda row: os.path.join(self.data_dir, row['subject_id'], f"{row['clip_id']}.npy"),
            axis=1
        )
        return metadata

    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculates inverse frequency class weights for handling imbalance."""
        if not self.keypoint_processor.config.class_weights.enabled:
            return None

        class_counts = self.metadata['label'].value_counts()
        total_samples = class_counts.sum()
        num_classes = len(self.label_map)

        # Ensure all classes are present, even if count is 0
        for label_str in self.label_map.keys():
            if label_str not in class_counts:
                class_counts[label_str] = 0

        # Sort by label_map order to match CrossEntropyLoss expectation
        sorted_class_counts = [class_counts[label_str] for label_str in sorted(self.label_map, key=self.label_map.get)]

        # Calculate inverse frequency weights
        weights = [total_samples / (count + 1e-5) for count in sorted_class_counts] # Add epsilon for stability
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return weights_tensor / weights_tensor.sum() * num_classes # Normalize to keep magnitude similar

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        keypoint_path = row['keypoint_path']
        label_str = row['label']

        # Load keypoints (num_frames, num_keypoints, 3)
        keypoints = np.load(keypoint_path)

        # Process keypoints (normalize, augment)
        processed_keypoints = self.keypoint_processor.process(keypoints, self.is_train) # (num_frames, num_keypoints * 3)

        # Pad/truncate sequence to fixed length
        if processed_keypoints.shape[0] > self.sequence_length:
            processed_keypoints = processed_keypoints[:self.sequence_length, :]
        elif processed_keypoints.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - processed_keypoints.shape[0], processed_keypoints.shape[1]))
            processed_keypoints = np.vstack((processed_keypoints, padding))

        # Convert to PyTorch tensors
        keypoints_tensor = torch.tensor(processed_keypoints, dtype=torch.float32)
        label_tensor = torch.tensor(self.label_map[label_str], dtype=torch.long)

        return keypoints_tensor, label_tensor

class SquatFormDatamodule:
    """
    PyTorch Lightning DataModule for Squat Form Classification.
    Handles data loading, splitting, and DataLoader creation.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config.data.dataset_dir
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.pin_memory = config.data.pin_memory

        # Define label map based on problem spec
        self.label_map = {
            "good_rep": 0,
            "knees_caving": 1,
            "not_deep_enough": 2,
            "butt_wink": 3,
            "spinal_flexion": 4,
        }
        self.num_classes = len(self.label_map)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def setup(self, stage: str = None):
        """
        Loads and splits the dataset.
        """
        # Load all metadata to get subject IDs for splitting
        all_labels_path = os.path.join(self.data_dir, "labels.csv")
        if not os.path.exists(all_labels_path):
            raise FileNotFoundError(f"labels.csv not found at {all_labels_path}")
        all_labels = pd.read_csv(all_labels_path)
        all_subjects = all_labels['subject_id'].unique().tolist()

        # Perform subject-stratified split
        np.random.seed(self.config.seed)
        np.random.shuffle(all_subjects)

        train_split_idx = int(len(all_subjects) * self.config.data.train_split_ratio)
        val_split_idx = int(len(all_subjects) * (self.config.data.train_split_ratio + self.config.data.val_split_ratio))

        train_subjects = all_subjects[:train_split_idx]
        val_subjects = all_subjects[train_split_idx:val_split_idx]
        test_subjects = all_subjects[val_split_idx:]

        # Create datasets
        self.train_dataset = SquatFormDataset(
            data_dir=self.data_dir,
            label_map=self.label_map,
            keypoint_processor_config=self.config.data.augmentations,
            is_train=True,
            split_subjects=train_subjects
        )
        self.val_dataset = SquatFormDataset(
            data_dir=self.data_dir,
            label_map=self.label_map,
            keypoint_processor_config=self.config.data.augmentations,
            is_train=False,
            split_subjects=val_subjects
        )
        self.test_dataset = SquatFormDataset(
            data_dir=self.data_dir,
            label_map=self.label_map,
            keypoint_processor_config=self.config.data.augmentations,
            is_train=False,
            split_subjects=test_subjects
        )

        # Get class weights from the training dataset
        self.class_weights = self.train_dataset.class_weights

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
