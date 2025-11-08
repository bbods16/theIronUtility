import torch
import numpy as np

class KeypointProcessor:
    """
    Handles normalization and augmentation of keypoint sequences.
    """
    def __init__(self, config):
        self.config = config

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalizes keypoint coordinates for scale and translation invariance.
        Assumes keypoints are (num_frames, num_keypoints, 3) for (x, y, z).
        Normalization by hip-to-shoulder distance.
        """
        if not self.config.normalization.enabled:
            return keypoints

        normalized_keypoints = keypoints.copy()

        # Assuming MediaPipe keypoint indices:
        # 11: left_hip, 12: right_hip
        # 23: left_shoulder, 24: right_shoulder

        # Calculate hip center for translation
        left_hip = keypoints[:, 11, :3] # (num_frames, 3)
        right_hip = keypoints[:, 12, :3] # (num_frames, 3)
        hip_center = (left_hip + right_hip) / 2 # (num_frames, 3)

        # Calculate shoulder center for scaling
        left_shoulder = keypoints[:, 23, :3]
        right_shoulder = keypoints[:, 24, :3]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # Calculate hip-to-shoulder distance for scaling
        # Use a small epsilon to prevent division by zero
        scale_factor = np.linalg.norm(hip_center - shoulder_center, axis=-1, keepdims=True)
        scale_factor[scale_factor == 0] = 1e-6 # Avoid division by zero

        # Apply translation and scaling
        for i in range(keypoints.shape[0]): # Iterate over frames
            normalized_keypoints[i, :, :3] = (keypoints[i, :, :3] - hip_center[i]) / scale_factor[i]

        return normalized_keypoints

    def augment_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Applies augmentations like noise, dropout, and masking to keypoint sequences.
        Assumes keypoints are (num_frames, num_keypoints, 3) for (x, y, z).
        """
        augmented_keypoints = keypoints.copy()
        num_frames, num_keypoints, _ = keypoints.shape

        # Add Gaussian noise
        if self.config.noise.enabled:
            noise = np.random.normal(0, self.config.noise.std_dev, augmented_keypoints[:, :, :3].shape)
            augmented_keypoints[:, :, :3] += noise

        # Keypoint dropout
        if self.config.dropout.enabled:
            dropout_mask = np.random.rand(num_frames, num_keypoints) < self.config.dropout.rate
            # Set dropped keypoints to zero or a sentinel value
            augmented_keypoints[dropout_mask, :3] = 0.0 # Setting x,y,z to 0

        # Temporal masking (mask out sections of time)
        if self.config.masking.enabled and num_frames > self.config.masking.max_mask_length:
            mask_length = np.random.randint(1, self.config.masking.max_mask_length + 1)
            start_frame = np.random.randint(0, num_frames - mask_length)
            augmented_keypoints[start_frame : start_frame + mask_length, :, :3] = 0.0 # Masking x,y,z

        return augmented_keypoints

    def process(self, keypoints: np.ndarray, is_train: bool) -> np.ndarray:
        """
        Applies normalization and optionally augmentation.
        """
        processed_keypoints = self.normalize_keypoints(keypoints)
        if is_train and self.config.augmentations.enabled:
            processed_keypoints = self.augment_keypoints(processed_keypoints)
        return processed_keypoints.reshape(processed_keypoints.shape[0], -1) # Flatten to (num_frames, num_keypoints * 3)
