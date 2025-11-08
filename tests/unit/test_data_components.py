import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.components import KeypointProcessor

@pytest.fixture
def sample_keypoints():
    """
    Generates a sample keypoint sequence (num_frames, num_keypoints, 3).
    Using 33 keypoints as per MediaPipe BlazePose.
    """
    num_frames = 10
    num_keypoints = 33
    # Simulate some movement, e.g., hip moving
    keypoints = np.random.rand(num_frames, num_keypoints, 3).astype(np.float32) * 100

    # Ensure hip and shoulder keypoints are somewhat realistic for testing normalization
    # Left hip (11), Right hip (12), Left shoulder (23), Right shoulder (24)
    # Make hips slightly below shoulders
    keypoints[:, 11, 1] = 50 # y-coord for left hip
    keypoints[:, 12, 1] = 50 # y-coord for right hip
    keypoints[:, 23, 1] = 20 # y-coord for left shoulder
    keypoints[:, 24, 1] = 20 # y-coord for right shoulder

    # Simulate hip movement over frames
    keypoints[:, 11:13, 1] += np.linspace(0, 10, num_frames)[:, None] # Hips move down
    keypoints[:, 23:25, 1] += np.linspace(0, 5, num_frames)[:, None] # Shoulders move down less

    return keypoints

@pytest.fixture
def default_processor_config():
    """Default configuration for KeypointProcessor."""
    return OmegaConf.create({
        "augmentations": {
            "enabled": True,
            "normalization": {"enabled": True, "method": "hip_to_shoulder_distance"},
            "noise": {"enabled": True, "std_dev": 0.01},
            "dropout": {"enabled": True, "rate": 0.05},
            "masking": {"enabled": True, "max_mask_length": 5}
        },
        "class_weights": {"enabled": False} # Not relevant for processor tests
    })

@pytest.fixture
def no_aug_processor_config():
    """Configuration with augmentations disabled."""
    return OmegaConf.create({
        "augmentations": {
            "enabled": False,
            "normalization": {"enabled": True, "method": "hip_to_shoulder_distance"},
            "noise": {"enabled": False, "std_dev": 0.01},
            "dropout": {"enabled": False, "rate": 0.05},
            "masking": {"enabled": False, "max_mask_length": 5}
        },
        "class_weights": {"enabled": False}
    })

def test_keypoint_processor_initialization(default_processor_config):
    processor = KeypointProcessor(default_processor_config.augmentations)
    assert processor is not None
    assert processor.config == default_processor_config.augmentations

def test_normalize_keypoints_output_shape(sample_keypoints, default_processor_config):
    processor = KeypointProcessor(default_processor_config.augmentations)
    normalized_kps = processor.normalize_keypoints(sample_keypoints)
    assert normalized_kps.shape == sample_keypoints.shape
    assert normalized_kps.dtype == sample_keypoints.dtype

def test_normalize_keypoints_invariance(sample_keypoints, default_processor_config):
    processor = KeypointProcessor(default_processor_config.augmentations)

    # Test translation invariance: shift all keypoints
    shifted_keypoints = sample_keypoints + np.array([1000, 2000, 3000])
    normalized_original = processor.normalize_keypoints(sample_keypoints)
    normalized_shifted = processor.normalize_keypoints(shifted_keypoints)

    # Check if normalized results are approximately the same
    np.testing.assert_allclose(normalized_original, normalized_shifted, atol=1e-5)

    # Test scale invariance: scale all keypoints
    scaled_keypoints = sample_keypoints * 2
    normalized_scaled = processor.normalize_keypoints(scaled_keypoints)

    # Check if normalized results are approximately the same
    np.testing.assert_allclose(normalized_original, normalized_scaled, atol=1e-5)

def test_augment_keypoints_noise(sample_keypoints, default_processor_config):
    config_with_noise = OmegaConf.create({
        "enabled": True,
        "normalization": {"enabled": False}, # Disable for this specific test
        "noise": {"enabled": True, "std_dev": 0.1},
        "dropout": {"enabled": False},
        "masking": {"enabled": False}
    })
    processor = KeypointProcessor(config_with_noise)
    augmented_kps = processor.augment_keypoints(sample_keypoints)

    # Noise should change the values
    assert not np.allclose(augmented_kps, sample_keypoints)
    assert augmented_kps.shape == sample_keypoints.shape

def test_augment_keypoints_dropout(sample_keypoints, default_processor_config):
    config_with_dropout = OmegaConf.create({
        "enabled": True,
        "normalization": {"enabled": False},
        "noise": {"enabled": False},
        "dropout": {"enabled": True, "rate": 0.5}, # High rate to ensure some dropout
        "masking": {"enabled": False}
    })
    processor = KeypointProcessor(config_with_dropout)
    augmented_kps = processor.augment_keypoints(sample_keypoints)

    # Dropout should set some keypoints to 0.0
    assert np.any(augmented_kps[:, :, :3] == 0.0)
    assert augmented_kps.shape == sample_keypoints.shape

def test_augment_keypoints_masking(sample_keypoints, default_processor_config):
    config_with_masking = OmegaConf.create({
        "enabled": True,
        "normalization": {"enabled": False},
        "noise": {"enabled": False},
        "dropout": {"enabled": False},
        "masking": {"enabled": True, "max_mask_length": 2}
    })
    processor = KeypointProcessor(config_with_masking)
    augmented_kps = processor.augment_keypoints(sample_keypoints)

    # Masking should set a temporal slice of keypoints to 0.0
    assert np.any(augmented_kps[:, :, :3] == 0.0)
    assert augmented_kps.shape == sample_keypoints.shape

def test_process_method_train_with_aug(sample_keypoints, default_processor_config):
    processor = KeypointProcessor(default_processor_config.augmentations)
    processed_kps = processor.process(sample_keypoints, is_train=True)

    # Output should be flattened (num_frames, num_keypoints * 3)
    assert processed_kps.shape == (sample_keypoints.shape[0], sample_keypoints.shape[1] * 3)
    # Should be different from original due to normalization and augmentation
    assert not np.allclose(processed_kps.reshape(sample_keypoints.shape), sample_keypoints, atol=1e-5)

def test_process_method_eval_no_aug(sample_keypoints, no_aug_processor_config):
    processor = KeypointProcessor(no_aug_processor_config.augmentations)
    processed_kps = processor.process(sample_keypoints, is_train=False)

    # Output should be flattened (num_frames, num_keypoints * 3)
    assert processed_kps.shape == (sample_keypoints.shape[0], sample_keypoints.shape[1] * 3)

    # Should be different from original due to normalization, but not augmentation
    # To properly test this, we'd need to compare against a manually normalized version
    # For now, just check it's not identical to raw input
    assert not np.allclose(processed_kps.reshape(sample_keypoints.shape), sample_keypoints, atol=1e-5)

def test_process_method_no_normalization(sample_keypoints):
    config_no_norm = OmegaConf.create({
        "enabled": True,
        "normalization": {"enabled": False},
        "noise": {"enabled": False},
        "dropout": {"enabled": False},
        "masking": {"enabled": False}
    })
    processor = KeypointProcessor(config_no_norm)
    processed_kps = processor.process(sample_keypoints, is_train=False)

    # If no normalization and no augmentation, output should be close to original (flattened)
    np.testing.assert_allclose(processed_kps, sample_keypoints.reshape(sample_keypoints.shape[0], -1), atol=1e-5)
