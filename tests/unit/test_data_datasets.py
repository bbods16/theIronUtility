import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch, call
from omegaconf import OmegaConf
import torch

from src.data.datasets import SquatFormDataset, SquatFormDatamodule
from src.data.components import KeypointProcessor

# --- Fixtures for SquatFormDataset ---

@pytest.fixture
def mock_keypoint_processor_config():
    return OmegaConf.create({
        "enabled": True,
        "normalization": {"enabled": True, "method": "hip_to_shoulder_distance"},
        "noise": {"enabled": False},
        "dropout": {"enabled": False},
        "masking": {"enabled": False},
        "class_weights": {"enabled": True} # Enable for testing class weights
    })

@pytest.fixture
def mock_label_map():
    return {
        "good_rep": 0,
        "KNEE_VALGUS": 1,
        "NOT_DEEP_ENOUGH": 2,
        "BUTT_WINK": 3,
        "SPINAL_FLEXION": 4,
    }

@pytest.fixture
def mock_labels_csv_content():
    return pd.DataFrame({
        'clip_id': ['clip_A_rep_1', 'clip_A_rep_2', 'clip_B_rep_1', 'clip_C_rep_1', 'clip_C_rep_2'],
        'subject_id': ['subject_A', 'subject_A', 'subject_B', 'subject_C', 'subject_C'],
        'label': ['good_rep', 'KNEE_VALGUS', 'NOT_DEEP_ENOUGH', 'good_rep', 'BUTT_WINK']
    })

@pytest.fixture
def mock_subject_metadata_content():
    return pd.DataFrame({
        'subject_id': ['subject_A', 'subject_B', 'subject_C'],
        'skin_tone': ['Monk_1', 'Monk_3', 'Monk_5'],
        'body_type': ['athletic', 'average', 'heavy']
    })

@pytest.fixture
def mock_keypoint_data():
    # Simulate (num_frames, num_keypoints, 3)
    return np.random.rand(50, 33, 3).astype(np.float32)

# --- Tests for SquatFormDataset ---

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('numpy.load')
@patch('src.data.components.KeypointProcessor.process', side_effect=lambda kps, is_train: kps.reshape(kps.shape[0], -1))
def test_squat_form_dataset_init_and_len(mock_process, mock_np_load, mock_read_csv, mock_exists,
                                         mock_labels_csv_content, mock_subject_metadata_content,
                                         mock_label_map, mock_keypoint_processor_config):
    # Configure mock_read_csv to return different dataframes based on path
    mock_read_csv.side_effect = lambda path: {
        os.path.join("/mock/data", "labels.csv"): mock_labels_csv_content,
        os.path.join("/mock/data", "subject_metadata.csv"): mock_subject_metadata_content
    }[path]

    dataset = SquatFormDataset(
        data_dir="/mock/data",
        label_map=mock_label_map,
        keypoint_processor_config=mock_keypoint_processor_config,
        is_train=True,
        split_subjects=['subject_A', 'subject_B', 'subject_C']
    )
    assert len(dataset) == 5
    # Check that both CSVs were attempted to be read
    mock_read_csv.assert_has_calls([
        call(os.path.join("/mock/data", "labels.csv")),
        call(os.path.join("/mock/data", "subject_metadata.csv"))
    ], any_order=True)


@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('numpy.load')
@patch('src.data.components.KeypointProcessor.process', side_effect=lambda kps, is_train: kps.reshape(kps.shape[0], -1))
def test_squat_form_dataset_getitem(mock_process, mock_np_load, mock_read_csv, mock_exists,
                                    mock_labels_csv_content, mock_subject_metadata_content,
                                    mock_label_map, mock_keypoint_processor_config,
                                    mock_keypoint_data):
    mock_read_csv.side_effect = lambda path: {
        os.path.join("/mock/data", "labels.csv"): mock_labels_csv_content,
        os.path.join("/mock/data", "subject_metadata.csv"): mock_subject_metadata_content
    }[path]
    mock_np_load.return_value = mock_keypoint_data

    dataset = SquatFormDataset(
        data_dir="/mock/data",
        label_map=mock_label_map,
        keypoint_processor_config=mock_keypoint_processor_config,
        is_train=True,
        sequence_length=100,
        split_subjects=['subject_A', 'subject_B', 'subject_C']
    )

    keypoints_tensor, label_tensor, item_metadata = dataset[0] # clip_A_rep_1, good_rep

    assert keypoints_tensor.shape == (100, 33 * 3) # 100 frames, 33 keypoints * 3 coords
    assert label_tensor.item() == mock_label_map["good_rep"]
    assert item_metadata['skin_tone'] == 'Monk_1' # Check metadata is included
    # Use os.path.join for platform-agnostic path comparison
    mock_np_load.assert_called_with(os.path.join("/mock/data", "subject_A", "clip_A_rep_1.npy"))
    mock_process.assert_called_once()

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('numpy.load')
@patch('src.data.components.KeypointProcessor.process', side_effect=lambda kps, is_train: kps.reshape(kps.shape[0], -1))
def test_squat_form_dataset_class_weights(mock_process, mock_np_load, mock_read_csv, mock_exists,
                                          mock_labels_csv_content, mock_subject_metadata_content,
                                          mock_label_map, mock_keypoint_processor_config):
    mock_read_csv.side_effect = lambda path: {
        os.path.join("/mock/data", "labels.csv"): mock_labels_csv_content,
        os.path.join("/mock/data", "subject_metadata.csv"): mock_subject_metadata_content
    }[path]
    dataset = SquatFormDataset(
        data_dir="/mock/data",
        label_map=mock_label_map,
        keypoint_processor_config=mock_keypoint_processor_config,
        is_train=True,
        split_subjects=['subject_A', 'subject_B', 'subject_C']
    )
    weights = dataset.class_weights
    assert weights is not None
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (len(mock_label_map),)
    assert torch.isclose(weights.sum(), torch.tensor(len(mock_label_map), dtype=torch.float32)) # Normalized to sum to num_classes

    # Expected counts: good_rep: 2, KNEE_VALGUS: 1, NOT_DEEP_ENOUGH: 1, BUTT_WINK: 1, SPINAL_FLEXION: 0
    # Total samples: 5
    # Inverse frequencies (unnormalized):
    # good_rep: 5/2 = 2.5
    # KNEE_VALGUS: 5/1 = 5
    # NOT_DEEP_ENOUGH: 5/1 = 5
    # BUTT_WINK: 5/1 = 5
    # SPINAL_FLEXION: 5/0.00001 = 500000 (due to epsilon)
    # Sum of unnormalized: 2.5 + 5 + 5 + 5 + 500000 = ~500017.5
    # Normalized (sum to 5 classes):
    # good_rep: 2.5 / 500017.5 * 5 = ~0.000025
    # KNEE_VALGUS: 5 / 500017.5 * 5 = ~0.00005
    # ...
    # SPINAL_FLEXION: 500000 / 500017.5 * 5 = ~4.9998

    # Let's check the relative order and magnitude for non-zero classes
    # good_rep should have the lowest weight among the non-zero classes
    assert weights[mock_label_map["good_rep"]] < weights[mock_label_map["KNEE_VALGUS"]]
    assert weights[mock_label_map["SPINAL_FLEXION"]] > weights[mock_label_map["good_rep"]] # Spinal flexion has 0 count, so highest weight

# --- Fixtures for SquatFormDatamodule ---

@pytest.fixture
def mock_datamodule_config(mock_keypoint_processor_config):
    return OmegaConf.create({
        "data": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "train_split_ratio": 0.6,
            "val_split_ratio": 0.2,
            "test_split_ratio": 0.2,
            "split_strategy": "subject_stratified",
            "augmentations": mock_keypoint_processor_config,
            "class_weights": {"enabled": True}
        },
        "paths": { # Added paths structure to match actual config
            "data_dir": "/mock/data_root",
            "output_dir": "/mock/output",
            "log_dir": "/mock/output/logs",
            "checkpoint_dir": "/mock/output/checkpoints",
            "model_dir": "/mock/output/models"
        },
        "seed": 42
    })

@pytest.fixture
def mock_squat_form_dataset_instance():
    """Returns a MagicMock configured to act like a SquatFormDataset instance."""
    mock_ds = MagicMock(spec=SquatFormDataset)
    mock_ds.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]) # Mock class weights
    mock_ds.__len__.return_value = 5 # Example length
    # Mock __getitem__ to return (keypoints, label, metadata)
    mock_ds.__getitem__.side_effect = lambda i: (torch.rand(100, 99), torch.tensor(0), {'skin_tone': 'Monk_1'})
    return mock_ds

# --- Tests for SquatFormDatamodule ---

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('src.data.datasets.SquatFormDataset') # Patch the class itself
@patch('src.data.datasets.SquatFormDatamodule._load_subjects_from_file', side_effect=[
    ['subject_A', 'subject_B'], # train subjects
    ['subject_C'], # val subjects
    ['subject_D'] # test subjects (assuming more subjects than in mock_labels_csv_content for split test)
])
def test_squat_form_datamodule_setup_with_files(mock_load_subjects, mock_squat_form_dataset_class, mock_read_csv, mock_exists,
                                     mock_labels_csv_content, mock_datamodule_config,
                                     mock_squat_form_dataset_instance):
    mock_read_csv.return_value = mock_labels_csv_content # Only labels.csv is read if splits are loaded from files
    mock_squat_form_dataset_class.return_value = mock_squat_form_dataset_instance

    datamodule = SquatFormDatamodule(mock_datamodule_config)
    datamodule.setup()

    # Check that datasets were initialized
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Check that SquatFormDataset was called for each split
    call_args_list = mock_squat_form_dataset_class.call_args_list
    assert len(call_args_list) == 3

    # Ensure class weights are correctly assigned from the mocked dataset
    assert datamodule.class_weights is not None
    assert torch.equal(datamodule.class_weights, mock_squat_form_dataset_instance.class_weights)

@patch('pandas.read_csv')
@patch('src.data.datasets.SquatFormDataset')
@patch('src.data.datasets.SquatFormDatamodule._load_subjects_from_file', return_value=None) # Ensure internal split
def test_squat_form_datamodule_setup_internal_split(mock_load_subjects, mock_squat_form_dataset_class, mock_read_csv,
                                     mock_labels_csv_content, mock_subject_metadata_content,
                                     mock_datamodule_config, mock_squat_form_dataset_instance):
    # Configure mock_read_csv for internal split (reads labels.csv and subject_metadata.csv)
    mock_read_csv.side_effect = lambda path: {
        os.path.join("/mock/data_root", "processed", "squat_form", "labels.csv"): mock_labels_csv_content,
        os.path.join("/mock/data_root", "processed", "squat_form", "subject_metadata.csv"): mock_subject_metadata_content
    }[path]

    # Patch os.path.exists specifically for this test to control what files are "found"
    with patch('os.path.exists') as mock_os_path_exists:
        mock_os_path_exists.side_effect = lambda path: \
            path == os.path.join("/mock/data_root", "processed", "squat_form", "labels.csv") or \
            path == os.path.join("/mock/data_root", "processed", "squat_form", "subject_metadata.csv")

        mock_squat_form_dataset_class.return_value = mock_squat_form_dataset_instance

        datamodule = SquatFormDatamodule(mock_datamodule_config)
        datamodule.setup()

        # Check that datasets were initialized
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None

        # Check that SquatFormDataset was called for each split
        call_args_list = mock_squat_form_dataset_class.call_args_list
        assert len(call_args_list) == 3

        # Ensure class weights are correctly assigned from the mocked dataset
        assert datamodule.class_weights is not None
        assert torch.equal(datamodule.class_weights, mock_squat_form_dataset_instance.class_weights)


@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('src.data.datasets.SquatFormDataset')
@patch('src.data.datasets.SquatFormDatamodule._load_subjects_from_file', side_effect=[
    ['subject_A', 'subject_B'], # train subjects
    ['subject_C'], # val subjects
    ['subject_D'] # test subjects
])
def test_squat_form_datamodule_dataloaders(mock_load_subjects, mock_squat_form_dataset_class, mock_read_csv, mock_exists,
                                           mock_labels_csv_content, mock_datamodule_config,
                                           mock_squat_form_dataset_instance):
    mock_read_csv.return_value = mock_labels_csv_content
    mock_squat_form_dataset_class.return_value = mock_squat_form_dataset_instance

    datamodule = SquatFormDatamodule(mock_datamodule_config)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    # Check batch size and data shape by iterating one batch
    batch_x, batch_y, batch_meta = next(iter(train_loader))
    assert batch_x.shape == (mock_datamodule_config.data.batch_size, 100, 99)
    assert batch_y.shape == (mock_datamodule_config.data.batch_size,)
    assert isinstance(batch_meta, dict)
    assert 'skin_tone' in batch_meta
