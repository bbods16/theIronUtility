import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import MagicMock, patch
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
        "knees_caving": 1,
        "not_deep_enough": 2,
        "butt_wink": 3,
        "spinal_flexion": 4,
    }

@pytest.fixture
def mock_labels_csv_content():
    return pd.DataFrame({
        'clip_id': ['clip_A_rep_1', 'clip_A_rep_2', 'clip_B_rep_1', 'clip_C_rep_1', 'clip_C_rep_2'],
        'subject_id': ['subject_A', 'subject_A', 'subject_B', 'subject_C', 'subject_C'],
        'label': ['good_rep', 'knees_caving', 'not_deep_enough', 'good_rep', 'butt_wink']
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
                                         mock_labels_csv_content, mock_label_map, mock_keypoint_processor_config):
    mock_read_csv.return_value = mock_labels_csv_content
    dataset = SquatFormDataset(
        data_dir="/mock/data",
        label_map=mock_label_map,
        keypoint_processor_config=mock_keypoint_processor_config,
        is_train=True,
        split_subjects=['subject_A', 'subject_B', 'subject_C']
    )
    assert len(dataset) == 5
    mock_read_csv.assert_called_once_with("/mock/data/labels.csv")

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('numpy.load')
@patch('src.data.components.KeypointProcessor.process', side_effect=lambda kps, is_train: kps.reshape(kps.shape[0], -1))
def test_squat_form_dataset_getitem(mock_process, mock_np_load, mock_read_csv, mock_exists,
                                    mock_labels_csv_content, mock_label_map, mock_keypoint_processor_config,
                                    mock_keypoint_data):
    mock_read_csv.return_value = mock_labels_csv_content
    mock_np_load.return_value = mock_keypoint_data

    dataset = SquatFormDataset(
        data_dir="/mock/data",
        label_map=mock_label_map,
        keypoint_processor_config=mock_keypoint_processor_config,
        is_train=True,
        sequence_length=100,
        split_subjects=['subject_A', 'subject_B', 'subject_C']
    )

    keypoints_tensor, label_tensor = dataset[0] # clip_A_rep_1, good_rep

    assert keypoints_tensor.shape == (100, 33 * 3) # 100 frames, 33 keypoints * 3 coords
    assert label_tensor.item() == mock_label_map["good_rep"]
    mock_np_load.assert_called_with("/mock/data/subject_A/clip_A_rep_1.npy")
    mock_process.assert_called_once()

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('numpy.load')
@patch('src.data.components.KeypointProcessor.process', side_effect=lambda kps, is_train: kps.reshape(kps.shape[0], -1))
def test_squat_form_dataset_class_weights(mock_process, mock_np_load, mock_read_csv, mock_exists,
                                          mock_labels_csv_content, mock_label_map, mock_keypoint_processor_config):
    mock_read_csv.return_value = mock_labels_csv_content
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

    # Expected counts: good_rep: 2, knees_caving: 1, not_deep_enough: 1, butt_wink: 1, spinal_flexion: 0
    # Total samples: 5
    # Inverse frequencies (unnormalized):
    # good_rep: 5/2 = 2.5
    # knees_caving: 5/1 = 5
    # not_deep_enough: 5/1 = 5
    # butt_wink: 5/1 = 5
    # spinal_flexion: 5/0.00001 = 500000 (due to epsilon)
    # Sum of unnormalized: 2.5 + 5 + 5 + 5 + 500000 = ~500017.5
    # Normalized (sum to 5 classes):
    # good_rep: 2.5 / 500017.5 * 5 = ~0.000025
    # knees_caving: 5 / 500017.5 * 5 = ~0.00005
    # ...
    # spinal_flexion: 500000 / 500017.5 * 5 = ~4.9998

    # Let's check the relative order and magnitude for non-zero classes
    # good_rep should have the lowest weight among the non-zero classes
    assert weights[mock_label_map["good_rep"]] < weights[mock_label_map["knees_caving"]]
    assert weights[mock_label_map["spinal_flexion"]] > weights[mock_label_map["good_rep"]] # Spinal flexion has 0 count, so highest weight

# --- Fixtures for SquatFormDatamodule ---

@pytest.fixture
def mock_datamodule_config(mock_keypoint_processor_config):
    return OmegaConf.create({
        "data": {
            "dataset_dir": "/mock/data",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "train_split_ratio": 0.6,
            "val_split_ratio": 0.2,
            "test_split_ratio": 0.2,
            "split_strategy": "subject_stratified",
            "augmentations": mock_keypoint_processor_config.augmentations,
            "class_weights": {"enabled": True}
        },
        "seed": 42
    })

# --- Tests for SquatFormDatamodule ---

@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('src.data.datasets.SquatFormDataset.__init__', return_value=None) # Mock dataset init
@patch('src.data.datasets.SquatFormDataset._calculate_class_weights', return_value=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
def test_squat_form_datamodule_setup(mock_calc_weights, mock_dataset_init, mock_read_csv, mock_exists,
                                     mock_labels_csv_content, mock_datamodule_config):
    mock_read_csv.return_value = mock_labels_csv_content
    datamodule = SquatFormDatamodule(mock_datamodule_config)
    datamodule.setup()

    # Check that datasets were initialized
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    # Check that SquatFormDataset.__init__ was called for each split
    # There are 3 unique subjects in mock_labels_csv_content: A, B, C
    # With 0.6/0.2/0.2 split, it should be 1 train, 1 val, 1 test subject
    # The exact subjects depend on shuffling, but the counts should be correct.
    call_args_list = mock_dataset_init.call_args_list
    assert len(call_args_list) == 3

    train_subjects = call_args_list[0].kwargs['split_subjects']
    val_subjects = call_args_list[1].kwargs['split_subjects']
    test_subjects = call_args_list[2].kwargs['split_subjects']

    # Ensure no overlap between splits
    assert not set(train_subjects).intersection(val_subjects)
    assert not set(train_subjects).intersection(test_subjects)
    assert not set(val_subjects).intersection(test_subjects)

    # Ensure all subjects are covered
    all_split_subjects = set(train_subjects + val_subjects + test_subjects)
    assert all_split_subjects == set(['subject_A', 'subject_B', 'subject_C'])

    assert datamodule.class_weights is not None
    assert torch.equal(datamodule.class_weights, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))


@patch('os.path.exists', return_value=True)
@patch('pandas.read_csv')
@patch('src.data.datasets.SquatFormDataset.__init__', return_value=None)
@patch('src.data.datasets.SquatFormDataset._calculate_class_weights', return_value=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))
def test_squat_form_datamodule_dataloaders(mock_calc_weights, mock_dataset_init, mock_read_csv, mock_exists,
                                           mock_labels_csv_content, mock_datamodule_config):
    mock_read_csv.return_value = mock_labels_csv_content
    datamodule = SquatFormDatamodule(mock_datamodule_config)
    datamodule.setup()

    # Mock the datasets to return dummy data for dataloader iteration
    mock_train_dataset = MagicMock(spec=SquatFormDataset)
    mock_train_dataset.__len__.return_value = 5
    mock_train_dataset.__getitem__.side_effect = lambda i: (torch.rand(100, 99), torch.tensor(0))
    mock_train_dataset.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    mock_val_dataset = MagicMock(spec=SquatFormDataset)
    mock_val_dataset.__len__.return_value = 2
    mock_val_dataset.__getitem__.side_effect = lambda i: (torch.rand(100, 99), torch.tensor(1))
    mock_val_dataset.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    mock_test_dataset = MagicMock(spec=SquatFormDataset)
    mock_test_dataset.__len__.return_value = 2
    mock_test_dataset.__getitem__.side_effect = lambda i: (torch.rand(100, 99), torch.tensor(2))
    mock_test_dataset.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])


    datamodule.train_dataset = mock_train_dataset
    datamodule.val_dataset = mock_val_dataset
    datamodule.test_dataset = mock_test_dataset

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    # Check batch size and data shape
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (mock_datamodule_config.data.batch_size, 100, 99)
    assert batch_y.shape == (mock_datamodule_config.data.batch_size,)
