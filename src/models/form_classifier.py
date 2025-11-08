import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

class Chomp1d(nn.Module):
    """
    Custom module to remove padding from the end of a sequence.
    Used in TCNs to ensure causality.
    """
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float) -> None:
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # Use custom Chomp1d
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding) # Use custom Chomp1d
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence classification.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int],
                 kernel_size: int, num_layers: int, dropout: float) -> None:
        super(TemporalConvolutionalNetwork, self).__init__()

        layers = []
        num_channels = [input_dim] + hidden_dims # Input dim is the first channel count

        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_channels[i]
            out_channels = num_channels[i+1] if i < num_layers -1 else hidden_dims[-1] # Ensure last block outputs last hidden_dim
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dims[-1], num_classes) # Final FC layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        # TCN expects (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)

        # Apply TCN layers
        tcn_output = self.tcn(x)

        # Global average pooling over the sequence dimension
        # (batch_size, hidden_dims[-1], sequence_length) -> (batch_size, hidden_dims[-1])
        pooled_output = F.adaptive_avg_pool1d(tcn_output, 1).squeeze(-1)

        # Final classification layer
        logits = self.fc(pooled_output)
        return logits

class FormClassifier(nn.Module):
    """
    Wrapper for the Form Classification model, integrating the TCN.
    """
    def __init__(self, classifier: Dict[str, Any], num_classes: int, input_dim: int) -> None:
        super(FormClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        if classifier['type'] == 'TCN':
            self.model = TemporalConvolutionalNetwork(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=classifier['hidden_dims'],
                kernel_size=classifier['kernel_size'],
                num_layers=classifier['num_layers'],
                dropout=classifier['dropout']
            )
        elif classifier['type'] in ['GRU', 'LSTM']:
            # Placeholder for RNN implementation
            # For now, we'll raise an error if not TCN
            raise NotImplementedError(f"RNN type {classifier['type']} not yet implemented. Use TCN.")
        else:
            raise ValueError(f"Unsupported classifier type: {classifier['type']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
