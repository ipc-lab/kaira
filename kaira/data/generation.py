"""Data generation utilities for Kaira.

This module provides functions for generating various types of data tensors commonly used in
communication systems and information theory experiments.
"""

from typing import List, Optional, Union

import torch
from torch.utils.data import Dataset


def create_binary_tensor(
    size: Union[List[int], torch.Size],
    prob: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a random binary tensor with specified probability of 1s.

    Args:
        size: Shape of the tensor to generate
        prob: Probability of generating 1s (default: 0.5 for uniform distribution)
        device: Device to create the tensor on (default: None, uses default device)

    Returns:
        A binary tensor with random 0s and 1s according to the specified probability
    """
    return torch.bernoulli(torch.full(size, prob, device=device))


def create_uniform_tensor(
    size: Union[List[int], torch.Size],
    low: float = 0.0,
    high: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a tensor with uniformly distributed random values.

    Args:
        size: Shape of the tensor to generate
        low: Lower bound of the uniform distribution (inclusive)
        high: Upper bound of the uniform distribution (exclusive)
        device: Device to create the tensor on (default: None, uses default device)

    Returns:
        A tensor with random values uniformly distributed between low and high
    """
    return torch.rand(size, device=device) * (high - low) + low


class BinaryTensorDataset(Dataset):
    """Dataset of randomly generated binary tensors.

    Creates a dataset where each sample is a binary tensor generated with a specified
    probability of 1s. Useful for simulating binary sources or discrete channels in
    communication systems experiments.

    Attributes:
        data: The generated binary tensor data with shape (n_samples, *feature_dims)
    """

    def __init__(self, size: Union[List[int], torch.Size], prob: float = 0.5, device: Optional[torch.device] = None):
        """Initialize the binary tensor dataset.

        Args:
            size: Shape of the tensor to generate, where the first dimension represents
                 the number of samples in the dataset
            prob: Probability of generating 1s (default: 0.5 for uniform distribution)
            device: Device to create the tensor on (default: None, uses default device)
        """
        self.data = create_binary_tensor(size, prob, device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class UniformTensorDataset(Dataset):
    """Dataset of uniformly distributed random tensors.

    Creates a dataset where each sample is a tensor with values uniformly distributed
    between specified bounds. Useful for simulating continuous sources or analog signals
    in communication systems and information theory experiments.

    Attributes:
        data: The generated uniform tensor data with shape (n_samples, *feature_dims)
    """

    def __init__(self, size: Union[List[int], torch.Size], low: float = 0.0, high: float = 1.0, device: Optional[torch.device] = None):
        """Initialize the uniform tensor dataset.

        Args:
            size: Shape of the tensor to generate, where the first dimension represents
                 the number of samples in the dataset
            low: Lower bound of the uniform distribution (inclusive)
            high: Upper bound of the uniform distribution (exclusive)
            device: Device to create the tensor on (default: None, uses default device)
        """
        self.data = create_uniform_tensor(size, low, high, device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]
