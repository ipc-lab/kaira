"""Simple and efficient dataset implementations for Kaira.

This module provides dataset classes for communication systems and information theory experiments.
All datasets generate data on-demand for memory efficiency and support PyTorch DataLoader.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    """Dataset for binary tensor data with configurable probability.

    Generates binary tensors on-demand with specified probability of 1s.
    Useful for digital communication and coding theory experiments.
    """

    def __init__(
        self,
        length: int,
        shape: Union[int, Tuple[int, ...]] = (128,),
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """Initialize the binary dataset.

        Args:
            length: Number of samples in the dataset
            shape: Shape of each tensor (int for 1D, tuple for multi-dimensional)
            prob: Probability of generating 1s (default: 0.5)
            seed: Random seed for reproducibility
        """
        self.length = length
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.prob = prob
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a binary tensor sample.

        Args:
            idx: Index of the sample (used for deterministic generation)

        Returns:
            Binary tensor with values 0 or 1
        """
        # Use index as additional seed for deterministic generation
        local_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        data = local_rng.binomial(1, self.prob, size=self.shape).astype(np.float32)
        return torch.from_numpy(data)

class UniformDataset(Dataset):
    """Dataset for uniformly distributed tensor data.

    Generates tensors with uniformly distributed random values on-demand.
    Useful for noise generation and random signal experiments.
    """

    def __init__(
        self,
        length: int,
        shape: Union[int, Tuple[int, ...]] = (128,),
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the uniform dataset.

        Args:
            length: Number of samples in the dataset
            shape: Shape of each tensor (int for 1D, tuple for multi-dimensional)
            low: Lower bound for uniform distribution
            high: Upper bound for uniform distribution
            seed: Random seed for reproducibility
        """
        self.length = length
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.low = low
        self.high = high
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a uniform tensor sample.

        Args:
            idx: Index of the sample (used for deterministic generation)

        Returns:
            Tensor with uniformly distributed values
        """
        # Use index as additional seed for deterministic generation
        local_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        data = local_rng.uniform(self.low, self.high, size=self.shape).astype(np.float32)
        return torch.from_numpy(data)


class GaussianDataset(Dataset):
    """Dataset for Gaussian distributed tensor data.

    Generates tensors with Gaussian distributed random values on-demand.
    Useful for noise modeling and channel simulation.
    """

    def __init__(
        self,
        length: int,
        shape: Union[int, Tuple[int, ...]] = (128,),
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the Gaussian dataset.

        Args:
            length: Number of samples in the dataset
            shape: Shape of each tensor (int for 1D, tuple for multi-dimensional)
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
            seed: Random seed for reproducibility
        """
        self.length = length
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.mean = mean
        self.std = std
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate a Gaussian tensor sample.

        Args:
            idx: Index of the sample (used for deterministic generation)

        Returns:
            Tensor with Gaussian distributed values
        """
        # Use index as additional seed for deterministic generation
        local_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        data = local_rng.normal(self.mean, self.std, size=self.shape).astype(np.float32)
        return torch.from_numpy(data)


class CorrelatedDataset(Dataset):
    """Dataset for correlated data pairs.

    Generates pairs of correlated tensors useful for Wyner-Ziv coding,
    side information experiments, and correlation modeling.
    """

    def __init__(
        self,
        length: int,
        shape: Union[int, Tuple[int, ...]] = (128,),
        correlation: float = 0.8,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize the correlated dataset.

        Args:
            length: Number of samples in the dataset
            shape: Shape of each tensor (int for 1D, tuple for multi-dimensional)
            correlation: Correlation coefficient between source and side info (0-1)
            noise_std: Standard deviation of noise added to create correlation
            seed: Random seed for reproducibility
        """
        self.length = length
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.correlation = correlation
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a correlated tensor pair.

        Args:
            idx: Index of the sample (used for deterministic generation)

        Returns:
            Tuple of (source, side_info) tensors
        """
        # Use index as additional seed for deterministic generation
        local_rng = np.random.RandomState(self.rng.randint(0, 2**31) + idx)
        
        # Generate source signal
        source = local_rng.normal(0, 1, size=self.shape).astype(np.float32)
        
        # Generate independent noise for side information
        noise = local_rng.normal(0, 1, size=self.shape).astype(np.float32)
        
        # Create correlated side information using the standard formula
        side_info = (self.correlation * source + np.sqrt(1 - self.correlation**2) * noise).astype(np.float32)
        
        return torch.from_numpy(source), torch.from_numpy(side_info)


class FunctionDataset(Dataset):
    """Dataset that applies a custom function to generate data.

    Flexible dataset for custom data generation using user-provided functions.
    Useful for complex signal generation and custom experiments.
    """

    def __init__(
        self,
        length: int,
        generator_fn: Callable[[int], torch.Tensor],
        seed: Optional[int] = None,
    ):
        """Initialize the function dataset.

        Args:
            length: Number of samples in the dataset
            generator_fn: Function that takes an index and returns a tensor
            seed: Random seed for reproducibility
        """
        self.length = length
        self.generator_fn = generator_fn
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Generate data using the custom function.

        Args:
            idx: Index of the sample

        Returns:
            Tensor generated by the custom function
        """
        return self.generator_fn(idx)


__all__ = [
    "BinaryDataset",
    "UniformDataset", 
    "GaussianDataset",
    "CorrelatedDataset",
    "FunctionDataset",
]
