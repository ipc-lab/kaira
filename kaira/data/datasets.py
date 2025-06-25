"""HuggingFace-compatible dataset implementations for Kaira.

This module provides dataset classes that are compatible with HuggingFace datasets and PyTorch
DataLoader for communication systems.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class BinaryTensorDataset(Dataset):
    """Dataset for random binary tensor data compatible with PyTorch and HuggingFace.

    This dataset generates binary tensors with specified probability of 1s. It implements the
    standard PyTorch Dataset interface with __len__ and __getitem__.
    """

    def __init__(
        self,
        n_samples: int,
        feature_shape: Union[List[int], tuple, int] = (128,),
        prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """Initialize the binary tensor dataset.

        Args:
            n_samples: Number of samples in the dataset
            feature_shape: Shape of each feature tensor
            prob: Probability of generating 1s (default: 0.5)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.feature_shape = feature_shape if isinstance(feature_shape, (list, tuple)) else (feature_shape,)
        self.prob = prob

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate all data for consistency
        self.data = []
        for _ in range(n_samples):
            tensor = np.random.binomial(1, prob, size=self.feature_shape).astype(np.float32)
            self.data.append(tensor)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'data' key containing the binary tensor
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return {"data": self.data[idx]}


class UniformTensorDataset(Dataset):
    """Dataset for uniform tensor data compatible with PyTorch and HuggingFace.

    This dataset generates tensors with uniformly distributed random values. It implements the
    standard PyTorch Dataset interface with __len__ and __getitem__.
    """

    def __init__(
        self,
        n_samples: int,
        feature_shape: Union[List[int], tuple, int] = (128,),
        low: float = 0.0,
        high: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize the uniform tensor dataset.

        Args:
            n_samples: Number of samples in the dataset
            feature_shape: Shape of each feature tensor
            low: Lower bound for uniform distribution
            high: Upper bound for uniform distribution
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.feature_shape = feature_shape if isinstance(feature_shape, (list, tuple)) else (feature_shape,)
        self.low = low
        self.high = high

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate all data for consistency
        self.data = []
        for _ in range(n_samples):
            tensor = np.random.uniform(low, high, size=self.feature_shape).astype(np.float32)
            self.data.append(tensor)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'data' key containing the uniform tensor
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return {"data": self.data[idx]}


class WynerZivCorrelationDataset(Dataset):
    """Dataset for Wyner-Ziv coding with correlated data.

    This dataset generates correlated source and side information pairs for Wyner-Ziv coding
    experiments. It implements the standard PyTorch Dataset interface with __len__ and __getitem__.
    """

    def __init__(
        self,
        n_samples: int,
        feature_shape: Union[List[int], tuple, int] = (128,),
        correlation_type: str = "gaussian",
        correlation_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the Wyner-Ziv correlation dataset.

        Args:
            n_samples: Number of samples in the dataset
            feature_shape: Shape of each feature tensor
            correlation_type: Type of correlation ("gaussian", "binary", or "custom")
            correlation_params: Parameters for correlation model
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.feature_shape = feature_shape if isinstance(feature_shape, (list, tuple)) else (feature_shape,)
        self.correlation_type = correlation_type
        self.correlation_params = correlation_params or {}

        if seed is not None:
            np.random.seed(seed)

        # Pre-generate all data for consistency
        self.data = []
        for _ in range(n_samples):
            source, side_info = self._generate_correlated_pair()
            self.data.append({"source": source, "side_info": side_info})

    def _generate_correlated_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a correlated source and side information pair."""
        if self.correlation_type == "binary":
            source = np.random.binomial(1, 0.5, size=self.feature_shape).astype(np.float32)
            crossover_prob = self.correlation_params.get("crossover_prob", 0.1)
            flip_mask = np.random.binomial(1, crossover_prob, size=self.feature_shape)
            side_info = np.logical_xor(source.astype(bool), flip_mask.astype(bool)).astype(np.float32)
        elif self.correlation_type == "gaussian":
            source = np.random.uniform(0, 1, size=self.feature_shape).astype(np.float32)
            sigma = self.correlation_params.get("sigma", 0.1)
            noise = np.random.normal(0, sigma, size=self.feature_shape).astype(np.float32)
            side_info = source + noise
        elif self.correlation_type == "custom":
            source = np.random.uniform(0, 1, size=self.feature_shape).astype(np.float32)
            transform_fn = self.correlation_params.get("transform_fn")
            if transform_fn is None:
                raise ValueError("Custom correlation type requires 'transform_fn' in correlation_params")
            side_info = transform_fn(source)
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")

        return source, side_info

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'source' and 'side_info' keys containing the correlated tensors
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]


__all__ = [
    "BinaryTensorDataset",
    "UniformTensorDataset",
    "WynerZivCorrelationDataset",
]
