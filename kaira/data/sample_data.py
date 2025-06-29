"""Simple image dataset utilities for Kaira.

This module provides basic image dataset functionality for testing and examples.
"""

from typing import Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset


class ImageDataset(Dataset):
    """Simple wrapper for common image datasets.

    Provides easy access to CIFAR-10, CIFAR-100, and MNIST datasets with consistent interface and
    optional preprocessing.
    """

    def __init__(
        self,
        name: str = "cifar10",
        train: bool = True,
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        root: str = "~/.cache/kaira",
    ):
        """Initialize the image dataset.

        Args:
            name: Dataset name ("cifar10", "cifar100", "mnist")
            train: Whether to use training split
            size: Target image size (H, W). If None, uses original size
            normalize: Whether to normalize images to [0, 1]
            root: Root directory for dataset storage
        """
        self.name = name.lower()

        # Build transforms
        transform_list = []
        if size is not None:
            transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.ToTensor())
        if not normalize:
            # Convert back to [0, 255] range if normalization is disabled
            transform_list.append(transforms.Lambda(lambda x: x * 255))

        transform = transforms.Compose(transform_list)

        # Load dataset
        if self.name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        elif self.name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        elif self.name == "mnist":
            self.dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        return self.dataset[idx]

    def subset(self, size: int, seed: Optional[int] = None) -> "Subset":
        """Create a random subset of the dataset.

        Args:
            size: Number of samples in the subset
            seed: Random seed for reproducibility

        Returns:
            Subset of the dataset
        """
        if seed is not None:
            torch.manual_seed(seed)

        indices = torch.randperm(len(self))[:size]
        return Subset(self, indices)


__all__ = ["ImageDataset"]
