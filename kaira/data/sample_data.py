"""Utilities for loading sample data, such as standard test images."""

import os
from typing import Literal, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms


def load_sample_images(dataset: Literal["cifar10", "cifar100", "mnist"] = "cifar10", num_samples: int = 4, seed: Optional[int] = None, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load sample images from popular datasets for demonstrations.

    This function provides easy access to sample images from standard datasets like
    CIFAR-10, CIFAR-100, and MNIST for demonstration purposes.

    Args:
        dataset: Name of the dataset to sample from ('cifar10', 'cifar100', 'mnist')
        num_samples: Number of sample images to return
        seed: Random seed for reproducibility
        normalize: Whether to normalize the images to [0,1] range

    Returns:
        Tuple containing:
            - Tensor of images with shape (num_samples, C, H, W)
            - Tensor of labels with shape (num_samples,)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Define transforms
    if normalize:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # Load the appropriate dataset
    # Get the root library directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the root library directory (two levels up)
    root_library_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    root_path = os.path.join(root_library_dir, ".cache", "data")
    os.makedirs(root_path, exist_ok=True)

    if dataset.lower() == "cifar10":
        data = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
    elif dataset.lower() == "cifar100":
        data = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=transform)
    elif dataset.lower() == "mnist":
        data = torchvision.datasets.MNIST(root=root_path, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose from 'cifar10', 'cifar100', or 'mnist'")

    # Create a subset of the data
    indices = torch.randperm(len(data))[:num_samples]
    images = []
    labels = []

    for idx in indices:
        img, label = data[idx]
        images.append(img)
        labels.append(label)

    # Stack into batches
    images = torch.stack(images)
    labels = torch.tensor(labels)

    return images, labels
