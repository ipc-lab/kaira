"""HuggingFace-compatible sample dataset implementations for Kaira.

This module provides sample dataset classes that are compatible with HuggingFace datasets and
PyTorch DataLoader for standard test images and popular ML datasets.
"""

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SampleImagesDataset(Dataset):
    """Dataset for sample test images compatible with PyTorch and HuggingFace.

    This dataset provides access to commonly used test images in image processing research such as
    coins, astronaut, coffee, and camera images from scikit-image.
    """

    # Standard test images often used in image processing
    TEST_IMAGES = {
        "coins.png": ("https://raw.githubusercontent.com/scikit-image/scikit-image/" "v0.21.0/skimage/data/coins.png"),
        "astronaut.png": ("https://raw.githubusercontent.com/scikit-image/scikit-image/" "v0.21.0/skimage/data/astronaut.png"),
        "coffee.png": ("https://raw.githubusercontent.com/scikit-image/scikit-image/" "v0.21.0/skimage/data/coffee.png"),
        "camera.png": ("https://raw.githubusercontent.com/scikit-image/scikit-image/" "v0.21.0/skimage/data/camera.png"),
    }

    def __init__(
        self,
        n_samples: int = 4,
        target_size: Optional[tuple] = (128, 128),
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the sample image dataset.

        Args:
            n_samples: Number of samples to include in dataset (max 4)
            target_size: Target size (H, W) for images
            seed: Random seed for reproducibility
            cache_dir: Directory to cache downloaded images
        """
        self.n_samples = min(n_samples, len(self.TEST_IMAGES))
        self.target_size = target_size
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Set up cache directory
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
            self.cache_dir = Path(root_dir) / ".cache" / "sample_images"
        else:
            self.cache_dir = Path(cache_dir)

        # Download and prepare images
        self._download_images()
        self.data = self._load_images()

    def _download_images(self, max_retries: int = 3, delay: int = 1) -> None:
        """Download sample test images if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for filename, url in self.TEST_IMAGES.items():
            output_path = self.cache_dir / filename
            if not output_path.exists():
                print(f"Downloading {filename}...")
                success = False

                for attempt in range(max_retries):
                    try:
                        urllib.request.urlretrieve(url, output_path)  # nosec B310
                        print(f"Saved to {output_path}")
                        success = True
                        break
                    except urllib.error.HTTPError as e:
                        print(f"Attempt {attempt+1}/{max_retries} failed: " f"HTTP Error {e.code}: {e.reason}")
                    except urllib.error.URLError as e:
                        print(f"Attempt {attempt+1}/{max_retries} failed: " f"URL Error: {e.reason}")

                    if attempt < max_retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                if not success:
                    print(f"Failed to download {filename} after {max_retries} attempts.")

    def _load_images(self) -> List[Dict[str, Any]]:
        """Load and preprocess images."""
        data = []
        transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ]
        )

        available_images = list(self.TEST_IMAGES.keys())[: self.n_samples]

        for filename in available_images:
            image_path = self.cache_dir / filename
            if image_path.exists():
                try:
                    # Load image and convert to tensor
                    pil_image = Image.open(image_path).convert("RGB")
                    tensor_image = transform(pil_image)

                    # Convert to numpy for HuggingFace compatibility
                    image_array = tensor_image.numpy()

                    data.append(
                        {
                            "image": image_array,
                            "filename": filename.split(".")[0],  # Remove extension
                            "shape": image_array.shape,
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        if not data:
            raise RuntimeError("No test images could be loaded. Check internet connection.")

        return data

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'image', 'filename', and 'shape' keys
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]


class TorchVisionDataset(Dataset):
    """Dataset wrapper for popular ML datasets compatible with PyTorch and HuggingFace.

    This dataset provides access to common datasets like CIFAR-10, CIFAR-100, and MNIST with
    HuggingFace-style interface.
    """

    def __init__(
        self,
        dataset_name: str = "cifar10",
        n_samples: int = 100,
        train: bool = True,
        target_size: Optional[tuple] = None,
        normalize: bool = False,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the TorchVision dataset.

        Args:
            dataset_name: Name of dataset ("cifar10", "cifar100", "mnist")
            n_samples: Number of samples to include in dataset
            train: Whether to use training or test split
            target_size: Target size (H, W) for images
            normalize: Whether to normalize images
            seed: Random seed for reproducibility
            cache_dir: Directory to cache downloaded data
        """
        self.dataset_name = dataset_name.lower()
        self.n_samples = n_samples
        self.train = train
        self.target_size = target_size
        self.normalize = normalize
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Set up cache directory
        if cache_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
            self.cache_dir = Path(root_dir) / ".cache" / "data"
        else:
            self.cache_dir = Path(cache_dir)

        # Load and prepare data
        self.data = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset and prepare samples."""
        # Define transforms
        transform_list = [transforms.ToTensor()]
        if self.target_size is not None:
            transform_list.insert(0, transforms.Resize(self.target_size))
        if self.normalize:
            # Add standard normalization for common datasets
            if self.dataset_name in ["cifar10", "cifar100"]:
                transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            elif self.dataset_name == "mnist":
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

        transform = transforms.Compose(transform_list)

        # Load the appropriate dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root=str(self.cache_dir), train=self.train, download=True, transform=transform)
        elif self.dataset_name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root=str(self.cache_dir), train=self.train, download=True, transform=transform)
        elif self.dataset_name == "mnist":
            dataset = torchvision.datasets.MNIST(root=str(self.cache_dir), train=self.train, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. " f"Choose from 'cifar10', 'cifar100', or 'mnist'")

        # Create subset of data
        indices = torch.randperm(len(dataset))[: self.n_samples]
        data = []

        for idx in indices:
            img, label = dataset[idx]
            # Convert to numpy for HuggingFace compatibility
            image_array = img.numpy()

            data.append(
                {
                    "image": image_array,
                    "label": int(label),
                    "dataset": self.dataset_name,
                    "shape": image_array.shape,
                }
            )

        return data

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary with 'image', 'label', 'dataset', and 'shape' keys
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]


def download_image(url: str, save_dir: str) -> str:
    """Download an image from a URL and save it to a directory.

    Args:
        url: URL of the image to download
        save_dir: Directory to save the downloaded image

    Returns:
        Path to the saved image file

    Raises:
        ValueError: If the URL is not from a trusted HTTPS source
    """
    # Validate URL for security
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed for security reasons")

    # Allow only trusted domains for additional security
    trusted_domains = {
        "raw.githubusercontent.com",
        "github.com",
        "cdn.example.com",
    }

    if parsed_url.netloc not in trusted_domains:
        raise ValueError(f"URL domain {parsed_url.netloc} is not in the list of trusted domains")

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, url.split("/")[-1])

    # Download the image if it doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)  # nosec B310 - URL validated
        print("Download complete.")

    return filename


__all__ = [
    "SampleImagesDataset",
    "TorchVisionDataset",
    "download_image",
]
