"""Utilities for loading sample data, such as standard test images."""

import os
import time
import urllib.request
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class SampleDataLoader:
    """A unified class for loading both dataset and standard test images.

    This class provides a unified interface for loading images from popular datasets (CIFAR-10,
    CIFAR-100, MNIST) as well as standard test images commonly used in image processing and
    computer vision research.
    """

    # Standard test images often used in image processing
    TEST_IMAGES = {
        "coins.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/coins.png",
        "astronaut.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/astronaut.png",
        "coffee.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/coffee.png",
        "camera.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/camera.png",
    }

    def __init__(self):
        """Initialize the SampleDataLoader."""
        self._cache_dir = self._get_cache_directory()

    def _get_cache_directory(self) -> Path:
        """Get the cache directory for storing downloaded data."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_library_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
        return Path(root_library_dir) / ".cache"

    def _download_test_images(self, max_retries: int = 3, delay: int = 1) -> Path:
        """Download standard test images used in examples.

        Args:
            max_retries: Maximum number of download attempts per image
            delay: Delay in seconds between retry attempts

        Returns:
            Path to the directory containing downloaded images
        """
        images_cache = self._cache_dir / "sample_images"
        images_cache.mkdir(parents=True, exist_ok=True)

        for filename, url in self.TEST_IMAGES.items():
            output_path = images_cache / filename
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
                        print(f"Attempt {attempt+1}/{max_retries} failed: HTTP Error {e.code}: {e.reason}")
                    except urllib.error.URLError as e:
                        print(f"Attempt {attempt+1}/{max_retries} failed: URL Error: {e.reason}")

                    if attempt < max_retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                if not success:
                    print(f"Failed to download {filename} after {max_retries} attempts.")

        return images_cache

    def load_images(
        self, source: Union[Literal["dataset", "test"], str] = "test", dataset: Literal["cifar10", "cifar100", "mnist"] = "cifar10", num_samples: int = 4, seed: Optional[int] = None, target_size: Optional[Tuple[int, int]] = None, normalize: bool = False
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, list]]:
        """Unified interface for loading images from either datasets or test images.

        Args:
            source: Image source - "dataset" for CIFAR/MNIST, "test" for standard test images
            dataset: Dataset name (only used when source="dataset")
            num_samples: Number of sample images to return
            seed: Random seed for reproducibility
            target_size: Target size (H, W) to resize images. If None, uses default for each source.
            normalize: Whether to normalize images (only used for dataset images)

        Returns:
            Tuple containing:
                - Tensor of images with shape (num_samples, C, H, W)
                - For datasets: tensor of labels with shape (num_samples,)
                - For test images: list of image names
        """
        if source == "dataset":
            return self._load_dataset_images(dataset, num_samples, seed, normalize, target_size)
        elif source == "test":
            return self._load_test_images(num_samples, seed, target_size)
        else:
            raise ValueError(f"Invalid source: {source}. Choose 'dataset' or 'test'")

    def _load_dataset_images(self, dataset: Literal["cifar10", "cifar100", "mnist"] = "cifar10", num_samples: int = 4, seed: Optional[int] = None, normalize: bool = False, target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load sample images from popular datasets for demonstrations."""
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Define transforms
        transform = transforms.Compose([transforms.ToTensor()])

        # Load the appropriate dataset
        dataset_cache = self._cache_dir / "data"
        dataset_cache.mkdir(parents=True, exist_ok=True)

        if dataset.lower() == "cifar10":
            data = torchvision.datasets.CIFAR10(root=str(dataset_cache), train=True, download=True, transform=transform)
        elif dataset.lower() == "cifar100":
            data = torchvision.datasets.CIFAR100(root=str(dataset_cache), train=True, download=True, transform=transform)
        elif dataset.lower() == "mnist":
            data = torchvision.datasets.MNIST(root=str(dataset_cache), train=True, download=True, transform=transform)
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

        # Apply target_size if specified
        if target_size is not None:
            resize_transform = transforms.Resize(target_size)
            images = torch.stack([resize_transform(img) for img in images])

        return images, labels

    def _load_test_images(self, num_samples: int = 4, seed: Optional[int] = None, target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, list]:
        """Load standard test images for demonstrations."""
        # Use smaller default size for better PNG compression
        if target_size is None:
            target_size = (128, 128)

        # Download images if needed
        image_dir = self._download_test_images()

        # Get available image files
        available_images = list(self.TEST_IMAGES.keys())[:num_samples]

        images = []
        names = []

        # Create transform pipeline with resize
        transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])

        for filename in available_images:
            image_path = image_dir / filename
            if image_path.exists():
                try:
                    # Load image and convert to tensor
                    pil_image = Image.open(image_path).convert("RGB")
                    tensor_image = transform(pil_image)
                    images.append(tensor_image)
                    names.append(filename.split(".")[0])  # Remove extension
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")

        if not images:
            raise RuntimeError("No test images could be loaded. Check internet connection.")

        # Stack into batch tensor
        images_tensor = torch.stack(images[:num_samples])

        return images_tensor, names[:num_samples]


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
    trusted_domains = {"raw.githubusercontent.com", "github.com", "cdn.example.com"}  # Add other trusted domains as needed

    if parsed_url.netloc not in trusted_domains:
        raise ValueError(f"URL domain {parsed_url.netloc} is not in the list of trusted domains")

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, url.split("/")[-1])

    # Download the image if it doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)  # nosec B310 - URL is validated for HTTPS and trusted domains
        print("Download complete.")

    return filename


# Standard test images dictionary for reference
TEST_IMAGES = SampleDataLoader.TEST_IMAGES
