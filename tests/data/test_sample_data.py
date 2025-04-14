\
import pytest
import torch
import os
from kaira.data.sample_data import load_sample_images

# Define expected shapes for different datasets
EXPECTED_SHAPES = {
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'mnist': (1, 28, 28),
}

@pytest.mark.parametrize("dataset_name", ['cifar10', 'cifar100', 'mnist'])
def test_load_sample_images_basic(dataset_name):
    """Test basic loading for each supported dataset."""
    num_samples = 10
    images, labels = load_sample_images(dataset=dataset_name, num_samples=num_samples)

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (num_samples, *EXPECTED_SHAPES[dataset_name])
    assert labels.shape == (num_samples,)
    # Check if images are generally in [0, 1] range after ToTensor()
    assert images.min() >= 0.0
    assert images.max() <= 1.0

def test_load_sample_images_num_samples():
    """Test loading a different number of samples."""
    num_samples = 5
    images, labels = load_sample_images(dataset='cifar10', num_samples=num_samples)
    assert images.shape[0] == num_samples
    assert labels.shape[0] == num_samples

def test_load_sample_images_seed():
    """Test reproducibility with a fixed seed."""
    seed = 42
    num_samples = 3
    images1, labels1 = load_sample_images(dataset='mnist', num_samples=num_samples, seed=seed)
    images2, labels2 = load_sample_images(dataset='mnist', num_samples=num_samples, seed=seed)

    assert torch.equal(images1, images2)
    assert torch.equal(labels1, labels2)

    # Test that different seeds produce different results (highly likely)
    images3, labels3 = load_sample_images(dataset='mnist', num_samples=num_samples, seed=seed + 1)
    assert not torch.equal(images1, images3)
    assert not torch.equal(labels1, labels3)


def test_load_sample_images_normalize_flag():
    """Test the normalize flag (even though it currently doesn't change behavior)."""
    # This test ensures the code path for normalize=True is executed.
    # Currently, both True and False use transforms.ToTensor() which scales to [0, 1]
    num_samples = 2
    images_norm, labels_norm = load_sample_images(dataset='cifar10', num_samples=num_samples, normalize=True)
    images_no_norm, labels_no_norm = load_sample_images(dataset='cifar10', num_samples=num_samples, normalize=False)

    assert images_norm.shape == (num_samples, *EXPECTED_SHAPES['cifar10'])
    assert labels_norm.shape == (num_samples,)
    assert images_norm.min() >= 0.0
    assert images_norm.max() <= 1.0

    # Check that the results are likely different due to random sampling unless seeded
    # (or identical if the underlying dataset loading caches)
    # We mainly care that the normalize=True path runs without error.
    assert images_no_norm.shape == images_norm.shape


def test_load_sample_images_invalid_dataset():
    """Test that an invalid dataset name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported dataset: invalid_dataset"):
        load_sample_images(dataset='invalid_dataset')

def test_cache_directory_creation():
    """Test that the cache directory is created."""
    # Determine expected cache path relative to the test file execution
    # Assuming tests run from the root directory
    root_path = os.path.abspath(os.path.join('.', '.cache', 'data'))

    # Ensure the directory doesn't exist before the call (might be flaky if tests run in parallel)
    # For simplicity, we'll just check it exists *after* the call.
    if os.path.exists(root_path) and os.path.isdir(root_path):
         # Clean up potential existing directory contents if needed, be careful!
         # For this test, we just rely on load_sample_images creating it.
         pass

    load_sample_images(dataset='mnist', num_samples=1) # Use MNIST as it's small

    assert os.path.exists(root_path)
    assert os.path.isdir(root_path)

