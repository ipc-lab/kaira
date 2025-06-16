import pytest
import torch

from kaira.data.sample_data import SampleDataLoader

# Define expected shapes for different datasets
EXPECTED_SHAPES = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
}


@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100", "mnist"])
def test_sample_data_loader_basic(dataset_name):
    """Test basic loading for each supported dataset."""
    num_samples = 10
    loader = SampleDataLoader()
    images, labels = loader.load_images(source="dataset", dataset=dataset_name, num_samples=num_samples)

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == (num_samples, *EXPECTED_SHAPES[dataset_name])
    assert labels.shape == (num_samples,)
    # Check if images are generally in [0, 1] range after ToTensor()
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_sample_data_loader_num_samples():
    """Test loading a different number of samples."""
    num_samples = 5
    loader = SampleDataLoader()
    images, labels = loader.load_images(source="dataset", dataset="cifar10", num_samples=num_samples)
    assert images.shape[0] == num_samples
    assert labels.shape[0] == num_samples


def test_sample_data_loader_seed():
    """Test reproducibility with a fixed seed."""
    seed = 42
    num_samples = 3
    loader = SampleDataLoader()
    images1, labels1 = loader.load_images(source="dataset", dataset="mnist", num_samples=num_samples, seed=seed)
    images2, labels2 = loader.load_images(source="dataset", dataset="mnist", num_samples=num_samples, seed=seed)

    assert torch.equal(images1, images2)
    assert torch.equal(labels1, labels2)

    # Test that different seeds produce different results (highly likely)
    images3, labels3 = loader.load_images(source="dataset", dataset="mnist", num_samples=num_samples, seed=seed + 1)
    assert not torch.equal(images1, images3)
    assert not torch.equal(labels1, labels3)


def test_sample_data_loader_normalize_flag():
    """Test the normalize flag (even though it currently doesn't change behavior)."""
    # This test ensures the code path for normalize=True is executed.
    # Currently, both True and False use transforms.ToTensor() which scales to [0, 1]
    num_samples = 2
    loader = SampleDataLoader()
    images_norm, labels_norm = loader.load_images(source="dataset", dataset="cifar10", num_samples=num_samples, normalize=True)
    images_no_norm, labels_no_norm = loader.load_images(source="dataset", dataset="cifar10", num_samples=num_samples, normalize=False)

    assert images_norm.shape == (num_samples, *EXPECTED_SHAPES["cifar10"])
    assert labels_norm.shape == (num_samples,)
    assert images_norm.min() >= 0.0
    assert images_norm.max() <= 1.0

    # Check that the results are likely different due to random sampling unless seeded
    # (or identical if the underlying dataset loading caches)
    # We mainly care that the normalize=True path runs without error.
    assert images_no_norm.shape == images_norm.shape


def test_sample_data_loader_invalid_dataset():
    """Test that an invalid dataset name raises ValueError."""
    loader = SampleDataLoader()
    with pytest.raises(ValueError, match="Unsupported dataset: invalid_dataset"):
        loader.load_images(source="dataset", dataset="invalid_dataset")


def test_sample_data_loader_test_images():
    """Test loading test images."""
    loader = SampleDataLoader()
    images, names = loader.load_images(source="test", num_samples=2, target_size=(128, 128))

    assert isinstance(images, torch.Tensor)
    assert isinstance(names, list)
    assert images.shape == (2, 3, 128, 128)
    assert len(names) == 2
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_sample_data_loader_unified_interface():
    """Test the unified load_images interface."""
    loader = SampleDataLoader()

    # Test dataset loading
    dataset_images, labels = loader.load_images("dataset", dataset="cifar10", num_samples=2)
    assert isinstance(dataset_images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert dataset_images.shape == (2, 3, 32, 32)

    # Test test image loading
    test_images, names = loader.load_images("test", num_samples=2, target_size=(128, 128))
    assert isinstance(test_images, torch.Tensor)
    assert isinstance(names, list)
    assert test_images.shape == (2, 3, 128, 128)


def test_cache_directory_creation():
    """Test that the cache directory is created."""
    loader = SampleDataLoader()

    # Test dataset loading which should create cache directory
    loader.load_images(source="dataset", dataset="mnist", num_samples=1)

    # Check that cache directory exists
    cache_dir = loader._get_cache_directory()
    assert cache_dir.exists()
    assert (cache_dir / "data").exists()
