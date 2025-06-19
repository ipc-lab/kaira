import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kaira.data.sample_data import SampleImagesDataset, TorchVisionDataset, download_image

# Define expected shapes for different datasets
EXPECTED_SHAPES = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
}

EXPECTED_NORMALIZED_SHAPES = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
}


class TestTorchVisionDataset:
    """Test class for TorchVisionDataset."""

    @pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100", "mnist"])
    def test_basic_loading(self, dataset_name):
        """Test basic loading for each supported dataset."""
        n_samples = 5
        dataset = TorchVisionDataset(dataset_name=dataset_name, n_samples=n_samples, train=False)

        assert len(dataset) == n_samples

        # Test first sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert "dataset" in sample
        assert "shape" in sample

        # Check image properties
        image = sample["image"]
        assert isinstance(image, np.ndarray)
        assert image.shape == EXPECTED_SHAPES[dataset_name]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

        # Check label
        assert isinstance(sample["label"], int)
        assert sample["dataset"] == dataset_name

    def test_invalid_dataset_name(self):
        """Test that invalid dataset names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset: invalid_dataset"):
            TorchVisionDataset(dataset_name="invalid_dataset")

    def test_custom_cache_dir(self):
        """Test using a custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = TorchVisionDataset(dataset_name="mnist", n_samples=2, cache_dir=tmpdir, train=False)
            assert len(dataset) == 2
            assert Path(tmpdir).exists()

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""
        seed = 42
        dataset1 = TorchVisionDataset(dataset_name="mnist", n_samples=3, seed=seed, train=False)
        dataset2 = TorchVisionDataset(dataset_name="mnist", n_samples=3, seed=seed, train=False)

        # Should get the same images with same seed
        for i in range(len(dataset1)):
            sample1 = dataset1[i]
            sample2 = dataset2[i]
            np.testing.assert_array_equal(sample1["image"], sample2["image"])
            assert sample1["label"] == sample2["label"]

    def test_target_size_resizing(self):
        """Test that target_size parameter works correctly."""
        target_size = (64, 64)
        dataset = TorchVisionDataset(dataset_name="cifar10", n_samples=2, target_size=target_size, train=False)

        sample = dataset[0]
        image = sample["image"]
        assert image.shape == (3, 64, 64)  # Should be resized

    def test_normalization(self):
        """Test normalization functionality."""
        # Test CIFAR-10 normalization
        dataset_cifar = TorchVisionDataset(dataset_name="cifar10", n_samples=2, normalize=True, train=False)
        sample = dataset_cifar[0]
        image = sample["image"]
        # After normalization, values can be negative due to mean subtraction
        assert image.shape == EXPECTED_NORMALIZED_SHAPES["cifar10"]

        # Test MNIST normalization
        dataset_mnist = TorchVisionDataset(dataset_name="mnist", n_samples=2, normalize=True, train=False)
        sample = dataset_mnist[0]
        image = sample["image"]
        assert image.shape == EXPECTED_NORMALIZED_SHAPES["mnist"]

    def test_train_test_split(self):
        """Test using train vs test split."""
        dataset_train = TorchVisionDataset(dataset_name="mnist", n_samples=2, train=True)
        dataset_test = TorchVisionDataset(dataset_name="mnist", n_samples=2, train=False)

        assert len(dataset_train) == 2
        assert len(dataset_test) == 2

    def test_index_out_of_range(self):
        """Test IndexError for out of range access."""
        dataset = TorchVisionDataset(dataset_name="mnist", n_samples=2, train=False)

        with pytest.raises(IndexError, match="Index 2 out of range for dataset of size 2"):
            dataset[2]

        with pytest.raises(IndexError, match="Index 5 out of range for dataset of size 2"):
            dataset[5]


class TestSampleImagesDataset:
    """Test class for SampleImagesDataset."""

    def test_basic_loading(self):
        """Test basic loading of sample test images."""
        n_samples = 2
        dataset = SampleImagesDataset(n_samples=n_samples, target_size=(128, 128))

        assert len(dataset) == n_samples

        # Test first sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample
        assert "filename" in sample
        assert "shape" in sample

        # Check image properties
        image = sample["image"]
        assert isinstance(image, np.ndarray)
        assert image.shape == (3, 128, 128)  # 3 channels, 128x128
        assert image.min() >= 0.0
        assert image.max() <= 1.0

        # Check filename
        assert isinstance(sample["filename"], str)
        assert sample["filename"] in ["coins", "astronaut", "coffee", "camera"]

    def test_max_samples_limit(self):
        """Test that n_samples is limited to available images."""
        # There are 4 test images, requesting more should limit to 4
        dataset = SampleImagesDataset(n_samples=10)
        assert len(dataset) == 4  # Should be limited to max available

    def test_custom_cache_dir(self):
        """Test using a custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SampleImagesDataset(n_samples=1, cache_dir=tmpdir)
            assert len(dataset) == 1
            assert Path(tmpdir).exists()

    def test_seed_functionality(self):
        """Test seed parameter (affects numpy seed)."""
        seed = 42
        dataset = SampleImagesDataset(n_samples=2, seed=seed)
        assert len(dataset) == 2
        # The seed is set in the constructor but doesn't affect image order
        # since we use a fixed order from TEST_IMAGES

    def test_default_target_size(self):
        """Test default target size behavior."""
        dataset = SampleImagesDataset(n_samples=1, target_size=(256, 256))
        sample = dataset[0]
        assert sample["image"].shape == (3, 256, 256)

    def test_index_out_of_range(self):
        """Test IndexError for out of range access."""
        dataset = SampleImagesDataset(n_samples=2)

        with pytest.raises(IndexError, match="Index 2 out of range for dataset of size 2"):
            dataset[2]

    @patch("kaira.data.sample_data.urllib.request.urlretrieve")
    @patch("kaira.data.sample_data.Path.exists")
    def test_download_failure_handling(self, mock_exists, mock_urlretrieve):
        """Test handling of download failures."""
        # Mock that files don't exist initially
        mock_exists.return_value = False

        # Mock download failures
        import urllib.error

        mock_urlretrieve.side_effect = urllib.error.HTTPError(url="test", code=404, msg="Not Found", hdrs=None, fp=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should handle download failures gracefully
            # The _load_images method will raise RuntimeError if no images are loaded
            with pytest.raises(RuntimeError, match="No test images could be loaded"):
                SampleImagesDataset(n_samples=1, cache_dir=tmpdir)

    @patch("kaira.data.sample_data.Image.open")
    @patch("kaira.data.sample_data.Path.exists")
    def test_image_loading_failure(self, mock_exists, mock_image_open):
        """Test handling of image loading failures."""
        # Mock that files exist
        mock_exists.return_value = True

        # Mock image loading failure
        mock_image_open.side_effect = Exception("Image loading failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should handle image loading failures gracefully
            with pytest.raises(RuntimeError, match="No test images could be loaded"):
                SampleImagesDataset(n_samples=1, cache_dir=tmpdir)


class TestDownloadImage:
    """Test class for download_image function."""

    def test_non_https_url(self):
        """Test that non-HTTPS URLs are rejected."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Only HTTPS URLs are allowed for security reasons"):
                download_image("http://example.com/image.png", tmpdir)

    def test_untrusted_domain(self):
        """Test that untrusted domains are rejected."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="URL domain evil.com is not in the list of trusted domains"):
                download_image("https://evil.com/image.png", tmpdir)

    def test_trusted_domain_github(self):
        """Test that GitHub URLs are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kaira.data.sample_data.urllib.request.urlretrieve") as mock_retrieve:
                with patch("kaira.data.sample_data.os.path.exists", return_value=False):
                    url = "https://raw.githubusercontent.com/test/test/image.png"
                    result = download_image(url, tmpdir)

                    # Should call urlretrieve and return the filename
                    mock_retrieve.assert_called_once()
                    assert result.endswith("image.png")

    def test_file_already_exists(self):
        """Test behavior when file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            test_file = os.path.join(tmpdir, "image.png")
            with open(test_file, "w") as f:
                f.write("dummy")

            url = "https://raw.githubusercontent.com/test/test/image.png"
            result = download_image(url, tmpdir)

            # Should return existing file path without downloading
            assert result == test_file
            assert os.path.exists(result)
