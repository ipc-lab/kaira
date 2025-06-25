"""Tests for the datasets module."""

import numpy as np
import pytest

from kaira.data.datasets import (
    BinaryTensorDataset,
    UniformTensorDataset,
    WynerZivCorrelationDataset,
)


class TestBinaryTensorDataset:
    """Test class for BinaryTensorDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        n_samples = 10
        feature_shape = (64,)
        prob = 0.3

        dataset = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape, prob=prob)

        assert len(dataset) == n_samples

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "data" in sample
        assert isinstance(sample["data"], np.ndarray)
        assert sample["data"].shape == feature_shape
        assert sample["data"].dtype == np.float32

        # Check that values are binary (0 or 1)
        data = sample["data"]
        assert np.all((data == 0) | (data == 1))

    def test_multidimensional_feature_shape(self):
        """Test with multidimensional feature shapes."""
        n_samples = 5
        feature_shape = (3, 32, 32)

        dataset = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == feature_shape

    def test_feature_shape_as_int(self):
        """Test with feature_shape as single integer."""
        n_samples = 5
        feature_shape = 128

        dataset = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == (128,)

    def test_feature_shape_as_list(self):
        """Test with feature_shape as list."""
        n_samples = 5
        feature_shape = [10, 20]

        dataset = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == (10, 20)

    def test_probability_values(self):
        """Test different probability values."""
        n_samples = 1000
        feature_shape = (100,)

        # Test prob = 0 (all zeros)
        dataset_zeros = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape, prob=0.0, seed=42)

        sample = dataset_zeros[0]
        assert np.all(sample["data"] == 0)

        # Test prob = 1 (all ones)
        dataset_ones = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape, prob=1.0, seed=42)

        sample = dataset_ones[0]
        assert np.all(sample["data"] == 1)

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""
        n_samples = 10
        feature_shape = (50,)
        seed = 42

        dataset1 = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape, seed=seed)

        dataset2 = BinaryTensorDataset(n_samples=n_samples, feature_shape=feature_shape, seed=seed)

        # Should get the same data with same seed
        for i in range(n_samples):
            np.testing.assert_array_equal(dataset1[i]["data"], dataset2[i]["data"])

    def test_index_out_of_range(self):
        """Test IndexError for out of range access."""
        dataset = BinaryTensorDataset(n_samples=3, feature_shape=(10,))

        with pytest.raises(IndexError, match="Index 3 out of range for dataset of size 3"):
            dataset[3]

        with pytest.raises(IndexError, match="Index 10 out of range for dataset of size 3"):
            dataset[10]

    def test_data_pregeneration(self):
        """Test that data is pre-generated and consistent."""
        dataset = BinaryTensorDataset(n_samples=5, feature_shape=(10,), seed=42)

        # Getting the same sample multiple times should return the same data
        sample1 = dataset[0]
        sample2 = dataset[0]

        np.testing.assert_array_equal(sample1["data"], sample2["data"])


class TestUniformTensorDataset:
    """Test class for UniformTensorDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        n_samples = 10
        feature_shape = (64,)
        low = 0.0
        high = 1.0

        dataset = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape, low=low, high=high)

        assert len(dataset) == n_samples

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "data" in sample
        assert isinstance(sample["data"], np.ndarray)
        assert sample["data"].shape == feature_shape
        assert sample["data"].dtype == np.float32

        # Check that values are within bounds
        data = sample["data"]
        assert np.all(data >= low)
        assert np.all(data <= high)

    def test_multidimensional_feature_shape(self):
        """Test with multidimensional feature shapes."""
        n_samples = 5
        feature_shape = (3, 32, 32)

        dataset = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == feature_shape

    def test_feature_shape_as_int(self):
        """Test with feature_shape as single integer."""
        n_samples = 5
        feature_shape = 128

        dataset = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == (128,)

    def test_feature_shape_as_list(self):
        """Test with feature_shape as list."""
        n_samples = 5
        feature_shape = [10, 20]

        dataset = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape)

        sample = dataset[0]
        assert sample["data"].shape == (10, 20)

    def test_custom_bounds(self):
        """Test different low and high bounds."""
        n_samples = 10
        feature_shape = (100,)
        low = -5.0
        high = 10.0

        dataset = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape, low=low, high=high)

        sample = dataset[0]
        data = sample["data"]
        assert np.all(data >= low)
        assert np.all(data <= high)

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""
        n_samples = 10
        feature_shape = (50,)
        seed = 42

        dataset1 = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape, seed=seed)

        dataset2 = UniformTensorDataset(n_samples=n_samples, feature_shape=feature_shape, seed=seed)

        # Should get the same data with same seed
        for i in range(n_samples):
            np.testing.assert_array_equal(dataset1[i]["data"], dataset2[i]["data"])

    def test_index_out_of_range(self):
        """Test IndexError for out of range access."""
        dataset = UniformTensorDataset(n_samples=3, feature_shape=(10,))

        with pytest.raises(IndexError, match="Index 3 out of range for dataset of size 3"):
            dataset[3]

        with pytest.raises(IndexError, match="Index 10 out of range for dataset of size 3"):
            dataset[10]

    def test_data_pregeneration(self):
        """Test that data is pre-generated and consistent."""
        dataset = UniformTensorDataset(n_samples=5, feature_shape=(10,), seed=42)

        # Getting the same sample multiple times should return the same data
        sample1 = dataset[0]
        sample2 = dataset[0]

        np.testing.assert_array_equal(sample1["data"], sample2["data"])


class TestWynerZivCorrelationDataset:
    """Test class for WynerZivCorrelationDataset."""

    def test_basic_initialization_binary(self):
        """Test basic initialization with binary correlation."""
        n_samples = 10
        feature_shape = (64,)

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary")

        assert len(dataset) == n_samples

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "source" in sample
        assert "side_info" in sample

        source = sample["source"]
        side_info = sample["side_info"]

        assert isinstance(source, np.ndarray)
        assert isinstance(side_info, np.ndarray)
        assert source.shape == feature_shape
        assert side_info.shape == feature_shape
        assert source.dtype == np.float32
        assert side_info.dtype == np.float32

        # Check that values are binary (0 or 1)
        assert np.all((source == 0) | (source == 1))
        assert np.all((side_info == 0) | (side_info == 1))

    def test_binary_correlation_with_crossover_prob(self):
        """Test binary correlation with custom crossover probability."""
        n_samples = 5
        feature_shape = (100,)
        crossover_prob = 0.2

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary", correlation_params={"crossover_prob": crossover_prob}, seed=42)

        sample = dataset[0]
        source = sample["source"]
        side_info = sample["side_info"]

        # Check that values are binary
        assert np.all((source == 0) | (source == 1))
        assert np.all((side_info == 0) | (side_info == 1))

    def test_gaussian_correlation(self):
        """Test Gaussian correlation type."""
        n_samples = 10
        feature_shape = (64,)

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="gaussian")

        sample = dataset[0]
        source = sample["source"]
        side_info = sample["side_info"]

        assert source.shape == feature_shape
        assert side_info.shape == feature_shape

        # For Gaussian, source should be in [0,1] range
        assert np.all(source >= 0.0)
        assert np.all(source <= 1.0)

    def test_gaussian_correlation_with_sigma(self):
        """Test Gaussian correlation with custom sigma."""
        n_samples = 5
        feature_shape = (50,)
        sigma = 0.05

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="gaussian", correlation_params={"sigma": sigma}, seed=42)

        sample = dataset[0]
        source = sample["source"]
        side_info = sample["side_info"]

        assert source.shape == feature_shape
        assert side_info.shape == feature_shape

    def test_custom_correlation_with_transform(self):
        """Test custom correlation type with transform function."""
        n_samples = 5
        feature_shape = (20,)

        def custom_transform(x):
            """Simple transform: add 0.1 and clip."""
            return np.clip(x + 0.1, 0, 1).astype(np.float32)

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="custom", correlation_params={"transform_fn": custom_transform})

        sample = dataset[0]
        source = sample["source"]
        side_info = sample["side_info"]

        assert source.shape == feature_shape
        assert side_info.shape == feature_shape

    def test_custom_correlation_missing_transform(self):
        """Test custom correlation type without transform function raises error."""
        with pytest.raises(ValueError, match="Custom correlation type requires 'transform_fn'"):
            WynerZivCorrelationDataset(n_samples=5, feature_shape=(10,), correlation_type="custom")

    def test_invalid_correlation_type(self):
        """Test that invalid correlation type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown correlation type: invalid"):
            WynerZivCorrelationDataset(n_samples=5, feature_shape=(10,), correlation_type="invalid")

    def test_multidimensional_feature_shape(self):
        """Test with multidimensional feature shapes."""
        n_samples = 3
        feature_shape = (2, 4, 4)

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary")

        sample = dataset[0]
        assert sample["source"].shape == feature_shape
        assert sample["side_info"].shape == feature_shape

    def test_feature_shape_as_int(self):
        """Test with feature_shape as single integer."""
        n_samples = 3
        feature_shape = 128

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="gaussian")

        sample = dataset[0]
        assert sample["source"].shape == (128,)
        assert sample["side_info"].shape == (128,)

    def test_feature_shape_as_list(self):
        """Test with feature_shape as list."""
        n_samples = 3
        feature_shape = [8, 16]

        dataset = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary")

        sample = dataset[0]
        assert sample["source"].shape == (8, 16)
        assert sample["side_info"].shape == (8, 16)

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""
        n_samples = 5
        feature_shape = (20,)
        seed = 42

        dataset1 = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary", seed=seed)

        dataset2 = WynerZivCorrelationDataset(n_samples=n_samples, feature_shape=feature_shape, correlation_type="binary", seed=seed)

        # Should get the same data with same seed
        for i in range(n_samples):
            np.testing.assert_array_equal(dataset1[i]["source"], dataset2[i]["source"])
            np.testing.assert_array_equal(dataset1[i]["side_info"], dataset2[i]["side_info"])

    def test_index_out_of_range(self):
        """Test IndexError for out of range access."""
        dataset = WynerZivCorrelationDataset(n_samples=3, feature_shape=(10,), correlation_type="gaussian")

        with pytest.raises(IndexError, match="Index 3 out of range for dataset of size 3"):
            dataset[3]

        with pytest.raises(IndexError, match="Index 10 out of range for dataset of size 3"):
            dataset[10]

    def test_data_pregeneration(self):
        """Test that data is pre-generated and consistent."""
        dataset = WynerZivCorrelationDataset(n_samples=3, feature_shape=(10,), correlation_type="binary", seed=42)

        # Getting the same sample multiple times should return the same data
        sample1 = dataset[0]
        sample2 = dataset[0]

        np.testing.assert_array_equal(sample1["source"], sample2["source"])
        np.testing.assert_array_equal(sample1["side_info"], sample2["side_info"])

    def test_default_correlation_params(self):
        """Test default correlation parameters are handled correctly."""
        dataset = WynerZivCorrelationDataset(n_samples=2, feature_shape=(10,), correlation_type="binary")

        # Should work with default correlation_params (None -> {})
        sample = dataset[0]
        assert "source" in sample
        assert "side_info" in sample
