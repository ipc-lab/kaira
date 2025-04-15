"""Tests for correlation models."""
import pytest
import torch

from kaira.data.correlation import WynerZivCorrelationDataset

# ======== Fixtures ========


@pytest.fixture
def binary_source():
    """Fixture providing binary source data for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (100,), dtype=torch.float32)


@pytest.fixture
def continuous_source():
    """Fixture providing continuous source data for testing."""
    torch.manual_seed(42)
    return torch.randn(100, 10)


@pytest.fixture
def large_binary_source():
    """Fixture providing a larger binary dataset for better statistical estimates."""
    torch.manual_seed(123)
    return torch.randint(0, 2, (10000,), dtype=torch.float32)


@pytest.fixture
def multidimensional_source():
    """Fixture providing multi-dimensional input (like images)."""
    torch.manual_seed(42)
    return torch.randn(10, 3, 32, 32)  # Batch of 10 RGB images of size 32x32


# ======== WynerZivCorrelationDataset Tests ========


class TestWynerZivCorrelationDataset:
    """Tests for the WynerZivCorrelationDataset class."""

    def test_dataset_basics(self, binary_source):
        """Test basic functionality of WynerZivCorrelationDataset."""
        # Create dataset with binary correlation
        dataset = WynerZivCorrelationDataset(binary_source, correlation_type="binary", correlation_params={"crossover_prob": 0.1})

        # Check dataset length
        assert len(dataset) == len(binary_source)

        # Check that data and correlated_data have the same shape
        assert dataset.data.shape == dataset.correlated_data.shape

    def test_dataset_getitem(self, binary_source):
        """Test __getitem__ functionality of WynerZivCorrelationDataset."""
        # Create dataset with binary correlation
        dataset = WynerZivCorrelationDataset(binary_source, correlation_type="binary", correlation_params={"crossover_prob": 0.1})

        # Test single element access
        source, side_info = dataset[0]
        assert source == binary_source[0]
        assert torch.is_tensor(side_info)
        assert isinstance(side_info.item(), float)

        # Test slicing
        sources, side_infos = dataset[0:10]
        assert torch.all(sources == binary_source[0:10])
        assert sources.shape == torch.Size([10])
        assert side_infos.shape == torch.Size([10])
