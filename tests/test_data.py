# tests/test_data.py
import pytest
import torch

from kaira.data import (
    BinaryTensorDataset,
    UniformTensorDataset,
    WynerZivCorrelationDataset,
    WynerZivCorrelationModel,
    create_binary_tensor,
    create_uniform_tensor,
)


@pytest.mark.parametrize("size", [(10, 20), [5, 15, 3]])
def test_create_binary_tensor(size):
    """Test binary tensor creation with different shapes."""
    tensor = create_binary_tensor(size)
    
    # Check shape
    assert tensor.shape == torch.Size(size)
    
    # Check binary values (0 or 1)
    assert torch.all((tensor == 0) | (tensor == 1))


@pytest.mark.parametrize("prob", [0.3, 0.7])
def test_create_binary_tensor_probability(prob):
    """Test binary tensor creation with different probabilities."""
    size = (1000, 1000)  # Large tensor to check probability
    tensor = create_binary_tensor(size, prob=prob)
    
    # Check probability of 1s (should be close to the specified probability)
    mean = tensor.float().mean().item()
    assert abs(mean - prob) < 0.01  # Allow small statistical deviation


@pytest.mark.parametrize("low,high", [(0.0, 1.0), (-2.0, 3.0)])
def test_create_uniform_tensor(low, high):
    """Test uniform tensor creation with different bounds."""
    size = (10, 20)
    tensor = create_uniform_tensor(size, low=low, high=high)
    
    # Check shape
    assert tensor.shape == torch.Size(size)
    
    # Check bounds
    assert torch.all(tensor >= low)
    assert torch.all(tensor < high)
    
    # Check distribution (approximately uniform)
    if size[0] * size[1] > 1000:  # Only check for larger tensors
        hist = torch.histc(tensor, bins=10, min=low, max=high)
        # All bins should be roughly equal in a uniform distribution
        expected_count = tensor.numel() / 10
        normalized_hist = hist / expected_count
        assert torch.all((normalized_hist > 0.8) & (normalized_hist < 1.2))


def test_binary_tensor_dataset():
    """Test BinaryTensorDataset functionality."""
    size = (100, 5, 10)
    dataset = BinaryTensorDataset(size)
    
    # Check length
    assert len(dataset) == size[0]
    
    # Check item retrieval
    item = dataset[0]
    assert item.shape == torch.Size(size[1:])
    assert torch.all((item == 0) | (item == 1))
    
    # Test slicing
    batch = dataset[10:20]
    assert batch.shape == torch.Size([10, *size[1:]])


def test_uniform_tensor_dataset():
    """Test UniformTensorDataset functionality."""
    size = (100, 5, 10)
    low, high = -1.0, 2.0
    dataset = UniformTensorDataset(size, low=low, high=high)
    
    # Check length
    assert len(dataset) == size[0]
    
    # Check item retrieval
    item = dataset[0]
    assert item.shape == torch.Size(size[1:])
    assert torch.all((item >= low) & (item < high))
    
    # Test slicing
    batch = dataset[10:20]
    assert batch.shape == torch.Size([10, *size[1:]])


def test_wyner_ziv_correlation_model():
    """Test WynerZivCorrelationModel functionality."""
    # Use the actual parameters expected by the WynerZivCorrelationModel
    model = WynerZivCorrelationModel(
        correlation_type="binary", 
        correlation_params={"crossover_prob": 0.2}
    )
    
    # Generate correlated sequences
    x = torch.randint(0, 2, (1000,)).float()
    y = model(x)
    
    # Check shapes
    assert y.shape == x.shape
    
    # Check binary values
    assert torch.all((y == 0) | (y == 1))
    
    # Check correlation (approximately)
    # Expected correlation is 1 - 2*crossover_prob for binary case
    expected_corr = 1 - 2 * 0.2  # = 0.6
    empirical_corr = 1 - 2 * ((x != y).float().mean().item())
    assert abs(empirical_corr - expected_corr) < 0.05  # Allow some statistical deviation


def test_wyner_ziv_correlation_dataset():
    """Test WynerZivCorrelationDataset functionality."""
    # Create source tensor
    source = torch.randint(0, 2, (100, 20)).float()
    
    # Create dataset with the proper parameters
    dataset = WynerZivCorrelationDataset(
        source=source,
        correlation_type="binary",
        correlation_params={"crossover_prob": 0.15}
    )
    
    # Check length
    assert len(dataset) == 100  # First dimension of source
    
    # Check item retrieval
    x, y = dataset[0]
    assert x.shape == torch.Size([20])  # Second dimension of source
    assert y.shape == torch.Size([20])
    
    # Check binary values
    assert torch.all((x == 0) | (x == 1))
    assert torch.all((y == 0) | (y == 1))
    
    # Test batch retrieval
    batch_x, batch_y = dataset[10:20]
    assert batch_x.shape == torch.Size([10, 20])
    assert batch_y.shape == torch.Size([10, 20])