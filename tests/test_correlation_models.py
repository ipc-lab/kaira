"""Tests for correlation models."""
import pytest
import torch
import numpy as np

from kaira.data.correlation import WynerZivCorrelationModel, WynerZivCorrelationDataset


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


def test_gaussian_correlation_model(continuous_source):
    """Test Gaussian correlation model."""
    sigma = 0.5
    model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": sigma})
    
    # Generate correlated side information
    side_info = model(continuous_source)
    
    # Check shape is preserved
    assert side_info.shape == continuous_source.shape
    
    # Check that side info is different from source (due to added noise)
    assert not torch.allclose(side_info, continuous_source)
    
    # Calculate the empirical standard deviation of the difference (should be close to sigma)
    noise = side_info - continuous_source
    emp_std = torch.std(noise).item()
    
    # Allow some statistical variation since we're using a finite sample
    assert abs(emp_std - sigma) < 0.1


def test_binary_correlation_model(binary_source):
    """Test binary correlation model."""
    crossover_prob = 0.2
    model = WynerZivCorrelationModel(correlation_type="binary", correlation_params={"crossover_prob": crossover_prob})
    
    # Generate correlated side information
    side_info = model(binary_source)
    
    # Check shape is preserved
    assert side_info.shape == binary_source.shape
    
    # Check that side info contains only binary values
    assert torch.all((side_info == 0) | (side_info == 1))
    
    # Calculate empirical bit-flip rate
    flipped = (side_info != binary_source).float().mean().item()
    
    # Allow some statistical variation
    assert abs(flipped - crossover_prob) < 0.1


def test_custom_correlation_model(continuous_source):
    """Test custom correlation model."""
    # Define a custom transform function
    def custom_transform(x):
        return x * 2 + 1
    
    model = WynerZivCorrelationModel(
        correlation_type="custom", 
        correlation_params={"transform_fn": custom_transform}
    )
    
    # Generate correlated side information
    side_info = model(continuous_source)
    
    # Check shape is preserved
    assert side_info.shape == continuous_source.shape
    
    # Check that the custom transform was applied correctly
    expected = custom_transform(continuous_source)
    assert torch.allclose(side_info, expected)


def test_missing_custom_transform():
    """Test error handling for missing custom transform function."""
    model = WynerZivCorrelationModel(correlation_type="custom", correlation_params={})
    
    # Attempting to use the model without a transform function should raise ValueError
    with pytest.raises(ValueError):
        model(torch.randn(10))


def test_unknown_correlation_type():
    """Test error handling for unknown correlation type."""
    # Invalid correlation type
    model = WynerZivCorrelationModel(correlation_type="invalid")
    
    # Attempting to use the model should raise ValueError
    with pytest.raises(ValueError):
        model(torch.randn(10))


def test_correlation_dataset(binary_source):
    """Test WynerZivCorrelationDataset functionality."""
    # Create dataset with binary correlation
    dataset = WynerZivCorrelationDataset(
        binary_source, 
        correlation_type="binary", 
        correlation_params={"crossover_prob": 0.1}
    )
    
    # Check dataset length
    assert len(dataset) == len(binary_source)
    
    # Test __getitem__
    source, side_info = dataset[0]
    assert source == binary_source[0]
    
    # Test with slice
    sources, side_infos = dataset[0:10]
    assert torch.all(sources == binary_source[0:10])
    
    # Check that data and correlated_data have the same shape
    assert dataset.data.shape == dataset.correlated_data.shape


def test_correlation_statistical_properties():
    """Test statistical properties of the correlation models."""
    # Create a larger binary dataset for better statistical estimates
    torch.manual_seed(123)
    n_samples = 10000
    source = torch.randint(0, 2, (n_samples,), dtype=torch.float32)
    
    # Test with different crossover probabilities
    for p in [0.1, 0.3, 0.5]:
        model = WynerZivCorrelationModel(correlation_type="binary", correlation_params={"crossover_prob": p})
        side_info = model(source)
        
        # Calculate correlation coefficient (phi coefficient for binary data)
        n11 = torch.sum((source == 1) & (side_info == 1)).item()
        n10 = torch.sum((source == 1) & (side_info == 0)).item()
        n01 = torch.sum((source == 0) & (side_info == 1)).item()
        n00 = torch.sum((source == 0) & (side_info == 0)).item()
        
        n1_ = n11 + n10
        n0_ = n01 + n00
        n_1 = n11 + n01
        n_0 = n10 + n00
        
        # Expected correlation for BSC with crossover prob p is (1-2p)
        if n1_ > 0 and n0_ > 0 and n_1 > 0 and n_0 > 0:
            phi = (n11 * n00 - n10 * n01) / np.sqrt(n1_ * n0_ * n_1 * n_0)
            expected_phi = 1 - 2 * p
            assert abs(phi - expected_phi) < 0.05


def test_gaussian_correlation_with_multidimensional_input():
    """Test Gaussian correlation model with multi-dimensional input."""
    torch.manual_seed(42)
    source = torch.randn(10, 3, 32, 32)  # Batch of 10 RGB images of size 32x32
    sigma = 0.5
    
    model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": sigma})
    side_info = model(source)
    
    # Check shape preservation
    assert side_info.shape == source.shape
    
    # Calculate empirical noise standard deviation
    noise = side_info - source
    emp_std = torch.std(noise).item()
    assert abs(emp_std - sigma) < 0.1