"""Tests for the Wyner-Ziv model with complex scenarios."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, IdentityChannel
from kaira.constraints import TotalPowerConstraint
from kaira.data.correlation import WynerZivCorrelationModel
from kaira.models import WynerZivModel


class SimpleEncoder(nn.Module):
    """Simple encoder for testing the WynerZiv model."""
    
    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing the WynerZiv model."""
    
    def __init__(self, input_dim=5, side_info_dim=5, output_dim=10):
        super().__init__()
        self.side_info_proj = nn.Linear(side_info_dim, input_dim)
        self.combined_layer = nn.Linear(input_dim * 2, output_dim)
    
    def forward(self, x, side_info):
        # Project side information
        side_info_proj = self.side_info_proj(side_info)
        # Combine with received signal
        combined = torch.cat([x, side_info_proj], dim=1)
        return self.combined_layer(combined)


class SimpleQuantizer(nn.Module):
    """Simple quantizer that rounds to the nearest integer."""
    
    def forward(self, x):
        return torch.round(x)


class SimpleSyndromeGenerator(nn.Module):
    """Simple syndrome generator for testing."""
    
    def forward(self, x):
        return x * 0.5  # Just reduce amplitude as a simple transformation


@pytest.fixture
def wyner_ziv_components():
    """Fixture providing components for testing the Wyner-Ziv model."""
    input_dim = 10
    latent_dim = 5
    
    encoder = SimpleEncoder(input_dim=input_dim, output_dim=latent_dim)
    decoder = SimpleDecoder(input_dim=latent_dim, side_info_dim=latent_dim, output_dim=input_dim)
    channel = AWGNChannel(snr_db=20)
    correlation_model = WynerZivCorrelationModel(
        correlation_type="gaussian", 
        correlation_params={"sigma": 0.5}
    )
    quantizer = SimpleQuantizer()
    syndrome_generator = SimpleSyndromeGenerator()
    constraint = TotalPowerConstraint(total_power=1.0)
    
    return {
        "encoder": encoder,
        "decoder": decoder,
        "channel": channel,
        "correlation_model": correlation_model,
        "quantizer": quantizer,
        "syndrome_generator": syndrome_generator,
        "constraint": constraint
    }


@pytest.fixture
def multi_dim_source():
    """Fixture providing multi-dimensional source data."""
    torch.manual_seed(42)
    batch_size = 8
    channels = 3
    height = 16
    width = 16
    
    # Create a mock image batch
    return torch.randn(batch_size, channels, height, width)


def test_wyner_ziv_with_multi_dim_source(wyner_ziv_components, multi_dim_source):
    """Test Wyner-Ziv model with multi-dimensional source data."""
    # Adjust components for multi-dimensional input
    input_shape = multi_dim_source.shape[1:]  # [channels, height, width]
    input_dim = torch.prod(torch.tensor(input_shape)).item()
    latent_dim = input_dim // 4  # Using 4:1 compression ratio
    
    # Create encoder for multi-dimensional input
    class MultidimEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(input_dim, latent_dim)
        
        def forward(self, x):
            batch_size = x.shape[0]
            flattened = self.flatten(x)
            return self.linear(flattened)
    
    # Create decoder for multi-dimensional input
    class MultidimDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(latent_dim * 2, input_dim)
            self.unflatten = nn.Unflatten(1, input_shape)
        
        def forward(self, x, side_info):
            combined = torch.cat([x, side_info], dim=1)
            flattened_output = self.linear(combined)
            return self.unflatten(flattened_output)
    
    # Replace encoder and decoder
    wyner_ziv_components["encoder"] = MultidimEncoder()
    wyner_ziv_components["decoder"] = MultidimDecoder()
    
    # Create the model
    model = WynerZivModel(**wyner_ziv_components)
    
    # Run the model
    result = model(multi_dim_source)
    
    # Check output shape matches input shape
    assert result["decoded"].shape == multi_dim_source.shape
    
    # Check all intermediate results have correct shapes
    assert result["encoded"].shape[0] == multi_dim_source.shape[0]
    assert result["encoded"].shape[1] == latent_dim
    assert result["side_info"].shape == result["encoded"].shape


def test_wyner_ziv_training_compatibility(wyner_ziv_components):
    """Test that the Wyner-Ziv model can be trained with backpropagation."""
    # Create a small source
    batch_size = 16
    input_dim = 10
    source = torch.randn(batch_size, input_dim)
    
    # Create model
    model = WynerZivModel(**wyner_ziv_components)
    
    # Set up a simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Initial loss
    result = model(source)
    initial_loss = torch.mean((result["decoded"] - source) ** 2)
    
    # Run a few optimization steps
    for _ in range(5):
        optimizer.zero_grad()
        result = model(source)
        loss = torch.mean((result["decoded"] - source) ** 2)
        loss.backward()
        optimizer.step()
    
    # Final loss
    result = model(source)
    final_loss = torch.mean((result["decoded"] - source) ** 2)
    
    # Loss should decrease
    assert final_loss < initial_loss


def test_wyner_ziv_different_correlation_models(wyner_ziv_components):
    """Test Wyner-Ziv model with different correlation models."""
    # Create a fixed source
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 10
    source = torch.randn(batch_size, input_dim)
    
    # Test with different correlation models
    correlation_types = [
        ("gaussian", {"sigma": 0.1}),
        ("gaussian", {"sigma": 1.0}),
        ("binary", {"crossover_prob": 0.1}),
    ]
    
    results = []
    
    for corr_type, corr_params in correlation_types:
        # Update correlation model
        wyner_ziv_components["correlation_model"] = WynerZivCorrelationModel(
            correlation_type=corr_type,
            correlation_params=corr_params
        )
        
        # Create model
        model = WynerZivModel(**wyner_ziv_components)
        
        # Run model
        result = model(source)
        results.append(result)
        
        # Check basic outputs
        assert result["decoded"].shape == source.shape
        assert result["side_info"] is not None
    
    # The results should be different for different correlation models
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            # Compare side information
            assert not torch.allclose(results[i]["side_info"], results[j]["side_info"])
            # Compare decoded outputs
            assert not torch.allclose(results[i]["decoded"], results[j]["decoded"])


def test_wyner_ziv_without_optional_components():
    """Test Wyner-Ziv model behavior without optional components."""
    # Create minimal components
    input_dim = 10
    latent_dim = 5
    
    encoder = SimpleEncoder(input_dim=input_dim, output_dim=latent_dim)
    decoder = SimpleDecoder(input_dim=latent_dim, side_info_dim=latent_dim, output_dim=input_dim)
    channel = IdentityChannel()  # Use identity channel for deterministic testing
    
    # Create model with only required components
    model = WynerZivModel(encoder=encoder, decoder=decoder, channel=channel)
    
    # Run with side info (should work)
    source = torch.randn(16, input_dim)
    side_info = torch.randn(16, latent_dim)
    result = model(source, side_info)
    
    # Check outputs
    assert result["decoded"].shape == source.shape
    assert torch.allclose(result["side_info"], side_info)
    assert "encoded" in result
    assert "received" in result
    
    # Check that these keys are None since we didn't provide the components
    assert "quantized" not in result
    assert "syndromes" not in result
    assert "constrained" not in result
    
    # Without side info and correlation model, should raise ValueError
    with pytest.raises(ValueError):
        model(source)