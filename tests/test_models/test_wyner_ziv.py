# tests/test_models/test_wyner_ziv.py
import pytest
import torch
import torch.nn as nn

from kaira.channels import IdentityChannel
from kaira.constraints import TotalPowerConstraint
from kaira.data.correlation import WynerZivCorrelationModel
from kaira.models import WynerZivModel


class SimpleEncoder(nn.Module):
    """Simple encoder for testing WynerZivModel."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing WynerZivModel."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(5, 10)
        
    def forward(self, x, side_info):
        # Combine received signal with side information
        combined = x + 0.1 * side_info
        return self.layer(combined)


class SimpleQuantizer(nn.Module):
    """Simple quantizer for testing WynerZivModel."""
    
    def forward(self, x):
        return torch.round(x)


class SimpleSyndromeGenerator(nn.Module):
    """Simple syndrome generator for testing WynerZivModel."""
    
    def forward(self, x):
        return x * 0.5


@pytest.fixture
def wyner_ziv_components():
    """Fixture providing components for Wyner-Ziv model testing."""
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    channel = IdentityChannel()
    correlation_model = WynerZivCorrelationModel(correlation=0.8)
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


def test_wyner_ziv_model_initialization(wyner_ziv_components):
    """Test WynerZivModel initialization with all components."""
    model = WynerZivModel(**wyner_ziv_components)
    
    # Check all components are properly assigned
    assert model.encoder == wyner_ziv_components["encoder"]
    assert model.decoder == wyner_ziv_components["decoder"]
    assert model.channel == wyner_ziv_components["channel"]
    assert model.correlation_model == wyner_ziv_components["correlation_model"]
    assert model.quantizer == wyner_ziv_components["quantizer"]
    assert model.syndrome_generator == wyner_ziv_components["syndrome_generator"]
    assert model.constraint == wyner_ziv_components["constraint"]


def test_wyner_ziv_model_minimal_initialization(wyner_ziv_components):
    """Test WynerZivModel initialization with only required components."""
    minimal_components = {
        "encoder": wyner_ziv_components["encoder"],
        "decoder": wyner_ziv_components["decoder"],
        "channel": wyner_ziv_components["channel"],
    }
    
    model = WynerZivModel(**minimal_components)
    
    # Check required components are properly assigned
    assert model.encoder == minimal_components["encoder"]
    assert model.decoder == minimal_components["decoder"]
    assert model.channel == minimal_components["channel"]
    
    # Check optional components are None
    assert model.correlation_model is None
    assert model.quantizer is None
    assert model.syndrome_generator is None
    assert model.constraint is None


def test_wyner_ziv_model_forward_with_side_info(wyner_ziv_components):
    """Test WynerZivModel forward pass with provided side information."""
    model = WynerZivModel(**wyner_ziv_components)
    
    # Create test input
    source = torch.randn(5, 10)  # 5 samples, 10 dimensions
    side_info = torch.randn(5, 5)  # 5 samples, 5 dimensions (matches encoder output)
    
    # Run model with provided side information
    result = model(source, side_info)
    
    # Check all intermediate results are present
    assert "encoded" in result
    assert "quantized" in result
    assert "syndromes" in result
    assert "constrained" in result
    assert "received" in result
    assert "side_info" in result
    assert "decoded" in result
    
    # Check output shapes
    assert result["encoded"].shape == (5, 5)
    assert result["decoded"].shape == (5, 10)
    assert result["side_info"].shape == side_info.shape
    
    # Check that side_info in output is the same as provided
    assert torch.all(result["side_info"] == side_info)


def test_wyner_ziv_model_forward_without_side_info(wyner_ziv_components):
    """Test WynerZivModel forward pass with generated side information."""
    model = WynerZivModel(**wyner_ziv_components)
    
    # Create test input
    source = torch.randn(5, 10)  # 5 samples, 10 dimensions
    
    # Run model without side information (should use correlation model)
    result = model(source)
    
    # Check all intermediate results are present
    assert "encoded" in result
    assert "quantized" in result
    assert "syndromes" in result
    assert "constrained" in result
    assert "received" in result
    assert "side_info" in result
    assert "decoded" in result
    
    # Check that side_info was generated
    assert result["side_info"] is not None
    
    # Check output shapes
    assert result["encoded"].shape == (5, 5)
    assert result["decoded"].shape == (5, 10)


def test_wyner_ziv_model_without_correlation_model():
    """Test WynerZivModel behavior without correlation model."""
    # Create minimal model without correlation model
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    channel = IdentityChannel()
    
    model = WynerZivModel(encoder=encoder, decoder=decoder, channel=channel)
    
    # Create test input
    source = torch.randn(5, 10)
    side_info = torch.randn(5, 5)
    
    # Model should work with provided side_info
    result = model(source, side_info)
    assert result["decoded"].shape == (5, 10)
    
    # Model should raise error when no side_info is provided and no correlation model exists
    with pytest.raises(ValueError):
        model(source)