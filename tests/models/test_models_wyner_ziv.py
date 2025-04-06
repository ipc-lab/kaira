# tests/test_models/test_wyner_ziv.py
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
    
    def forward(self, x, *args, **kwargs):
        # Forward pass ignoring extra args/kwargs
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing the WynerZiv model."""
    
    def __init__(self, input_dim=5, side_info_dim=5, output_dim=10):
        super().__init__()
        self.side_info_proj = nn.Linear(side_info_dim, input_dim)
        self.combined_layer = nn.Linear(input_dim * 2, output_dim)
    
    def forward(self, x, side_info, *args, **kwargs):
        # Handle different dimensional side_info
        if side_info.dim() != 2:
            # Flatten multi-dimensional side_info to match expected dimensionality
            side_info = side_info.view(side_info.size(0), -1)
        
        # Extract only the needed columns if side_info has too many features
        if side_info.size(1) > self.side_info_proj.in_features:
            side_info = side_info[:, :self.side_info_proj.in_features]
            
        # Pad with zeros if side_info has too few features
        elif side_info.size(1) < self.side_info_proj.in_features:
            padding = torch.zeros(
                side_info.size(0), 
                self.side_info_proj.in_features - side_info.size(1),
                device=side_info.device
            )
            side_info = torch.cat([side_info, padding], dim=1)
        
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
        
        def forward(self, x, side_info, *args, **kwargs):
            # Handle different dimensional side_info
            if side_info.dim() != 2:
                # Flatten multi-dimensional side_info to match x's dimensionality
                side_info = side_info.view(side_info.size(0), -1)
                
                # If reshaped side_info is larger than expected, slice it
                if side_info.size(1) > x.size(1):
                    side_info = side_info[:, :x.size(1)]
                # If reshaped side_info is smaller than expected, pad with zeros
                elif side_info.size(1) < x.size(1):
                    padding = torch.zeros(side_info.size(0), 
                                          x.size(1) - side_info.size(1),
                                          device=side_info.device)
                    side_info = torch.cat([side_info, padding], dim=1)
            
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
    # Only check batch size, not exact shape since correlation model may preserve dimensionality
    assert result["side_info"].shape[0] == result["encoded"].shape[0]


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
    
    # Even without optional components, the model still adds these keys
    # with default values (encoded value is passed through the pipeline)
    # Just check that they exist and have consistent values
    assert "quantized" in result
    assert torch.allclose(result["quantized"], result["encoded"])
    assert "syndromes" in result
    assert torch.allclose(result["syndromes"], result["encoded"])
    assert "constrained" in result
    assert torch.allclose(result["constrained"], result["encoded"])
    
    # Without side info and correlation model, should raise ValueError
    with pytest.raises(ValueError):
        model(source)


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


def test_wyner_ziv_model_with_kwargs(wyner_ziv_components):
    """Test WynerZivModel forward pass with additional keyword arguments."""
    model = WynerZivModel(**wyner_ziv_components)

    # Create test input
    source = torch.randn(5, 10)

    # Run model with additional kwargs
    result = model(source, additional_param="test", snr_db=15.0)

    # Basic checks to ensure the model runs with kwargs
    assert "decoded" in result
    assert result["decoded"].shape == source.shape


def test_wyner_ziv_model_without_quantizer(wyner_ziv_components):
    """Test WynerZivModel behavior when quantizer is None."""
    # Remove quantizer
    components = {k: v for k, v in wyner_ziv_components.items() if k != "quantizer"}
    
    model = WynerZivModel(**components)
    
    # Create test input
    source = torch.randn(5, 10)
    
    # Run model
    result = model(source)
    
    # Check that quantized equals encoded when no quantizer is present
    assert torch.all(result["quantized"] == result["encoded"])


def test_wyner_ziv_model_without_syndrome_generator(wyner_ziv_components):
    """Test WynerZivModel behavior when syndrome_generator is None."""
    # Remove syndrome_generator
    components = {k: v for k, v in wyner_ziv_components.items() if k != "syndrome_generator"}
    
    model = WynerZivModel(**components)
    
    # Create test input
    source = torch.randn(5, 10)
    
    # Run model
    
    result = model(source)
    
    # Check that syndromes equals quantized when no syndrome_generator is present
    assert torch.all(result["syndromes"] == result["quantized"])


def test_wyner_ziv_model_without_constraint(wyner_ziv_components):
    """Test WynerZivModel behavior when constraint is None."""
    # Remove constraint
    components = {k: v for k, v in wyner_ziv_components.items() if k != "constraint"}
    
    model = WynerZivModel(**components)
    
    # Create test input
    source = torch.randn(5, 10)
    
    # Run model
    result = model(source)
    
    # Check that constrained equals syndromes when no constraint is present
    assert torch.all(result["constrained"] == result["syndromes"])


def test_wyner_ziv_model_device_compatibility(wyner_ziv_components):
    """Test WynerZivModel compatibility with different devices."""
    model = WynerZivModel(**wyner_ziv_components)
    
    # Create test input
    source = torch.randn(5, 10)
    
    # Move model to CPU explicitly
    model = model.to("cpu")
    source = source.to("cpu")
    
    # Forward pass should work on CPU
    output_cpu = model(source)
    assert output_cpu["decoded"].device.type == "cpu"
    
    # Skip GPU test if not available
    if torch.cuda.is_available():
        # Move model to GPU
        model = model.to("cuda")
        source = source.to("cuda")
        
        # Forward pass should work on GPU
        output_gpu = model(source)
        assert output_gpu["decoded"].device.type == "cuda"


