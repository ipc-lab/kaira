# tests/test_models/test_wyner_ziv.py
"""Tests for the Wyner-Ziv model with complex scenarios."""
import numpy as np
import pytest
import torch
import torch.nn as nn

# Assuming necessary imports from kaira and helper classes like SimpleEncoder etc. exist
from kaira.channels import IdentityChannel
from kaira.models.base import BaseModel
from kaira.models.wyner_ziv import WynerZivCorrelationModel, WynerZivModel


# Dummy components for testing
class SimpleEncoder(BaseModel):
    def forward(self, x, *args, **kwargs):
        return x[:, :5]  # Reduce dim


class SimpleDecoder(BaseModel):
    def forward(self, x, side_info, *args, **kwargs):
        # Simple decoder might use side_info shape; ensure compatibility or adjust
        # Example: Use only first 5 dims of side_info if needed
        side_info_used = side_info[:, :5] if side_info.shape[1] > 5 else side_info
        # Combine processed input 'x' and 'side_info_used'
        # For testing, just return something with the original source shape (e.g., expand x)
        return torch.cat((x, side_info_used), dim=1)  # Example: returns shape (B, 10)


class SimpleQuantizer(nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.round(x * 10) / 10


class SimpleSyndromeGen(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x  # Passthrough


# Fixture providing standard components
@pytest.fixture
def wyner_ziv_components():
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    channel = IdentityChannel()
    correlation_model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": 0.5})
    quantizer = SimpleQuantizer()
    syndrome_generator = SimpleSyndromeGen()
    constraint = None  # No constraint for basic tests

    return {"encoder": encoder, "decoder": decoder, "channel": channel, "correlation_model": correlation_model, "quantizer": quantizer, "syndrome_generator": syndrome_generator, "constraint": constraint}


# ======== WynerZivCorrelationModel Tests ========


class TestWynerZivCorrelationModel:
    """Tests for the WynerZivCorrelationModel class."""

    # ----- Gaussian correlation tests -----

    def test_gaussian_correlation_basic(self, continuous_source):
        """Test basic functionality of Gaussian correlation model."""
        sigma = 0.5
        model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": sigma})

        # Generate correlated side information
        side_info = model(continuous_source)

        # Check shape is preserved
        assert side_info.shape == continuous_source.shape

        # Check that side info is different from source (due to added noise)
        assert not torch.allclose(side_info, continuous_source)

        # Calculate the empirical standard deviation of the difference
        noise = side_info - continuous_source
        emp_std = torch.std(noise).item()

        # Allow some statistical variation since we're using a finite sample
        assert abs(emp_std - sigma) < 0.1

    def test_gaussian_correlation_statistics(self, continuous_source):
        """Test statistical properties of Gaussian correlation."""
        sigma = 0.5
        model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": sigma})
        correlated = model(continuous_source)

        # Mean should be approximately preserved
        assert torch.allclose(correlated.mean(), continuous_source.mean(), atol=0.1)

        # Theoretical variance: Var(X+N) = Var(X) + Var(N) where N ~ N(0, sigma²)
        expected_std = (continuous_source.std().item() ** 2 + sigma**2) ** 0.5
        assert torch.allclose(correlated.std(), torch.tensor(expected_std), atol=0.1)

    def test_gaussian_with_multidimensional_input(self, multidimensional_source):
        """Test Gaussian correlation model with multi-dimensional input."""
        sigma = 0.5

        model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": sigma})
        side_info = model(multidimensional_source)

        # Check shape preservation
        assert side_info.shape == multidimensional_source.shape

        # Calculate empirical noise standard deviation
        noise = side_info - multidimensional_source
        emp_std = torch.std(noise).item()
        assert abs(emp_std - sigma) < 0.1

    # ----- Binary correlation tests -----

    def test_binary_correlation_basic(self, binary_source):
        """Test basic functionality of binary correlation model."""
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

    def test_binary_correlation_statistics(self, large_binary_source):
        """Test statistical properties of binary correlation models."""
        # Test with different crossover probabilities
        for p in [0.1, 0.3, 0.5]:
            model = WynerZivCorrelationModel(correlation_type="binary", correlation_params={"crossover_prob": p})
            side_info = model(large_binary_source)

            # Calculate correlation coefficient (phi coefficient for binary data)
            n11 = torch.sum((large_binary_source == 1) & (side_info == 1)).item()
            n10 = torch.sum((large_binary_source == 1) & (side_info == 0)).item()
            n01 = torch.sum((large_binary_source == 0) & (side_info == 1)).item()
            n00 = torch.sum((large_binary_source == 0) & (side_info == 0)).item()

            n1_ = n11 + n10
            n0_ = n01 + n00
            n_1 = n11 + n01
            n_0 = n10 + n00

            # Expected correlation for BSC with crossover prob p is (1-2p)
            if n1_ > 0 and n0_ > 0 and n_1 > 0 and n_0 > 0:
                phi = (n11 * n00 - n10 * n01) / np.sqrt(n1_ * n0_ * n_1 * n_0)
                expected_phi = 1 - 2 * p
                assert abs(phi - expected_phi) < 0.05

            # Alternative calculation of correlation
            empirical_corr = 1 - 2 * ((large_binary_source != side_info).float().mean().item())
            assert abs(empirical_corr - (1 - 2 * p)) < 0.05

    # ----- Custom correlation tests -----

    def test_custom_correlation(self, continuous_source):
        """Test custom correlation model."""

        # Define a custom transform function
        def custom_transform(x):
            return x * 2 + 1

        model = WynerZivCorrelationModel(correlation_type="custom", correlation_params={"transform_fn": custom_transform})

        # Generate correlated side information
        side_info = model(continuous_source)

        # Check shape is preserved
        assert side_info.shape == continuous_source.shape

        # Check that the custom transform was applied correctly
        expected = custom_transform(continuous_source)
        assert torch.allclose(side_info, expected)

    # ----- Error handling tests -----

    def test_missing_custom_transform(self):
        """Test error handling for missing custom transform function."""
        model = WynerZivCorrelationModel(correlation_type="custom", correlation_params={})

        # Attempting to use the model without a transform function should raise ValueError
        with pytest.raises(ValueError, match="requires 'transform_fn' parameter"):
            model(torch.randn(10))

    def test_unknown_correlation_type(self):
        """Test error handling for unknown correlation type."""
        # Invalid correlation type
        model = WynerZivCorrelationModel(correlation_type="invalid")

        # Attempting to use the model should raise ValueError
        with pytest.raises(ValueError, match="Unknown correlation type"):
            model(torch.randn(10))


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
    source = torch.randn(5, 10)
    # Manually create side_info (e.g., correlated or just random matching expected decoder input)
    # SimpleDecoder expects side_info dim 5 after processing
    side_info = torch.randn(5, 5)  # Provide side_info matching decoder needs

    # Run model with provided side information
    result = model(source, side_info)

    # Check output shape matches source shape
    assert result.shape == source.shape


# Renamed from test_wyner_ziv_model_forward_with_generated_side_info
def test_wyner_ziv_model_forward_without_side_info(wyner_ziv_components):
    """Test WynerZivModel forward pass generating side information via correlation model."""
    model = WynerZivModel(**wyner_ziv_components)
    assert model.correlation_model is not None

    # Create test input
    source = torch.randn(5, 10)

    # Run model WITHOUT providing side information (should use correlation model)
    result = model(source)  # Call without side_info

    # Check output shape matches source shape
    assert result.shape == source.shape


def test_wyner_ziv_model_without_correlation_model():
    """Test WynerZivModel behavior without correlation model."""
    # Create minimal model without correlation model
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    channel = IdentityChannel()

    model = WynerZivModel(encoder=encoder, decoder=decoder, channel=channel)  # No correlation_model

    # Create test input
    source = torch.randn(5, 10)
    side_info = torch.randn(5, 5)  # Manually create side_info

    # Model should work with provided side_info
    result = model(source, side_info)
    assert result.shape == source.shape

    # Model should raise ValueError when no side_info is provided AND no correlation model exists
    with pytest.raises(ValueError, match="Side information must be provided"):
        model(source)  # Call without side_info


def test_wyner_ziv_model_without_quantizer(wyner_ziv_components):
    """Test WynerZivModel behavior when quantizer is None."""
    components = {k: v for k, v in wyner_ziv_components.items() if k != "quantizer"}
    model = WynerZivModel(**components)
    assert model.correlation_model is not None

    source = torch.randn(5, 10)
    # Run model without side_info (should use correlation model)
    result = model(source)
    assert result.shape == source.shape


def test_wyner_ziv_model_without_syndrome_generator(wyner_ziv_components):
    """Test WynerZivModel behavior when syndrome_generator is None."""
    components = {k: v for k, v in wyner_ziv_components.items() if k != "syndrome_generator"}
    model = WynerZivModel(**components)
    assert model.correlation_model is not None

    source = torch.randn(5, 10)
    # Run model without side_info (should use correlation model)
    result = model(source)
    assert result.shape == source.shape


def test_wyner_ziv_model_without_constraint(wyner_ziv_components):
    """Test WynerZivModel behavior when constraint is None."""
    # Fixture already provides constraint=None, but let's be explicit
    components = {k: v for k, v in wyner_ziv_components.items() if k != "constraint"}
    model = WynerZivModel(**components)
    assert model.correlation_model is not None

    source = torch.randn(5, 10)
    # Run model without side_info (should use correlation model)
    result = model(source)
    assert result.shape == source.shape


def test_wyner_ziv_model_device_compatibility(wyner_ziv_components):
    """Test WynerZivModel compatibility with different devices."""
    model = WynerZivModel(**wyner_ziv_components).to("cpu")  # Ensure model starts on CPU
    assert model.correlation_model is not None

    source = torch.randn(5, 10).to("cpu")

    # Forward pass should work on CPU (without providing side_info)
    output_cpu = model(source)
    assert output_cpu.device.type == "cpu"
    assert output_cpu.shape == source.shape

    # Skip GPU test if not available
    if torch.cuda.is_available():
        model = model.to("cuda")
        source = source.to("cuda")

        # Forward pass should work on GPU (without providing side_info)
        output_gpu = model(source)
        assert output_gpu.device.type == "cuda"
        assert output_gpu.shape == source.shape


# Add any other relevant tests if needed
