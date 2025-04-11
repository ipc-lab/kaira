"""Tests for the Kurmukova2025TransCoder model."""
import pytest
import torch

from kaira.channels import AWGNChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models.binary.kurmukova2025_transcoder import Kurmukova2025TransCoder
from kaira.models.registry import ModelRegistry
from kaira.modulations import PSKDemodulator, PSKModulator


class MockEncoder(torch.nn.Module):
    """Mock encoder for testing."""

    def __init__(self, output_dim=16, binary_output=False):
        super().__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(10, output_dim)
        self.binary_output = binary_output

    def forward(self, x, *args, **kwargs):
        """Mock forward pass."""
        if isinstance(x, list) and len(x) > 1:
            x = x[0]  # Handle multi-input case

        # Handle complex tensor case
        if torch.is_complex(x):
            x = x.real

        # Ensure correct dimensions if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Ensure input has correct dimensions for the linear layer
        if x.shape[-1] != 10 and x.shape[-1] > 10:
            x = x[..., :10]  # Take first 10 dimensions
        elif x.shape[-1] != 10:
            # Pad with zeros if needed
            padding = torch.zeros(*x.shape[:-1], 10 - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)

        output = self.linear(x)

        # If binary output is needed (for PSK modulation), round to 0 or 1
        if self.binary_output:
            output = torch.sigmoid(output)
            output = torch.round(output)

        return output


class MockDecoder(torch.nn.Module):
    """Mock decoder for testing."""

    def __init__(self, input_dim=16, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, *args, **kwargs):
        """Mock forward pass that returns decoded output and soft estimates."""
        if isinstance(x, tuple):
            x = x[0]  # Handle tuple input case

        if isinstance(x, list) and len(x) > 1:
            x = x[0]  # Handle multi-input case

        # Handle complex tensor case
        if torch.is_complex(x):
            x = x.real

        # Ensure correct dimensions if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Ensure input has correct dimensions for the linear layer
        if x.shape[-1] != self.input_dim and x.shape[-1] > self.input_dim:
            x = x[..., : self.input_dim]  # Take first input_dim dimensions
        elif x.shape[-1] != self.input_dim:
            # Pad with zeros if needed
            padding = torch.zeros(*x.shape[:-1], self.input_dim - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)

        decoded = self.linear(x)
        soft_estimate = torch.sigmoid(decoded)  # Just a dummy soft estimate
        return decoded, soft_estimate


def test_transcoder_model_initialization():
    """Test that the TransCoder model initializes correctly."""
    # Create components
    encoder_tc = MockEncoder(output_dim=16)
    encoder_ec = MockEncoder(output_dim=10)
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model
    model = Kurmukova2025TransCoder(encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=2)

    # Check components are correctly assigned
    assert model.encoder_tc is encoder_tc
    assert model.encoder_ec is encoder_ec
    assert model.constraint is constraint
    assert model.modulator is modulator
    assert model.channel is channel
    assert model.demodulator is demodulator
    assert model.decoder_tc is decoder_tc
    assert model.decoder_ec is decoder_ec
    assert model.n_iterations == 2


def test_transcoder_model_registry():
    """Test that the TransCoder model is properly registered."""
    assert "transcoder" in ModelRegistry._models

    # Check model can be created from registry
    encoder_tc = MockEncoder(output_dim=16)
    encoder_ec = MockEncoder(output_dim=10)
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    model = ModelRegistry.create("transcoder", encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=1)

    assert isinstance(model, Kurmukova2025TransCoder)


def test_transcoder_forward_with_tc():
    """Test forward pass of TransCoder model with TransCoder neural encoder/decoder."""
    # Create components
    encoder_tc = MockEncoder(output_dim=16)
    encoder_ec = MockEncoder(output_dim=10)
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model
    model = Kurmukova2025TransCoder(encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=2)

    # Test data
    x = torch.randn(4, 10)  # Batch of 4, 10-dimensional input

    # Forward pass
    output = model(x)

    # Check output structure
    assert "final_output" in output
    assert "iterations" in output
    assert "history" in output

    # Check final output shape
    assert output["final_output"].shape == x.shape

    # Check number of iterations
    assert len(output["iterations"]) == 2

    # Check iteration output structure
    for iteration in output["iterations"]:
        assert "demodulated" in iteration
        assert "decoded" in iteration
        assert "soft_estimate" in iteration

    # Check history structure
    assert len(output["history"]) == 1
    assert "encoded" in output["history"][0]
    assert "constrained" in output["history"][0]
    assert "received" in output["history"][0]


def test_transcoder_forward_without_tc():
    """Test forward pass of TransCoder model without TransCoder neural encoder/decoder."""
    # Create components
    encoder_ec = MockEncoder(output_dim=10, binary_output=True)  # Set binary_output=True for PSK modulation
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model without TransCoder neural encoder/decoder
    model = Kurmukova2025TransCoder(encoder_tc=None, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=None, decoder_ec=decoder_ec, n_iterations=1)

    # Test data
    x = torch.randn(4, 10)  # Batch of 4, 10-dimensional input

    # Forward pass
    output = model(x)

    # Check output structure
    assert "final_output" in output
    assert "iterations" in output
    assert "history" in output

    # Check final output shape
    assert output["final_output"].shape == x.shape

    # Check number of iterations
    assert len(output["iterations"]) == 1

    # Check iteration output structure
    for iteration in output["iterations"]:
        assert "demodulated" in iteration
        assert "decoded" in iteration
        assert "soft_estimate" in iteration


def test_transcoder_multiple_iterations():
    """Test that multiple iterations work correctly in TransCoder model."""
    # Create components
    encoder_tc = MockEncoder(output_dim=16)
    encoder_ec = MockEncoder(output_dim=10)
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)  # Providing SNR in dB
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model
    model = Kurmukova2025TransCoder(encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=3)

    # Test data
    x = torch.randn(4, 10)  # Batch of 4, 10-dimensional input

    # Forward pass
    output = model(x)

    # Check number of iterations
    assert len(output["iterations"]) == 3

    # Check each iteration has the required outputs
    for i, iteration in enumerate(output["iterations"]):
        assert "demodulated" in iteration
        assert "decoded" in iteration
        assert "soft_estimate" in iteration


@pytest.mark.parametrize("use_tc_encoder, use_tc_decoder", [(True, True), (True, False), (False, True), (False, False)])
def test_transcoder_configurations(use_tc_encoder, use_tc_decoder):
    """Test different configurations of the TransCoder model."""
    # Create components
    encoder_tc = MockEncoder(output_dim=16) if use_tc_encoder else None
    # When not using TransCoder encoder, we need binary output for PSK modulation
    encoder_ec = MockEncoder(output_dim=10, binary_output=(not use_tc_encoder))
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)  # Providing SNR in dB
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10) if use_tc_decoder else None
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model
    model = Kurmukova2025TransCoder(encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=1)

    # Test data
    x = torch.randn(4, 10)  # Batch of 4, 10-dimensional input

    # Forward pass should work with any configuration
    output = model(x)

    # Check output structure
    assert "final_output" in output
    assert "iterations" in output
    assert "history" in output

    # Check final output shape
    assert output["final_output"].shape == x.shape


def test_transcoder_custom_args():
    """Test that custom arguments are properly passed through the TransCoder model."""
    # Create components
    encoder_tc = MockEncoder(output_dim=16)
    encoder_ec = MockEncoder(output_dim=10)
    constraint = TotalPowerConstraint(total_power=1.0)
    modulator = PSKModulator(order=4)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder_tc = MockDecoder(input_dim=16, output_dim=10)
    decoder_ec = MockDecoder(input_dim=10, output_dim=10)

    # Initialize model
    model = Kurmukova2025TransCoder(encoder_tc=encoder_tc, encoder_ec=encoder_ec, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder_tc=decoder_tc, decoder_ec=decoder_ec, n_iterations=1)

    # Test data
    x = torch.randn(4, 10)  # Batch of 4, 10-dimensional input

    # Custom args to pass through
    custom_arg1 = {"special_mode": True}
    custom_arg2 = 0.75

    # Forward pass with custom args
    output = model(x, custom_arg1, custom_arg2=custom_arg2)

    # Check output is still correct
    assert "final_output" in output
    assert "iterations" in output
    assert "history" in output
    assert output["final_output"].shape == x.shape
