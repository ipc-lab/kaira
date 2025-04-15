"""Comprehensive tests for the ChannelCodeModel class."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, IdentityChannel, PerfectChannel
from kaira.constraints import AveragePowerConstraint, BaseConstraint, IdentityConstraint
from kaira.models import ChannelCodeModel
from kaira.models.base import BaseModel
from kaira.models.generic import IdentityModel
from kaira.models.registry import ModelRegistry
from kaira.modulations import (
    BaseDemodulator,
    BaseModulator,
    IdentityDemodulator,
    IdentityModulator,
    PSKDemodulator,
    PSKModulator,
)


class SimpleEncoder(BaseModel):
    """A simple encoder for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)

    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class SimpleDecoder(BaseModel):
    """A simple decoder for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x, *args, **kwargs):
        decoded = self.fc(x)
        # Return both decoded data and a soft estimate
        return decoded


class SimpleModulator(BaseModulator):
    """A simple modulator for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # Simple identity modulation
        return x

    @property
    def bits_per_symbol(self) -> int:
        return 1


class SimpleDemodulator(BaseDemodulator):
    """A simple demodulator for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # Simple identity demodulation
        return x

    @property
    def bits_per_symbol(self) -> int:
        return 1


class SimpleConstraint(BaseConstraint):
    """A simple constraint for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        # Simple identity constraint
        return x


class ParityEncoder(torch.nn.Module):
    """Simple encoder that adds a parity bit."""

    def forward(self, x, *args, **kwargs):
        # Add a parity bit (sum of all elements % 2)
        parity = (torch.sum(x, dim=1) % 2).unsqueeze(1)
        return torch.cat([x, parity], dim=1)


class ParityDecoder(torch.nn.Module):
    """Simple decoder that removes the parity bit."""

    def forward(self, x, *args, **kwargs):
        # Remove the parity bit and return the original data
        original = x[:, :-1]
        return original


@pytest.fixture
def basic_channel_code_model():
    """Create a simple channel code model with identity components."""
    return ChannelCodeModel(
        encoder=IdentityModel(),
        constraint=IdentityConstraint(),
        modulator=IdentityModulator(),
        channel=IdentityChannel(),
        demodulator=IdentityDemodulator(),
        decoder=IdentityModel(),
    )


@pytest.fixture
def simple_channel_code_model():
    """Create a simple channel code model with custom components."""
    encoder = SimpleEncoder()
    modulator = SimpleModulator()
    constraint = SimpleConstraint()
    channel = PerfectChannel()
    demodulator = SimpleDemodulator()
    decoder = SimpleDecoder()

    return ChannelCodeModel(encoder=encoder, modulator=modulator, constraint=constraint, channel=channel, demodulator=demodulator, decoder=decoder)


@pytest.fixture
def realistic_channel_code_model():
    """Create a more realistic channel code model for testing."""
    encoder = SimpleEncoder()
    modulator = PSKModulator(order=4)  # QPSK
    constraint = AveragePowerConstraint(average_power=1.0)
    channel = AWGNChannel(snr_db=10.0)
    demodulator = PSKDemodulator(order=4)
    decoder = SimpleDecoder()

    return ChannelCodeModel(encoder=encoder, modulator=modulator, constraint=constraint, channel=channel, demodulator=demodulator, decoder=decoder)


@pytest.fixture
def parity_channel_code_model():
    """Create a channel code model with parity encoding/decoding."""
    return ChannelCodeModel(
        encoder=ParityEncoder(),
        constraint=IdentityConstraint(),
        modulator=IdentityModulator(),
        channel=IdentityChannel(),
        demodulator=IdentityDemodulator(),
        decoder=ParityDecoder(),
    )


class TestChannelCodeModel:
    """Comprehensive test suite for the ChannelCodeModel class."""

    def test_basic_init(self, basic_channel_code_model):
        """Test the initialization of the ChannelCodeModel with identity components."""
        model = basic_channel_code_model

        # Verify that the model has the correct components
        assert isinstance(model.encoder, IdentityModel)
        assert isinstance(model.decoder, IdentityModel)
        assert isinstance(model.constraint, IdentityConstraint)
        assert isinstance(model.modulator, IdentityModulator)
        assert isinstance(model.channel, IdentityChannel)
        assert isinstance(model.demodulator, IdentityDemodulator)

        # Verify that the steps are correctly set
        assert len(model.steps) == 6
        assert model.steps[0] == model.encoder
        assert model.steps[1] == model.modulator
        assert model.steps[2] == model.constraint
        assert model.steps[3] == model.channel
        assert model.steps[4] == model.demodulator
        assert model.steps[5] == model.decoder

    def test_custom_init(self, simple_channel_code_model):
        """Test model initialization with custom components."""
        model = simple_channel_code_model

        # Check component assignment
        assert isinstance(model.encoder, SimpleEncoder)
        assert isinstance(model.modulator, SimpleModulator)
        assert isinstance(model.constraint, SimpleConstraint)
        assert isinstance(model.channel, PerfectChannel)
        assert isinstance(model.demodulator, SimpleDemodulator)
        assert isinstance(model.decoder, SimpleDecoder)

        # Check steps in the sequential model
        assert len(model.steps) == 6

    def test_basic_forward(self, basic_channel_code_model):
        """Test the forward method with identity components."""
        model = basic_channel_code_model

        # Create a test input tensor
        batch_size = 2
        input_dim = 10
        input_data = torch.rand(batch_size, input_dim)

        # Process the input through the model
        output = model(input_data)

        # Check the output type and value (should be identical with identity components after binary conversion)
        assert isinstance(output, torch.Tensor)
        binary_input_data = (input_data > 0).float()
        assert torch.allclose(output, binary_input_data)

    def test_forward_perfect_channel(self, simple_channel_code_model):
        """Test forward pass with custom components and a perfect channel."""
        model = simple_channel_code_model
        batch_size = 5
        input_data = torch.randn(batch_size, 10)

        output = model(input_data)

        # Check output shape and type
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 10)

        # With a perfect channel, decoded should be deterministically related to input
        # but not identical because of the encoding/decoding transformations
        assert not torch.allclose(output, input_data, atol=1e-5)

    def test_forward_noisy_channel(self, realistic_channel_code_model):
        """Test forward pass with a realistic noisy channel."""
        # Create a fresh model with new components to ensure random state isolation
        encoder = SimpleEncoder()
        modulator = PSKModulator(order=4)  # QPSK
        constraint = AveragePowerConstraint(average_power=1.0)
        channel1 = AWGNChannel(snr_db=10.0)  # Create a separate channel for first run
        demodulator = PSKDemodulator(order=4)
        decoder = SimpleDecoder()

        model1 = ChannelCodeModel(encoder=encoder, modulator=modulator, constraint=constraint, channel=channel1, demodulator=demodulator, decoder=decoder)

        batch_size = 5
        torch.manual_seed(41)  # Set seed for generating input data
        # Generate random continuous values
        continuous_input = torch.randn(batch_size, 10)
        # Convert to binary values (0s and 1s) for the PSKModulator
        input_data = (continuous_input > 0).float()  # Use float() instead of long() for binary values

        # First run using this input
        torch.manual_seed(42)  # Set seed for first run
        model1.train()
        output1 = model1(input_data)

        # Check output shape and type
        assert isinstance(output1, torch.Tensor)
        assert output1.shape == (batch_size, 10)

        # Store result from first run
        first_run = output1.clone()

        # Create a completely separate model for the second run to avoid any state sharing
        channel2 = AWGNChannel(snr_db=10.0)  # Create a separate channel for second run
        model2 = ChannelCodeModel(
            encoder=SimpleEncoder(), modulator=PSKModulator(order=4), constraint=AveragePowerConstraint(average_power=1.0), channel=channel2, demodulator=PSKDemodulator(order=4), decoder=SimpleDecoder()  # New instance  # New instance  # New instance  # New instance  # New instance
        )

        # Second run with a different random seed
        torch.manual_seed(43)  # Set a different seed for the second run
        model2.train()
        output2 = model2(input_data)
        second_run = output2

        # The outputs should be different due to the random noise in the channel
        assert not torch.allclose(first_run, second_run, atol=1e-5)

    def test_parity_encoding(self, parity_channel_code_model):
        """Test a simple parity encoder/decoder scenario."""
        model = parity_channel_code_model

        # Create a test input tensor with binary values (0s and 1s)
        batch_size = 3
        input_dim = 5
        # Use binary inputs to ensure consistent behavior
        input_data = torch.randint(0, 2, (batch_size, input_dim)).float()

        # Process the input through the model
        output = model(input_data)

        # Check the output matches the input after encoding/decoding
        assert isinstance(output, torch.Tensor)
        assert torch.allclose(output, input_data)

    def test_with_keyword_arguments(self, simple_channel_code_model):
        """Test model forward pass with additional keyword arguments."""
        model = simple_channel_code_model
        input_data = torch.randn(3, 10)

        # Pass some additional kwargs
        output = model(input_data, extra_param=42, another_param="test")

        # Ensure the forward pass completes successfully and returns a tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 10)

    def test_device_compatibility(self, simple_channel_code_model):
        """Test model compatibility with different devices."""
        model = simple_channel_code_model
        input_data = torch.randn(3, 10)

        # Move model to CPU explicitly
        model = model.to("cpu")
        input_data = input_data.to("cpu")

        # Forward pass should work on CPU
        output_cpu = model(input_data)
        assert output_cpu.device.type == "cpu"

        # Skip GPU test if not available
        if torch.cuda.is_available():
            # Move model to GPU
            model = model.to("cuda")
            input_data = input_data.to("cuda")

            # Forward pass should work on GPU
            output_gpu = model(input_data)
            assert output_gpu.device.type == "cuda"

    def test_model_registry(self):
        """Test that channel code model is correctly registered in ModelRegistry."""
        # Check registration
        assert "channel_code" in ModelRegistry._models

        # Create model through registry
        encoder = IdentityModel()
        constraint = IdentityConstraint()
        modulator = IdentityModulator()
        channel = IdentityChannel()
        demodulator = IdentityDemodulator()
        decoder = IdentityModel()

        model = ModelRegistry.create(
            "channel_code",
            encoder=encoder,
            constraint=constraint,
            modulator=modulator,
            channel=channel,
            demodulator=demodulator,
            decoder=decoder,
        )

        assert isinstance(model, ChannelCodeModel)
