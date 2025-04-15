"""Tests for the Multiple Access Channel (MAC) model."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models import MultipleAccessChannelModel
from kaira.models.registry import ModelRegistry


class SimpleEncoder(nn.Module):
    """Simple encoder for testing MultipleAccessChannelModel."""

    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, *args, **kwargs):  # Accept *args and **kwargs
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing MultipleAccessChannelModel."""

    def __init__(self, input_dim=5, output_dim=10):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, *args, **kwargs):  # Accept *args and **kwargs
        return self.layer(x)


@pytest.fixture
def mac_components():
    """Fixture providing components for MultipleAccessChannelModel testing."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    num_devices = 3

    # Pass classes for encoder and decoder
    return {"channel": channel, "power_constraint": power_constraint, "encoders": SimpleEncoder, "decoder": SimpleDecoder, "num_devices": num_devices}  # Pass class  # Pass class


def test_mac_model_initialization(mac_components):
    """Test MultipleAccessChannelModel initialization with default parameters (non-shared)."""
    model = MultipleAccessChannelModel(**mac_components)

    # Check basic components
    assert model.channel == mac_components["channel"]
    assert model.power_constraint == mac_components["power_constraint"]
    assert model.num_devices == mac_components["num_devices"]
    assert model.num_users == mac_components["num_devices"]

    # Check encoder/decoder lists and shared flags
    assert len(model.encoders) == model.num_devices
    assert len(model.decoders) == model.num_devices
    assert model.shared_encoder is False
    assert model.shared_decoder is False

    # Check all encoders and decoders are different instances
    for i in range(model.num_devices - 1):
        assert model.encoders[i] is not model.encoders[i + 1]
        assert model.decoders[i] is not model.decoders[i + 1]


def test_mac_model_shared_components(mac_components):
    """Test MultipleAccessChannelModel with shared encoder and decoder."""
    components = mac_components.copy()
    components["shared_encoder"] = True
    components["shared_decoder"] = True

    model = MultipleAccessChannelModel(**components)

    # Check shared flags
    assert model.shared_encoder is True
    assert model.shared_decoder is True

    # Check encoder/decoder lists length
    assert len(model.encoders) == model.num_devices
    assert len(model.decoders) == model.num_devices

    # Check all encoders and decoders are the same instance
    for i in range(model.num_devices - 1):
        assert model.encoders[i] is model.encoders[i + 1]
        assert model.decoders[i] is model.decoders[i + 1]


def test_mac_model_forward_pass(mac_components):
    """Test the forward pass of MultipleAccessChannelModel (joint decoding)."""
    # Use non-shared components for this test
    model = MultipleAccessChannelModel(**mac_components, shared_encoder=False, shared_decoder=False)

    # Create test inputs
    inputs = [torch.randn(2, 10) for _ in range(model.num_devices)]

    # Run forward pass
    output = model(inputs)

    # Check output shape (single tensor for joint decoding)
    # Output dim should be num_devices * output_dim_per_decoder
    # SimpleDecoder output_dim is 10
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, model.num_devices * 10)


def test_mac_model_invalid_initialization():
    """Test error handling for invalid initialization parameters."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)

    # Test with num_devices mismatch (encoders list)
    with pytest.raises(ValueError, match="does not match the number of encoders"):
        MultipleAccessChannelModel(encoders=[SimpleEncoder(), SimpleEncoder()], decoder=SimpleDecoder, channel=channel, power_constraint=power_constraint, num_devices=3)

    # Test with num_devices mismatch (decoders list)
    with pytest.raises(ValueError, match="does not match the number of decoders"):
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoder=[SimpleDecoder(), SimpleDecoder()], channel=channel, power_constraint=power_constraint, num_devices=3)

    # Test shared_encoder=True with list of encoders
    with pytest.raises(ValueError, match="shared_encoder cannot be True"):
        MultipleAccessChannelModel(encoders=[SimpleEncoder(), SimpleEncoder()], decoder=SimpleDecoder, channel=channel, power_constraint=power_constraint, shared_encoder=True, num_devices=2)

    # Test shared_decoder=True with list of decoders
    with pytest.raises(ValueError, match="shared_decoder cannot be True"):
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoder=[SimpleDecoder(), SimpleDecoder()], channel=channel, power_constraint=power_constraint, shared_decoder=True, num_devices=2)

    # Test missing num_devices when using classes
    with pytest.raises(ValueError, match="num_devices must be specified"):
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoder=SimpleDecoder, channel=channel, power_constraint=power_constraint)

    # Test invalid encoder type
    # Match the actual error message (lowercase 'encoder')
    with pytest.raises(TypeError, match="Invalid type for encoder configuration"):
        MultipleAccessChannelModel(encoders=123, decoder=SimpleDecoder, channel=channel, power_constraint=power_constraint, num_devices=2)

    # Test invalid decoder type
    # Match the actual error message (lowercase 'decoder')
    with pytest.raises(TypeError, match="Invalid type for decoder configuration"):
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoder="abc", channel=channel, power_constraint=power_constraint, num_devices=2)

    # Test single instance with shared=False and num_devices > 1
    with pytest.raises(ValueError, match="A single Encoder instance was provided"):
        MultipleAccessChannelModel(encoders=SimpleEncoder(), decoder=SimpleDecoder, channel=channel, power_constraint=power_constraint, num_devices=2, shared_encoder=False)
    with pytest.raises(ValueError, match="A single Decoder instance was provided"):
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoder=SimpleDecoder(), channel=channel, power_constraint=power_constraint, num_devices=2, shared_decoder=False)


def test_mac_model_registry(mac_components):
    """Test that MAC model is correctly registered in ModelRegistry."""
    # Check registration
    assert "multiple_access_channel" in ModelRegistry._models

    # Create model through registry
    # Need to provide necessary args like encoders, decoder, num_devices
    model = ModelRegistry.create("multiple_access_channel", encoders=mac_components["encoders"], decoder=mac_components["decoder"], channel=mac_components["channel"], power_constraint=mac_components["power_constraint"], num_devices=mac_components["num_devices"])

    assert isinstance(model, MultipleAccessChannelModel)
    assert model.num_devices == mac_components["num_devices"]


def test_mac_model_with_csi_and_noise():
    """Test MultipleAccessChannelModel forward pass with explicit CSI and noise."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    num_devices = 2
    model = MultipleAccessChannelModel(channel=channel, power_constraint=power_constraint, encoders=SimpleEncoder, decoder=SimpleDecoder, num_devices=num_devices)

    # Create inputs
    inputs = [torch.randn(2, 10) for _ in range(num_devices)]

    # Create CSI and noise (assuming channel and decoder can handle them via **kwargs)
    csi = torch.randn(2, 5)  # Example shape
    noise = torch.randn(2, 5)  # Example shape

    # Run forward pass with CSI and noise
    output = model(inputs, csi=csi, noise=noise)

    # Basic output validation (single tensor)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, num_devices * 10)


def test_mac_model_invalid_forward_call():
    """Test error handling for invalid forward calls."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    model = MultipleAccessChannelModel(channel=channel, power_constraint=power_constraint, encoders=SimpleEncoder, decoder=SimpleDecoder, num_devices=2)

    # Test with incorrect positional argument (not a list)
    with pytest.raises(ValueError, match="Input 'x' must be a list"):
        model(torch.randn(2, 10))

    # Test with wrong number of inputs in list
    with pytest.raises(ValueError, match="Number of input tensors .* must match"):
        model([torch.randn(2, 10)])  # Only 1 input for num_devices=2

    # Test with non-tensor in list
    with pytest.raises(ValueError, match="Input 'x' must be a list of torch.Tensors"):
        model([torch.randn(2, 10), "not a tensor"])


def test_mac_model_device_compatibility(mac_components):
    """Test MultipleAccessChannelModel compatibility with different devices."""
    model = MultipleAccessChannelModel(**mac_components)

    # Create test inputs
    inputs = [torch.randn(2, 10) for _ in range(model.num_devices)]

    # Move model to CPU explicitly
    model = model.to("cpu")
    cpu_inputs = [x.to("cpu") for x in inputs]

    # Forward pass should work on CPU
    output_cpu = model(cpu_inputs)
    assert output_cpu.device.type == "cpu"

    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        gpu_inputs = [x.to("cuda") for x in inputs]
        output_gpu = model(gpu_inputs)
        assert output_gpu.device.type == "cuda"
