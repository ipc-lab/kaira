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
    # Use 'decoders' (plural) key
    return {"channel": channel, "power_constraint": power_constraint, "encoders": SimpleEncoder, "decoders": SimpleDecoder, "num_devices": num_devices}  # Pass class  # Pass class


def test_mac_model_initialization(mac_components):
    """Test MultipleAccessChannelModel initialization with classes (separate encoders, joint
    decoder)."""
    # Use 'decoders' (plural) key when unpacking
    model = MultipleAccessChannelModel(**mac_components)

    # Check basic components
    assert model.channel == mac_components["channel"]
    assert model.power_constraint == mac_components["power_constraint"]
    assert model.num_devices == mac_components["num_devices"]
    assert model.num_users == mac_components["num_devices"]

    # Check encoder/decoder lists based on class initialization
    # Encoders: Separate instances created from class
    assert len(model.encoders) == model.num_devices
    # Decoders: Single joint instance created from class
    assert len(model.decoders) == 1
    assert isinstance(model.decoders[0], SimpleDecoder)

    # Check encoders are different instances
    if model.num_devices > 1:
        assert model.encoders[0] is not model.encoders[1]


def test_mac_model_shared_components(mac_components):
    """Test MultipleAccessChannelModel with shared encoder and decoder instances."""
    components = mac_components.copy()
    # Create single instances to be shared
    shared_encoder_instance = SimpleEncoder()
    shared_decoder_instance = SimpleDecoder()
    components["encoders"] = shared_encoder_instance  # Pass instance for shared encoder
    components["decoders"] = shared_decoder_instance  # Pass instance for shared decoder

    # Remove shared_* flags, pass instances instead
    model = MultipleAccessChannelModel(**components)

    # Check encoder/decoder lists length and content for shared instances
    # Encoders: List contains num_devices references to the shared instance
    assert len(model.encoders) == model.num_devices
    # Decoders: List contains 1 reference to the shared instance
    assert len(model.decoders) == 1

    # Check all encoders and the decoder are the correct shared instances
    assert model.decoders[0] is shared_decoder_instance
    for i in range(model.num_devices):
        assert model.encoders[i] is shared_encoder_instance


def test_mac_model_forward_pass(mac_components):
    """Test the forward pass of MultipleAccessChannelModel (joint decoding)."""
    # Initialize with classes (default: separate encoders, joint decoder)
    # Use 'decoders' (plural) key, remove shared_* flags
    model = MultipleAccessChannelModel(**mac_components)

    # Create test inputs
    inputs = [torch.randn(2, 10) for _ in range(model.num_devices)]

    # Run forward pass
    output = model(inputs)

    # Check output shape (single tensor for joint decoding)
    # Output dim should be output_dim_of_decoder
    # SimpleDecoder output_dim is 10
    assert isinstance(output, torch.Tensor)
    # Correct shape for joint decoding
    assert output.shape == (2, 10)


def test_mac_model_invalid_initialization():
    """Test error handling for invalid initialization parameters."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)

    # Test with num_devices mismatch (encoders list)
    with pytest.raises(ValueError, match="does not match the number of encoders"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=[SimpleEncoder(), SimpleEncoder()], decoders=SimpleDecoder, channel=channel, power_constraint=power_constraint, num_devices=3)

    # Test with num_devices mismatch (decoders list - separate decoding case)
    with pytest.raises(ValueError, match="does not match the number of decoders"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoders=[SimpleDecoder(), SimpleDecoder()], channel=channel, power_constraint=power_constraint, num_devices=3)

    # Test with num_devices mismatch (decoders list vs encoders list - separate decoding case)
    # This case actually triggers the mismatch between num_devices and len(encoders) first.
    with pytest.raises(ValueError, match="Provided num_devices .* does not match the number of encoders"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=[SimpleEncoder()], decoders=[SimpleDecoder(), SimpleDecoder()], channel=channel, power_constraint=power_constraint, num_devices=2)  # num_devices=2 != len(encoders)=1

    # Test missing num_devices when using classes/instances
    with pytest.raises(ValueError, match="num_devices must be specified"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoders=SimpleDecoder, channel=channel, power_constraint=power_constraint)
    with pytest.raises(ValueError, match="num_devices must be specified"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=SimpleEncoder(), decoders=SimpleDecoder(), channel=channel, power_constraint=power_constraint)

    # Test invalid encoder type
    # Match the actual error message (lowercase 'encoder')
    with pytest.raises(TypeError, match="Invalid type for encoder configuration"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=123, decoders=SimpleDecoder, channel=channel, power_constraint=power_constraint, num_devices=2)

    # Test invalid decoder type
    # Match the actual error message (lowercase 'decoder')
    with pytest.raises(TypeError, match="Invalid type for decoder configuration"):
        # Use 'decoders' (plural)
        MultipleAccessChannelModel(encoders=SimpleEncoder, decoders="abc", channel=channel, power_constraint=power_constraint, num_devices=2)


def test_mac_model_registry(mac_components):
    """Test that MAC model is correctly registered in ModelRegistry."""
    # Check registration
    assert "multiple_access_channel" in ModelRegistry._models

    # Create model through registry
    # Need to provide necessary args like encoders, decoders, num_devices
    # Use 'decoders' (plural) argument
    model = ModelRegistry.create("multiple_access_channel", encoders=mac_components["encoders"], decoders=mac_components["decoders"], channel=mac_components["channel"], power_constraint=mac_components["power_constraint"], num_devices=mac_components["num_devices"])

    assert isinstance(model, MultipleAccessChannelModel)
    assert model.num_devices == mac_components["num_devices"]


def test_mac_model_with_csi_and_noise():
    """Test MultipleAccessChannelModel forward pass with explicit CSI and noise."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    num_devices = 2
    # Use 'decoders' (plural) argument
    model = MultipleAccessChannelModel(channel=channel, power_constraint=power_constraint, encoders=SimpleEncoder, decoders=SimpleDecoder, num_devices=num_devices)

    # Create inputs
    inputs = [torch.randn(2, 10) for _ in range(num_devices)]

    # Create CSI and noise (assuming channel and decoder can handle them via **kwargs)
    csi = torch.randn(2, 5)  # Example shape
    noise = torch.randn(2, 5)  # Example shape

    # Run forward pass with CSI and noise
    output = model(inputs, csi=csi, noise=noise)

    # Basic output validation (single tensor for joint decoding)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 10)  # Joint decoder output dim is 10


def test_mac_model_invalid_forward_call():
    """Test error handling for invalid forward calls."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    # Use 'decoders' (plural) argument
    model = MultipleAccessChannelModel(channel=channel, power_constraint=power_constraint, encoders=SimpleEncoder, decoders=SimpleDecoder, num_devices=2)

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
    # Use 'decoders' (plural) key when unpacking
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
