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

    def forward(self, x):
        return self.layer(x)


class SimpleDecoder(nn.Module):
    """Simple decoder for testing MultipleAccessChannelModel."""

    def __init__(self, input_dim=5, output_dim=10):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def mac_components():
    """Fixture providing components for MultipleAccessChannelModel testing."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)

    return {"channel": channel, "power_constraint": power_constraint, "encoder": SimpleEncoder, "decoder": SimpleDecoder, "num_devices": 3}


def test_mac_model_initialization(mac_components):
    """Test MultipleAccessChannelModel initialization with default parameters."""
    model = MultipleAccessChannelModel(**mac_components)

    # Check basic components
    assert model.channel == mac_components["channel"]
    assert model.power_constraint == mac_components["power_constraint"]
    assert model.num_devices == mac_components["num_devices"]

    # Check encoder/decoder lists
    assert len(model.encoders) == model.num_devices
    assert len(model.decoders) == model.num_devices

    # Check all encoders and decoders are different instances
    assert model.shared_encoder is False
    assert model.shared_decoder is False
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

    # Check all encoders and decoders are the same instance
    for i in range(model.num_devices - 1):
        assert model.encoders[i] is model.encoders[i + 1]
        assert model.decoders[i] is model.decoders[i + 1]


def test_mac_model_encode_function(mac_components):
    """Test the encode function of MultipleAccessChannelModel."""
    model = MultipleAccessChannelModel(**mac_components)

    # Create test inputs
    inputs = [torch.randn(2, 10) for _ in range(model.num_devices)]

    # Test encoding all devices
    encoded = model.encode(inputs)

    # Check number of encoded signals
    assert len(encoded) == model.num_devices

    # Check each encoded signal shape
    for signal in encoded:
        assert signal.shape == (2, 5)

    # Test encoding specific devices
    device_indices = [0, 2]
    specific_inputs = [inputs[0], inputs[2]]
    specific_encoded = model.encode(specific_inputs, device_indices)

    # Check number of encoded signals
    assert len(specific_encoded) == len(device_indices)


def test_mac_model_decode_function(mac_components):
    """Test the decode function of MultipleAccessChannelModel."""
    model = MultipleAccessChannelModel(**mac_components)

    # Create test received signal
    received = torch.randn(2, 5)

    # Test decoding for all devices
    decoded = model.decode(received)

    # Check number of decoded signals
    assert len(decoded) == model.num_devices

    # Check each decoded signal shape
    for signal in decoded:
        assert signal.shape == (2, 10)

    # Test decoding for specific devices
    device_indices = [0, 2]
    specific_decoded = model.decode(received, device_indices)

    # Check number of decoded signals
    assert len(specific_decoded) == len(device_indices)


def test_mac_model_forward_pass(mac_components):
    """Test the forward pass of MultipleAccessChannelModel."""
    model = MultipleAccessChannelModel(**mac_components)

    # Create test inputs
    inputs = [torch.randn(2, 10) for _ in range(model.num_devices)]

    # Run forward pass
    outputs = model(inputs)

    # Check number of output signals
    assert len(outputs) == model.num_devices

    # Check each output signal shape
    for signal in outputs:
        assert signal.shape == (2, 10)


def test_mac_model_forward_with_device_indices(mac_components):
    """Test the forward pass with specific device indices."""
    model = MultipleAccessChannelModel(**mac_components)

    # Create test inputs for specific devices
    device_indices = [0, 2]
    inputs = [torch.randn(2, 10) for _ in range(len(device_indices))]

    # Run forward pass with device indices
    outputs = model(inputs, device_indices=device_indices)

    # Check number of output signals
    assert len(outputs) == len(device_indices)


def test_mac_model_set_encoder(mac_components):
    """Test setting encoders for specific devices."""
    model = MultipleAccessChannelModel(**mac_components)

    # Define a different encoder
    class DifferentEncoder(nn.Module):
        def __init__(self, input_dim=10, output_dim=5):
            super().__init__()
            self.layer = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))

        def forward(self, x):
            return self.layer(x)

    # Set encoder for specific device
    model.set_encoder(DifferentEncoder, device_index=1)

    # Check that only the specified encoder was changed
    assert isinstance(model.encoders[1].layer, nn.Sequential)
    assert isinstance(model.encoders[0].layer, nn.Linear)
    assert isinstance(model.encoders[2].layer, nn.Linear)

    # Set encoder for all devices
    model.set_encoder(DifferentEncoder)

    # Check that all encoders were changed
    for i in range(model.num_devices):
        assert isinstance(model.encoders[i].layer, nn.Sequential)


def test_mac_model_set_decoder(mac_components):
    """Test setting decoders for specific devices."""
    model = MultipleAccessChannelModel(**mac_components)

    # Define a different decoder
    class DifferentDecoder(nn.Module):
        def __init__(self, input_dim=5, output_dim=10):
            super().__init__()
            self.layer = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))

        def forward(self, x):
            return self.layer(x)

    # Set decoder for specific device
    model.set_decoder(DifferentDecoder, device_index=1)

    # Check that only the specified decoder was changed
    assert isinstance(model.decoders[1].layer, nn.Sequential)
    assert isinstance(model.decoders[0].layer, nn.Linear)
    assert isinstance(model.decoders[2].layer, nn.Linear)

    # Set decoder for all devices
    model.set_decoder(DifferentDecoder)

    # Check that all decoders were changed
    for i in range(model.num_devices):
        assert isinstance(model.decoders[i].layer, nn.Sequential)


def test_mac_model_invalid_inputs():
    """Test error handling for invalid inputs."""
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    model = MultipleAccessChannelModel(channel=channel, power_constraint=power_constraint, encoder=SimpleEncoder, decoder=SimpleDecoder, num_devices=2)

    # Test with empty input list
    with pytest.raises(ValueError):
        model([])

    # Test with mismatched number of inputs and device_indices
    with pytest.raises(ValueError):
        model([torch.randn(2, 10)], device_indices=[0, 1])

    # Test with invalid device index
    with pytest.raises(IndexError):
        model.set_encoder(SimpleEncoder, device_index=5)

    with pytest.raises(IndexError):
        model.set_decoder(SimpleDecoder, device_index=-1)


def test_mac_model_registry():
    """Test that MAC model is correctly registered in ModelRegistry."""
    # Check registration
    assert "mac" in ModelRegistry._models

    # Create model through registry
    channel = AWGNChannel(avg_noise_power=0.1)
    power_constraint = TotalPowerConstraint(total_power=1.0)

    model = ModelRegistry.create("mac", channel=channel, power_constraint=power_constraint, num_devices=2)

    assert isinstance(model, MultipleAccessChannelModel)
    assert model.num_devices == 2
