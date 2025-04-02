"""Tests for advanced Multiple Access Channel (MAC) model scenarios."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, RayleighFadingChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models import MultipleAccessChannelModel
from kaira.models import ModelRegistry


class ComplexEncoder(nn.Module):
    """Complex encoder with multiple layers for testing MAC models."""
    
    def __init__(self, input_dim=10, hidden_dim=8, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ComplexDecoder(nn.Module):
    """Complex decoder with multiple layers for testing MAC models."""
    
    def __init__(self, input_dim=5, hidden_dim=8, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


@pytest.fixture
def mac_components():
    """Fixture providing components for testing MAC models."""
    channel = AWGNChannel(snr_db=15)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    
    return {
        "channel": channel,
        "power_constraint": power_constraint,
        "encoder": ComplexEncoder,
        "decoder": ComplexDecoder,
        "num_devices": 3
    }


@pytest.fixture
def heterogeneous_mac_model(mac_components):
    """Fixture providing a MAC model with heterogeneous encoders and decoders."""
    model = MultipleAccessChannelModel(**mac_components)
    
    # Set different encoders for each device
    class SmallEncoder(nn.Module):
        def __init__(self, input_dim=10, output_dim=3):
            super().__init__()
            self.net = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.net(x)
    
    class LargeEncoder(nn.Module):
        def __init__(self, input_dim=10, output_dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Set different encoders for different devices
    model.set_encoder(SmallEncoder, device_index=0)
    model.set_encoder(ComplexEncoder, device_index=1)  # Keep original for device 1
    model.set_encoder(LargeEncoder, device_index=2)
    
    return model


def test_mac_model_with_fading_channel(mac_components):
    """Test MAC model with Rayleigh fading channel."""
    # Replace the AWGN channel with a fading channel
    mac_components["channel"] = RayleighFadingChannel()
    
    # Create model
    model = MultipleAccessChannelModel(**mac_components)
    
    # Create test inputs
    batch_size = 4
    input_dim = 10
    inputs = [torch.randn(batch_size, input_dim) for _ in range(model.num_devices)]
    
    # Test forward pass
    outputs = model(inputs)
    
    # Check outputs
    assert len(outputs) == model.num_devices
    for output in outputs:
        assert output.shape == (batch_size, input_dim)


def test_mac_model_with_csi(mac_components):
    """Test MAC model with channel state information (CSI)."""
    # Create model
    model = MultipleAccessChannelModel(**mac_components)
    
    # Create test inputs
    batch_size = 4
    input_dim = 10
    inputs = [torch.randn(batch_size, input_dim) for _ in range(model.num_devices)]
    
    # Create CSI tensor
    csi = torch.ones(batch_size)  # Represent SNR values
    
    # Test forward pass with CSI
    outputs = model(inputs, csi=csi)
    
    # Check outputs
    assert len(outputs) == model.num_devices
    for output in outputs:
        assert output.shape == (batch_size, input_dim)


def test_heterogeneous_mac_encoders(heterogeneous_mac_model):
    """Test MAC model with different encoder architectures for each device."""
    # Create test inputs
    batch_size = 4
    input_dim = 10
    inputs = [torch.randn(batch_size, input_dim) for _ in range(heterogeneous_mac_model.num_devices)]
    
    # Get encoded outputs directly to check their shapes
    encoded = heterogeneous_mac_model.encode(inputs)
    
    # Check the encoded outputs have different shapes based on the encoders
    assert encoded[0].shape == (batch_size, 3)  # SmallEncoder
    assert encoded[1].shape == (batch_size, 5)  # ComplexEncoder (original)
    assert encoded[2].shape == (batch_size, 8)  # LargeEncoder
    
    # Test complete forward pass still works despite different encoded shapes
    outputs = heterogeneous_mac_model(inputs)
    
    # The decoder should still produce the correct output shapes
    assert len(outputs) == heterogeneous_mac_model.num_devices
    for output in outputs:
        assert output.shape == (batch_size, input_dim)


def test_mac_model_device_dropouts():
    """Test MAC model when some devices don't transmit."""
    # Create basic components
    channel = AWGNChannel(snr_db=15)
    power_constraint = TotalPowerConstraint(total_power=1.0)
    
    # Create model with 5 devices
    model = MultipleAccessChannelModel(
        channel=channel,
        power_constraint=power_constraint,
        encoder=ComplexEncoder,
        decoder=ComplexDecoder,
        num_devices=5
    )
    
    # Create test data for only 3 of the 5 devices
    batch_size = 4
    input_dim = 10
    active_device_indices = [0, 2, 4]  # Only devices 0, 2, and 4 are active
    inputs = [torch.randn(batch_size, input_dim) for _ in range(len(active_device_indices))]
    
    # Run forward pass with specific device indices
    outputs = model(inputs, device_indices=active_device_indices)
    
    # Should only get outputs for active devices
    assert len(outputs) == len(active_device_indices)
    for output in outputs:
        assert output.shape == (batch_size, input_dim)


def test_mac_model_training_compatibility(mac_components):
    """Test that the MAC model can be trained with backpropagation."""
    # Create model
    model = MultipleAccessChannelModel(**mac_components)
    
    # Create fixed test inputs and target outputs
    batch_size = 16
    input_dim = 10
    inputs = [torch.randn(batch_size, input_dim) for _ in range(model.num_devices)]
    targets = [input_tensor.clone() for input_tensor in inputs]  # Identity mapping as target
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Initial loss
    outputs = model(inputs)
    initial_loss = sum([torch.mean((outputs[i] - targets[i]) ** 2) for i in range(len(outputs))])
    
    # Run a few optimization steps
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = sum([torch.mean((outputs[i] - targets[i]) ** 2) for i in range(len(outputs))])
        loss.backward()
        optimizer.step()
    
    # Final loss
    outputs = model(inputs)
    final_loss = sum([torch.mean((outputs[i] - targets[i]) ** 2) for i in range(len(outputs))])
    
    # Loss should decrease
    assert final_loss < initial_loss


def test_mac_model_performance_with_varying_noise(mac_components):
    """Test MAC model performance under varying noise conditions."""
    # Create test inputs
    batch_size = 20
    input_dim = 10
    inputs = [torch.randn(batch_size, input_dim) for _ in range(mac_components["num_devices"])]
    
    # Test with different SNR values
    snr_values = [5, 10, 15, 20, 25]
    avg_errors = []
    
    for snr_db in snr_values:
        # Update channel
        mac_components["channel"] = AWGNChannel(snr_db=snr_db)
        
        # Create model
        model = MultipleAccessChannelModel(**mac_components)
        
        # Evaluate
        outputs = model(inputs)
        
        # Calculate average error across devices
        avg_error = 0
        for i in range(len(inputs)):
            avg_error += torch.mean(torch.abs(outputs[i] - inputs[i])).item()
        avg_error /= len(inputs)
        avg_errors.append(avg_error)
    
    # Errors should decrease as SNR increases
    for i in range(len(snr_values) - 1):
        assert avg_errors[i] >= avg_errors[i + 1]