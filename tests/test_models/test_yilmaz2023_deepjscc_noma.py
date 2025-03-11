"""Tests for the Yilmaz2023DeepJSCCNOMA model."""
import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models.image import Yilmaz2023DeepJSCCNOMA
from kaira.models.registry import ModelRegistry


class SimpleEncoder(nn.Module):
    def __init__(self, C=3):
        super().__init__()
        self.conv = nn.Conv2d(C, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 2 * 16 * 16)  # Assuming 32x32 inputs, 16x16 outputs with 2 channels
        
    def forward(self, input_data):
        x, _ = input_data  # Unpack the input and CSI
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), 2, 16, 16)  # 2-channel output


class SimpleDecoder(nn.Module):
    def __init__(self, C=None):
        super().__init__()
        self.conv = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 16 * 16, 3 * 32 * 32)  # Assuming 16x16 inputs, 32x32 outputs with 3 channels
        
    def forward(self, input_data):
        x, _ = input_data  # Unpack the input and CSI
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(x.size(0), 3, 32, 32)  # 3-channel output for RGB


def test_yilmaz2023_deepjscc_noma_instantiation():
    """Test that Yilmaz2023DeepJSCCNOMA can be instantiated."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMA(
        channel=channel,
        power_constraint=constraint,
        encoder=SimpleEncoder,
        decoder=SimpleDecoder,
        num_devices=2,
        M=0.5,
    )
    assert isinstance(model, Yilmaz2023DeepJSCCNOMA)
    assert model.num_devices == 2


def test_yilmaz2023_deepjscc_noma_forward():
    """Test the forward pass of Yilmaz2023DeepJSCCNOMA."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMA(
        channel=channel,
        power_constraint=constraint,
        encoder=SimpleEncoder,
        decoder=SimpleDecoder,
        num_devices=2,
        M=0.5,
    )
    
    # Create dummy input: [batch_size, num_devices, channels, height, width]
    x = torch.randn(4, 2, 3, 32, 32)
    csi = torch.ones(4)  # SNR values
    
    # Run forward pass
    output = model((x, csi))
    
    # Check output shape
    assert output.shape == (4, 2, 3, 32, 32)


def test_yilmaz2023_deepjscc_noma_registry():
    """Test that Yilmaz2023DeepJSCCNOMA is properly registered."""
    assert "deepjscc_noma" in ModelRegistry._models
    
    # Check model can be created from registry
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    
    model = ModelRegistry.create(
        "deepjscc_noma",
        channel=channel,
        power_constraint=constraint,
        encoder=SimpleEncoder,
        decoder=SimpleDecoder,
        num_devices=3,
        M=0.5,
    )
    
    assert isinstance(model, Yilmaz2023DeepJSCCNOMA)
    assert model.num_devices == 3


def test_yilmaz2023_deepjscc_noma_shared_components():
    """Test Yilmaz2023DeepJSCCNOMA with shared encoder/decoder."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMA(
        channel=channel,
        power_constraint=constraint,
        encoder=SimpleEncoder,
        decoder=SimpleDecoder,
        num_devices=3,
        M=0.5,
        shared_encoder=True,
        shared_decoder=True,
    )
    
    # Check that we only have one encoder and one decoder
    assert len(model.encoders) == 1
    assert len(model.decoders) == 1
    
    # Create dummy input
    x = torch.randn(2, 3, 3, 32, 32)  # [batch_size, num_devices, channels, height, width]
    csi = torch.ones(2)  # SNR values
    
    # Run forward pass
    output = model((x, csi))
    
    # Check output shape
    assert output.shape == (2, 3, 3, 32, 32)