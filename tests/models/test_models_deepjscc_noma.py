"""Tests for the Yilmaz2023DeepJSCCNOMA model."""
import torch

from kaira.channels import AWGNChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models.image import (
    Yilmaz2023DeepJSCCNOMADecoder,
    Yilmaz2023DeepJSCCNOMAEncoder,
    Yilmaz2023DeepJSCCNOMAModel,
)
from kaira.models.registry import ModelRegistry


def test_yilmaz2023_deepjscc_noma_encoder():
    """Test the encoder component."""
    encoder = Yilmaz2023DeepJSCCNOMAEncoder(C=3, latent_dim=16)
    x = torch.randn(4, 3, 32, 32)
    csi = torch.ones(4)

    output = encoder((x, csi))

    # Output should be [batch_size, 2, sqrt(latent_dim), sqrt(latent_dim)]
    assert output.shape == (4, 2, 4, 4)


def test_yilmaz2023_deepjscc_noma_decoder():
    """Test the decoder component."""
    decoder = Yilmaz2023DeepJSCCNOMADecoder(latent_dim=16)
    x = torch.randn(4, 2, 4, 4)  # [batch_size, 2, sqrt(latent_dim), sqrt(latent_dim)]
    csi = torch.ones(4)

    output = decoder((x, csi))

    # Output should be [batch_size, 3, height, width]
    assert output.shape == (4, 3, 32, 32)

    # Test shared decoder
    shared_decoder = Yilmaz2023DeepJSCCNOMADecoder(latent_dim=16, num_devices=2, shared_decoder=True)
    # The input for shared decoder would combine signals from all devices
    x = torch.randn(4, 4, 4, 4)  # [batch_size, 2*num_devices, sqrt(latent_dim), sqrt(latent_dim)]

    output = shared_decoder((x, csi))

    # Output should include channels for all devices
    assert output.shape == (4, 6, 32, 32)  # [batch_size, 3*num_devices, height, width]


def test_yilmaz2023_deepjscc_noma_instantiation():
    """Test that Yilmaz2023DeepJSCCNOMA can be instantiated with default components."""
    channel = AWGNChannel(snr_db=10.0)
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMAModel(
        channel=channel,
        power_constraint=constraint,
        num_devices=2,
        M=0.5,
    )
    assert isinstance(model, Yilmaz2023DeepJSCCNOMAModel)
    assert model.num_devices == 2

    # Check that encoders and decoders are properly instantiated
    assert len(model.encoders) == 2  # Default is not shared
    assert len(model.decoders) == 2  # Default is not shared
    assert isinstance(model.encoders[0], Yilmaz2023DeepJSCCNOMAEncoder)
    assert isinstance(model.decoders[0], Yilmaz2023DeepJSCCNOMADecoder)


def test_yilmaz2023_deepjscc_noma_forward():
    """Test the forward pass of Yilmaz2023DeepJSCCNOMA with default components."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMAModel(
        channel=channel,
        power_constraint=constraint,
        num_devices=2,
        M=0.5,
        latent_dim=16,
    )

    # Create dummy input: [batch_size, num_devices, channels, height, width]
    x = torch.randn(4, 2, 3, 32, 32)
    csi = torch.ones(4)  # SNR values

    # Run forward pass
    output = model(x, csi)

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
        num_devices=3,
        M=0.5,
    )

    assert isinstance(model, Yilmaz2023DeepJSCCNOMAModel)
    assert model.num_devices == 3


def test_yilmaz2023_deepjscc_noma_shared_components():
    """Test Yilmaz2023DeepJSCCNOMA with shared encoder/decoder."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMAModel(
        channel=channel,
        power_constraint=constraint,
        num_devices=3,
        M=0.5,
        latent_dim=16,
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
    output = model(x, csi)

    # Check output shape
    assert output.shape == (2, 3, 3, 32, 32)


def test_yilmaz2023_deepjscc_noma_perfect_sic():
    """Test Yilmaz2023DeepJSCCNOMA with perfect successive interference cancellation."""
    channel = AWGNChannel()
    constraint = TotalPowerConstraint(total_power=1.0)
    model = Yilmaz2023DeepJSCCNOMAModel(
        channel=channel,
        power_constraint=constraint,
        num_devices=2,
        M=0.5,
        latent_dim=16,
        use_perfect_sic=True,
    )

    # Create dummy input
    x = torch.randn(2, 2, 3, 32, 32)  # [batch_size, num_devices, channels, height, width]
    csi = torch.ones(2)  # SNR values

    # Run forward pass
    output = model(x, csi)

    # Check output shape
    assert output.shape == (2, 2, 3, 32, 32)