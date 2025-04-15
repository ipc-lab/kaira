"""Tests for Tung2022DeepJSCCQ2 model."""
import pytest
import torch

from kaira.models.image.tung2022_deepjscc_q import (
    Tung2022DeepJSCCQ2Decoder,
    Tung2022DeepJSCCQ2Encoder,
)


@pytest.fixture
def encoder2():
    """Fixture for creating a DeepJSCCQ2Encoder."""
    return Tung2022DeepJSCCQ2Encoder(N=32, M=64, csi_length=1)


@pytest.fixture
def decoder2():
    """Fixture for creating a DeepJSCCQ2Decoder."""
    return Tung2022DeepJSCCQ2Decoder(N=32, M=64, csi_length=1)


@pytest.fixture
def sample_image():
    """Fixture for creating a sample image tensor."""
    return torch.randn(1, 3, 32, 32)  # Example image: batch size 1, 3 channels, 32x32


@pytest.fixture
def sample_snr():
    """Fixture for creating a sample SNR tensor."""
    return torch.randn(1, 1)


def test_deepjsccq2_encoder_initialization(encoder2):
    """Test DeepJSCCQ2Encoder initialization."""
    assert isinstance(encoder2, Tung2022DeepJSCCQ2Encoder)
    # Verify number of layers
    assert len(encoder2.g_a) == 13


def test_deepjsccq2_decoder_initialization(decoder2):
    """Test DeepJSCCQ2Decoder initialization."""
    assert isinstance(decoder2, Tung2022DeepJSCCQ2Decoder)
    # Verify number of layers
    assert len(decoder2.g_s) == 14


def test_deepjsccq2_encoder_forward(encoder2, sample_image, sample_snr):
    """Test DeepJSCCQ2Encoder forward pass with SNR."""
    encoded_image = encoder2(sample_image, sample_snr)  # Pass arguments separately
    assert isinstance(encoded_image, torch.Tensor)
    # Check output shape (adjust based on expected downsampling)
    # The output shape depends on the architecture, so adjust accordingly
    assert encoded_image.shape == (1, 64, 8, 8)


def test_deepjsccq2_decoder_forward(decoder2, encoder2, sample_image, sample_snr):
    """Test DeepJSCCQ2Decoder forward pass with SNR."""
    encoded_image = encoder2(sample_image, sample_snr)  # Pass arguments separately
    decoded_image = decoder2(encoded_image, sample_snr)  # Pass arguments separately
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape  # Check output shape


def test_deepjsccq2_encoder_decoder_roundtrip(encoder2, decoder2, sample_image, sample_snr):
    """Test a full encoder-decoder roundtrip with SNR."""
    encoded_image = encoder2(sample_image, sample_snr)  # Pass arguments separately
    decoded_image = decoder2(encoded_image, sample_snr)  # Pass arguments separately

    # Check that the decoded image is not identical to the original
    assert not torch.equal(decoded_image, sample_image)

    # Check that the decoded image has the same shape as the original
    assert decoded_image.shape == sample_image.shape

    # Basic check for reasonable values (you might need more specific checks)
    assert torch.all(decoded_image > -5) and torch.all(decoded_image < 5)


@pytest.mark.parametrize("N, M, csi_length", [(16, 32, 2), (64, 128, 4)])
def test_deepjsccq2_encoder_decoder_different_sizes(N, M, csi_length):
    """Test encoder and decoder with different N, M, and csi_length values."""
    encoder = Tung2022DeepJSCCQ2Encoder(N=N, M=M, csi_length=csi_length)
    decoder = Tung2022DeepJSCCQ2Decoder(N=N, M=M, csi_length=csi_length)
    sample_image = torch.randn(1, 3, 32, 32)
    sample_snr = torch.randn(1, csi_length)
    encoded_image = encoder(sample_image, sample_snr)  # Pass arguments separately
    decoded_image = decoder(encoded_image, sample_snr)  # Pass arguments separately
    assert decoded_image.shape == sample_image.shape
