# tests/test_models/test_deepjsccq.py
import pytest
import torch

from kaira.models.image.tung2022_deepjscc_q import Tung2022DeepJSCCQDecoder, Tung2022DeepJSCCQEncoder


@pytest.fixture
def encoder():
    """Fixture for creating a DeepJSCCQEncoder."""
    return Tung2022DeepJSCCQEncoder(N=32, M=64)


@pytest.fixture
def decoder():
    """Fixture for creating a DeepJSCCQDecoder."""
    return Tung2022DeepJSCCQDecoder(N=32, M=64)


@pytest.fixture
def sample_image():
    """Fixture for creating a sample image tensor."""
    return torch.randn(1, 3, 32, 32)  # Example image: batch size 1, 3 channels, 32x32


def test_deepjsccq_encoder_initialization(encoder):
    """Test DeepJSCCQEncoder initialization."""
    assert isinstance(encoder, Tung2022DeepJSCCQEncoder)
    assert len(encoder.g_a) == 9  # Verify number of layers


def test_deepjsccq_decoder_initialization(decoder):
    """Test DeepJSCCQDecoder initialization."""
    assert isinstance(decoder, Tung2022DeepJSCCQDecoder)
    assert len(decoder.g_s) == 10  # Verify number of layers


def test_deepjsccq_encoder_forward(encoder, sample_image):
    """Test DeepJSCCQEncoder forward pass."""
    encoded_image = encoder(sample_image)
    assert isinstance(encoded_image, torch.Tensor)
    # Check output shape (adjust based on expected downsampling)
    assert encoded_image.shape == (1, 64, 2, 2)


def test_deepjsccq_decoder_forward(decoder, encoder, sample_image):
    """Test DeepJSCCQDecoder forward pass."""
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape  # Check output shape


def test_deepjsccq_encoder_decoder_roundtrip(encoder, decoder, sample_image):
    """Test a full encoder-decoder roundtrip."""
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)

    # Check that the decoded image is not identical to the original
    assert not torch.equal(decoded_image, sample_image)

    # Check that the decoded image has the same shape as the original
    assert decoded_image.shape == sample_image.shape

    # Basic check for reasonable values (you might need more specific checks)
    assert torch.all(decoded_image > -5) and torch.all(decoded_image < 5)


@pytest.mark.parametrize("N, M", [(16, 32), (64, 128)])
def test_deepjsccq_encoder_decoder_different_sizes(N, M):
    """Test encoder and decoder with different N and M values."""
    encoder = Tung2022DeepJSCCQEncoder(N=N, M=M)
    decoder = Tung2022DeepJSCCQDecoder(N=N, M=M)
    sample_image = torch.randn(1, 3, 32, 32)
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert decoded_image.shape == sample_image.shape
