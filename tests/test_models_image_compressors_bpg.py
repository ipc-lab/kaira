import pytest
import torch
from PIL import Image
from torchvision import transforms

from kaira.models.image.compressors.bpg import BPGCompressor


@pytest.fixture
def sample_image():
    """Fixture that provides a sample image tensor for testing."""
    img = Image.new("RGB", (32, 32), color="red")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dimension


@pytest.fixture
def bpg_compressor():
    """Fixture that provides a BPGCompressor instance for testing."""
    return BPGCompressor(quality=30)


def test_bpg_compressor_initialization():
    """Test BPGCompressor initialization with valid parameters."""
    compressor = BPGCompressor(quality=30)
    assert compressor.quality == 30
    assert compressor.max_bits_per_image is None


def test_bpg_compressor_forward(sample_image, bpg_compressor):
    """Test BPGCompressor forward pass."""
    output = bpg_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape


def test_bpg_compressor_forward_with_bits(sample_image, bpg_compressor):
    """Test BPGCompressor forward pass with bits per image."""
    bpg_compressor.return_bits = True
    output, bits_per_image = bpg_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1


def test_bpg_compressor_forward_with_compressed_data(sample_image, bpg_compressor):
    """Test BPGCompressor forward pass with compressed data."""
    bpg_compressor.return_compressed_data = True
    output, compressed_data = bpg_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1
    assert isinstance(compressed_data[0], bytes)


def test_bpg_compressor_forward_with_bits_and_compressed_data(sample_image, bpg_compressor):
    """Test BPGCompressor forward pass with bits and compressed data."""
    bpg_compressor.return_bits = True
    bpg_compressor.return_compressed_data = True
    output, bits_per_image, compressed_data = bpg_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1
    assert isinstance(compressed_data[0], bytes)


def test_bpg_compressor_invalid_quality():
    """Test BPGCompressor initialization with invalid quality."""
    with pytest.raises(ValueError):
        BPGCompressor(quality=60)


def test_bpg_compressor_invalid_executable_path():
    """Test BPGCompressor initialization with invalid executable path."""
    with pytest.raises(ValueError):
        BPGCompressor(quality=30, bpg_encoder_path="invalid;path")


def test_bpg_compressor_collect_stats(sample_image, bpg_compressor):
    """Test BPGCompressor with collect_stats enabled."""
    bpg_compressor.collect_stats = True
    bpg_compressor(sample_image)
    stats = bpg_compressor.get_stats()
    assert isinstance(stats, dict)
    assert "total_bits" in stats
    assert "avg_quality" in stats
    assert "processing_time" in stats


def test_bpg_compressor_get_bits_per_image(sample_image, bpg_compressor):
    """Test BPGCompressor get_bits_per_image method."""
    bits_per_image = bpg_compressor.get_bits_per_image(sample_image)
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1
    assert isinstance(bits_per_image[0], int)
