import pytest
import torch
from kaira.models.image.compressors.neural import NeuralCompressor

@pytest.fixture
def sample_image():
    """Fixture that provides a sample image tensor for testing."""
    return torch.rand(1, 3, 32, 32)  # Random image tensor

@pytest.fixture
def neural_compressor():
    """Fixture that provides a NeuralCompressor instance for testing."""
    return NeuralCompressor(method="bmshj2018_factorized", quality=4)

def test_neural_compressor_initialization():
    """Test NeuralCompressor initialization with valid parameters."""
    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=4)
    assert compressor.method == "bmshj2018_factorized"
    assert compressor.quality == 4
    assert compressor.max_bits_per_image is None

def test_neural_compressor_forward(sample_image, neural_compressor):
    """Test NeuralCompressor forward pass."""
    output = neural_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape

def test_neural_compressor_forward_with_bits(sample_image, neural_compressor):
    """Test NeuralCompressor forward pass with bits per image."""
    neural_compressor.return_bits = True
    output, bits_per_image = neural_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)

def test_neural_compressor_forward_with_compressed_data(sample_image, neural_compressor):
    """Test NeuralCompressor forward pass with compressed data."""
    neural_compressor.return_compressed_data = True
    output, compressed_data = neural_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1
    assert isinstance(compressed_data[0], dict)

def test_neural_compressor_forward_with_bits_and_compressed_data(sample_image, neural_compressor):
    """Test NeuralCompressor forward pass with bits and compressed data."""
    neural_compressor.return_bits = True
    neural_compressor.return_compressed_data = True
    output, bits_per_image, compressed_data = neural_compressor(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1
    assert isinstance(compressed_data[0], dict)

def test_neural_compressor_invalid_method():
    """Test NeuralCompressor initialization with invalid method."""
    with pytest.raises(ValueError):
        NeuralCompressor(method="invalid_method", quality=4)

def test_neural_compressor_invalid_quality():
    """Test NeuralCompressor initialization with invalid quality."""
    with pytest.raises(ValueError):
        NeuralCompressor(method="bmshj2018_factorized", quality=10)

def test_neural_compressor_collect_stats(sample_image, neural_compressor):
    """Test NeuralCompressor with collect_stats enabled."""
    neural_compressor.collect_stats = True
    neural_compressor(sample_image)
    stats = neural_compressor.get_stats()
    assert isinstance(stats, dict)
    assert "total_bits" in stats
    assert "avg_quality" in stats
    assert "processing_time" in stats

def test_neural_compressor_get_bits_per_image(sample_image, neural_compressor):
    """Test NeuralCompressor get_bits_per_image method."""
    bits_per_image = neural_compressor.get_bits_per_image(sample_image)
    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)
