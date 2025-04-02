from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from kaira.models.image.compressors.neural import NeuralCompressor


@pytest.fixture
def sample_image():
    """Fixture that provides a sample image tensor for testing."""
    img = Image.new("RGB", (32, 32), color="red")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dimension


@pytest.fixture
def mock_compressai_model():
    """Mock for CompressAI model."""
    model = MagicMock()
    model.return_value = {"x_hat": torch.ones(1, 3, 32, 32), "likelihoods": {"y": torch.ones(1, 3, 8, 8) * 0.5, "z": torch.ones(1, 3, 4, 4) * 0.5}}
    model.compress.return_value = {"strings": {"y": b"test", "z": b"test"}, "shape": {"y": [8, 8], "z": [4, 4]}}
    return model


def test_neural_compressor_initialization():
    """Test NeuralCompressor initialization with valid parameters."""
    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5)
    assert compressor.quality == 5
    assert compressor.max_bits_per_image is None

    compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=1000)
    assert compressor.quality is None
    assert compressor.max_bits_per_image == 1000

    # Test lazy loading default
    assert compressor.lazy_loading is True


def test_neural_compressor_initialization_errors():
    """Test NeuralCompressor initialization with invalid parameters."""
    # Test missing both parameters
    with pytest.raises(ValueError, match="At least one of the two parameters must be provided"):
        NeuralCompressor(method="bmshj2018_factorized")

    # Test invalid method
    with pytest.raises(ValueError, match="Method 'invalid_method' is not supported"):
        NeuralCompressor(method="invalid_method", quality=5)

    # Test invalid quality
    with pytest.raises(ValueError, match="Quality must be in"):
        NeuralCompressor(method="bmshj2018_factorized", quality=10)

    # Test invalid metric
    with pytest.raises(ValueError, match="Metric must be 'ms-ssim' or 'mse'"):
        NeuralCompressor(method="bmshj2018_factorized", quality=5, metric="invalid_metric")


def test_neural_compressor_get_model():
    """Test model loading and caching."""
    with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
        mock_model = MagicMock()
        mock_factory.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Create compressor with lazy loading
        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, lazy_loading=True)

        # First call should load the model
        compressor.get_model(5)
        mock_factory.assert_called_once_with(quality=5, pretrained=True, metric="mse")

        # Second call should use cached model
        mock_factory.reset_mock()
        compressor.get_model(5)
        mock_factory.assert_not_called()

        # Different quality should load new model
        compressor.get_model(4)
        mock_factory.assert_called_once_with(quality=4, pretrained=True, metric="mse")


def test_neural_compressor_eager_loading():
    """Test eager model loading (non-lazy)."""
    with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
        mock_model = MagicMock()
        mock_factory.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # With quality specified, should load one model
        NeuralCompressor(method="bmshj2018_factorized", quality=5, lazy_loading=False)
        mock_factory.assert_called_once_with(quality=5, pretrained=True, metric="mse")

        # With max_bits_per_image, should load all models
        mock_factory.reset_mock()
        NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=1000, lazy_loading=False)
        assert mock_factory.call_count == 8  # 8 possible qualities for this method


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_forward_fixed_quality(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor forward pass with fixed quality."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5)
    output = compressor(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_forward_fixed_quality_with_bits(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor forward pass with bits per image."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=True)
    output, bits_per_image = compressor(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_forward_fixed_quality_with_compressed_data(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor forward pass with compressed data."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=False, return_compressed_data=True)
    output, compressed_data = compressor(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_forward_fixed_quality_with_bits_and_compressed_data(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor forward pass with bits and compressed data."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=True, return_compressed_data=True)
    output, bits_per_image, compressed_data = compressor(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_bit_constrained_mode(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor in bit-constrained mode."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    # Set bits that would be produced by model mock
    bits_per_image = -torch.log2(torch.tensor(0.5)) * (3 * 8 * 8 + 3 * 4 * 4)
    max_bits = bits_per_image - 10  # Set max_bits below what the model would produce

    compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=max_bits)
    with patch.object(compressor, "compute_bits_compressai", return_value=torch.tensor([bits_per_image])):
        # Should try different qualities and settle on the lowest one
        output = compressor(sample_image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_collect_stats(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor with collect_stats enabled."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, collect_stats=True)
    compressor(sample_image)

    stats = compressor.get_stats()
    assert isinstance(stats, dict)
    assert "total_bits" in stats
    assert "avg_quality" in stats
    assert "model_name" in stats
    assert "metric" in stats
    assert "processing_time" in stats


def test_neural_compressor_get_stats_when_not_collected():
    """Test get_stats when stats were not collected."""
    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, collect_stats=False)

    with pytest.warns(UserWarning, match="Statistics not collected"):
        stats = compressor.get_stats()

    assert stats == {}


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_collect_stats_reset(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor stats reset functionality."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    # Create compressor with stats collection enabled
    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, collect_stats=True)
    
    # Process an image
    compressor(sample_image)
    
    # Get stats after processing
    stats1 = compressor.get_stats()
    assert "total_bits" in stats1
    assert stats1["total_bits"] > 0
    
    # Reset stats
    compressor.reset_stats()
    
    # Stats should be reset
    stats2 = compressor.get_stats()
    assert "total_bits" in stats2
    assert stats2["total_bits"] == 0


@patch("compressai.zoo.bmshj2018_factorized")
def test_neural_compressor_get_bits_per_image(mock_factory, sample_image, mock_compressai_model):
    """Test neural compressor get_bits_per_image method."""
    mock_model = mock_compressai_model
    mock_factory.return_value.eval.return_value = mock_model

    compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=False)
    bits_per_image = compressor.get_bits_per_image(sample_image)

    assert isinstance(bits_per_image, torch.Tensor)
    assert bits_per_image.shape == (1,)

    # Verify return_bits was temporarily changed and then restored
    assert compressor.return_bits is False
