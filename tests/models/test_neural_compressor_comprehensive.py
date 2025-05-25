"""Comprehensive tests for Neural Compressor to improve coverage.

These tests focus on exercising the actual implementation rather than mocking, targeting the
specific functionality that was missing from the coverage report.
"""

import warnings
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from kaira.models.image.compressors.neural import NeuralCompressor


@pytest.fixture
def sample_image_batch():
    """Fixture providing a batch of sample images for testing."""
    # Create multiple small images for batch testing
    images = []
    for i in range(3):
        img = Image.new("RGB", (32, 32), color=(i * 80, i * 80, i * 80))
        transform = transforms.ToTensor()
        images.append(transform(img))
    return torch.stack(images)


@pytest.fixture
def mock_compressai_zoo():
    """Mock the CompressAI zoo with realistic model behavior."""

    def create_mock_model(quality, metric="mse"):
        """Create a realistic mock model."""
        model = Mock()
        model.eval.return_value = model
        model.to.return_value = model

        def mock_forward(x):
            batch_size = x.shape[0]
            # Create realistic likelihood values based on quality
            likelihood_scale = 0.1 + (quality / 10.0)  # Higher quality = higher likelihood

            return {"x_hat": torch.rand_like(x) * 0.1 + x * 0.9, "likelihoods": {"y": torch.ones(batch_size, 3, 8, 8) * likelihood_scale, "z": torch.ones(batch_size, 3, 4, 4) * likelihood_scale}}  # Slight reconstruction noise

        def mock_compress(x):
            return {"strings": {"y": b"compressed_y", "z": b"compressed_z"}, "shape": {"y": [8, 8], "z": [4, 4]}}

        model.side_effect = mock_forward
        model.compress = mock_compress
        return model

    # Mock the zoo functions
    mock_methods = {}
    for method in ["bmshj2018_factorized", "cheng2020_anchor", "mbt2018"]:
        mock_methods[method] = lambda quality, pretrained=True, metric="mse", method=method: create_mock_model(quality, metric)

    return mock_methods


class TestNeuralCompressorRealFunctionality:
    """Test real functionality of neural compressor without excessive mocking."""

    def test_compute_bits_compressai_realistic(self):
        """Test the compute_bits_compressai method with realistic data."""
        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5)

        # Create realistic CompressAI output
        batch_size = 2
        result = {"x_hat": torch.randn(batch_size, 3, 32, 32), "likelihoods": {"y": torch.ones(batch_size, 3, 8, 8) * 0.5, "z": torch.ones(batch_size, 3, 4, 4) * 0.25}}  # probability = 0.5  # probability = 0.25

        bits = compressor.compute_bits_compressai(result)

        assert bits.shape == (batch_size,)
        assert torch.all(bits > 0)  # Should have positive bit counts

        # Rough check: -log2(0.5) = 1 bit per element, -log2(0.25) = 2 bits per element
        expected_bits_approx = (3 * 8 * 8 * 1) + (3 * 4 * 4 * 2)  # 192 + 96 = 288 bits per image
        assert torch.allclose(bits, torch.tensor([expected_bits_approx] * batch_size, dtype=torch.float), rtol=0.1)

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_forward_fixed_quality_real_implementation(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test forward pass in fixed quality mode with real implementation paths."""
        # Setup mock to return a realistic model
        mock_model = mock_compressai_zoo["bmshj2018_factorized"](quality=5)
        mock_factory.return_value = mock_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=True, collect_stats=True)

        # Test the actual forward implementation
        result = compressor.forward(sample_image_batch)

        assert isinstance(result, tuple)
        assert len(result) == 2
        reconstructed, bits = result

        assert reconstructed.shape == sample_image_batch.shape
        assert bits.shape == (sample_image_batch.shape[0],)
        assert torch.all(bits > 0)

        # Check that stats were collected
        stats = compressor.get_stats()
        assert "total_bits" in stats
        assert "avg_quality" in stats
        assert stats["avg_quality"] == 5
        assert "img_stats" in stats
        assert len(stats["img_stats"]) == sample_image_batch.shape[0]

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_forward_with_compressed_data_collection(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test collection of compressed data in fixed quality mode."""
        mock_model = mock_compressai_zoo["bmshj2018_factorized"](quality=5)
        mock_factory.return_value = mock_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_compressed_data=True, return_bits=False)  # Only compressed data, not bits

        result = compressor.forward(sample_image_batch)

        assert isinstance(result, tuple)
        assert len(result) == 2
        reconstructed, compressed_data = result

        assert reconstructed.shape == sample_image_batch.shape
        assert isinstance(compressed_data, list)
        assert len(compressed_data) == sample_image_batch.shape[0]

        # Check compressed data structure
        for comp_data in compressed_data:
            assert "strings" in comp_data
            assert "shape" in comp_data

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_bit_constrained_mode_real_implementation(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test bit-constrained mode with realistic bit calculations."""

        # Create models with different bit outputs based on quality
        def create_quality_model(**kwargs):
            quality = kwargs.get("quality", 1)
            model = Mock()
            model.eval.return_value = model
            model.to.return_value = model

            def mock_forward(x):
                batch_size = x.shape[0]
                # Higher quality = more bits (less compression)
                likelihood_value = 0.8 - (quality * 0.1)  # Quality 1: 0.7, Quality 8: 0.0
                likelihood_value = max(likelihood_value, 0.1)  # Minimum likelihood

                return {"x_hat": torch.rand_like(x) * 0.1 + x * 0.9, "likelihoods": {"y": torch.ones(batch_size, 3, 8, 8) * likelihood_value, "z": torch.ones(batch_size, 3, 4, 4) * likelihood_value}}

            model.side_effect = mock_forward
            model.compress = lambda x: {"strings": {"y": b"test", "z": b"test"}, "shape": {"y": [8, 8], "z": [4, 4]}}
            return model

        # Mock different qualities
        mock_factory.side_effect = create_quality_model

        # Set a bit budget that should force lower quality
        max_bits = 200  # Should require lower quality

        compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=max_bits, return_bits=True, collect_stats=True)

        result = compressor.forward(sample_image_batch)

        assert isinstance(result, tuple)
        assert len(result) == 2
        reconstructed, bits = result

        assert reconstructed.shape == sample_image_batch.shape
        assert bits.shape == (sample_image_batch.shape[0],)

        # Check that bit constraint was respected where possible
        # Some images might exceed if even lowest quality is too high
        stats = compressor.get_stats()
        assert "avg_quality" in stats
        assert stats["avg_quality"] <= 8  # Should not exceed max quality

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_bit_constrained_with_early_stopping(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test early stopping functionality in bit-constrained mode."""
        mock_model = mock_compressai_zoo["bmshj2018_factorized"](quality=1)  # Low quality, low bits
        mock_factory.return_value = mock_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=1000, early_stopping_threshold=0.9, return_bits=True)  # High bit budget  # Stop at 90% of budget

        result = compressor.forward(sample_image_batch)

        assert isinstance(result, tuple)
        reconstructed, bits = result

        # With early stopping, should use high quality and stop early
        assert torch.all(bits <= 1000)  # Should respect bit budget

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_bit_constrained_with_compressed_data(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test compressed data collection in bit-constrained mode."""
        mock_model = mock_compressai_zoo["bmshj2018_factorized"](quality=5)
        mock_factory.return_value = mock_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=500, return_compressed_data=True, return_bits=True)

        result = compressor.forward(sample_image_batch)

        assert isinstance(result, tuple)
        assert len(result) == 3  # reconstructed, bits, compressed_data
        reconstructed, bits, compressed_data = result

        assert reconstructed.shape == sample_image_batch.shape
        assert bits.shape == (sample_image_batch.shape[0],)
        assert isinstance(compressed_data, list)
        assert len(compressed_data) == sample_image_batch.shape[0]

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_lowest_quality_fallback_with_warning(self, mock_factory, sample_image_batch, mock_compressai_zoo):
        """Test fallback to lowest quality when all qualities exceed bit budget."""

        # Create a model that always produces high bits
        def create_high_bit_model(**kwargs):
            model = Mock()
            model.eval.return_value = model
            model.to.return_value = model

            def mock_forward(x):
                batch_size = x.shape[0]
                return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(batch_size, 3, 8, 8) * 0.01, "z": torch.ones(batch_size, 3, 4, 4) * 0.01}}  # Very low likelihood = high bits

            model.side_effect = mock_forward
            return model

        mock_factory.side_effect = create_high_bit_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=50, collect_stats=True)  # Very low bit budget

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compressor.forward(sample_image_batch)

            # Should get warning about exceeding budget
            assert len(w) > 0
            assert any("exceed max_bits_per_image even at lowest quality" in str(warning.message) for warning in w)

        assert result.shape == sample_image_batch.shape

        # Check stats reflect lowest quality usage
        stats = compressor.get_stats()
        assert stats["avg_quality"] == 1  # Should use lowest quality (1)

    def test_get_bits_per_image_method(self, sample_image_batch):
        """Test the get_bits_per_image convenience method."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            def mock_forward(x):
                return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.5, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.5}}

            mock_model.side_effect = mock_forward
            mock_factory.return_value = mock_model

            # Test with original settings that don't return bits
            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=False, return_compressed_data=False)  # Originally false

            bits = compressor.get_bits_per_image(sample_image_batch)

            assert isinstance(bits, torch.Tensor)
            assert bits.shape == (sample_image_batch.shape[0],)
            assert torch.all(bits > 0)

            # Verify original settings were restored
            assert not compressor.return_bits
            assert not compressor.return_compressed_data

    def test_get_bits_per_image_error_handling(self, sample_image_batch):
        """Test error handling in get_bits_per_image method."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            # Create a model that returns invalid output
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            # Mock forward method to return just a tensor instead of proper dict
            def mock_forward_invalid(x):
                return torch.rand_like(x)  # Returns tensor instead of dict with likelihoods

            mock_model.side_effect = mock_forward_invalid
            mock_factory.return_value = mock_model

            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=False)

            # This should trigger error handling path in get_bits_per_image
            with pytest.raises(TypeError, match="Forward method did not return expected tuple"):
                compressor.get_bits_per_image(sample_image_batch)

    def test_reset_stats_functionality(self, sample_image_batch):
        """Test the reset_stats method."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            def mock_forward(x):
                return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.5, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.5}}

            mock_model.side_effect = mock_forward
            mock_factory.return_value = mock_model

            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, collect_stats=True)

            # Run forward to collect stats
            compressor.forward(sample_image_batch)

            # Verify stats were collected
            stats = compressor.get_stats()
            assert stats["total_bits"] > 0
            assert len(stats["img_stats"]) > 0

            # Reset stats
            compressor.reset_stats()

            # Verify stats were reset
            stats = compressor.get_stats()
            assert stats["total_bits"] == 0
            assert len(stats["img_stats"]) == 0
            assert stats["avg_quality"] == 0

    def test_reset_stats_warning_when_not_collecting(self):
        """Test warning when trying to reset stats without collection enabled."""
        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, collect_stats=False)  # Stats collection disabled

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compressor.reset_stats()

            assert len(w) == 1
            assert "Statistics not collected" in str(w[0].message)

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_device_handling(self, mock_factory, sample_image_batch):
        """Test device handling functionality."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def mock_forward(x):
            return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.5, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.5}}

        mock_model.side_effect = mock_forward
        mock_factory.return_value = mock_model

        # Test explicit device setting
        compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, device="cpu")

        assert compressor.device == "cpu"

        # Test default device detection
        compressor_auto = NeuralCompressor(method="bmshj2018_factorized", quality=5)

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert compressor_auto.device == expected_device

    @patch("compressai.zoo.cheng2020_anchor")
    def test_different_method_and_metric(self, mock_factory, sample_image_batch):
        """Test different compression methods and metrics."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def mock_forward(x):
            return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.5, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.5}}

        mock_model.side_effect = mock_forward
        mock_factory.return_value = mock_model

        # Test different method and metric
        compressor = NeuralCompressor(method="cheng2020_anchor", quality=3, metric="ms-ssim", collect_stats=True)

        result = compressor.forward(sample_image_batch)

        # Verify model was called with correct parameters
        mock_factory.assert_called_with(quality=3, pretrained=True, metric="ms-ssim")

        # Should return just the reconstructed tensor since no return flags are set
        assert result.shape == sample_image_batch.shape

        stats = compressor.get_stats()
        assert stats["model_name"] == "cheng2020_anchor"
        assert stats["metric"] == "ms-ssim"

    @patch("compressai.zoo.bmshj2018_factorized")
    def test_early_break_in_bit_constrained_mode(self, mock_factory, sample_image_batch):
        """Test early break when all remaining images are False."""
        # Create a model that satisfies constraints quickly
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def mock_forward(x):
            return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.9, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.9}}  # High likelihood = low bits

        mock_model.side_effect = mock_forward
        mock_factory.return_value = mock_model

        compressor = NeuralCompressor(method="bmshj2018_factorized", max_bits_per_image=1000, collect_stats=True)  # High budget, should satisfy quickly

        result = compressor.forward(sample_image_batch)

        # Should return just the reconstructed tensor since no return flags are set
        assert result.shape == sample_image_batch.shape

        # Should use high quality since constraint is easily satisfied
        stats = compressor.get_stats()
        assert stats["avg_quality"] >= 6  # Should use high quality


class TestNeuralCompressorEdgeCases:
    """Test edge cases and error conditions."""

    def test_max_bits_warning_in_fixed_quality_mode(self, sample_image_batch):
        """Test warning when fixed quality exceeds max_bits constraint."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            def mock_forward(x):
                return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.01, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.01}}  # Low likelihood = high bits

            mock_model.side_effect = mock_forward
            mock_factory.return_value = mock_model

            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=8, max_bits_per_image=50)  # High quality  # Low budget

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                compressor.forward(sample_image_batch)

                # Should warn about exceeding constraint
                assert len(w) > 0
                assert any("exceed the max_bits_per_image constraint" in str(warning.message) for warning in w)

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            # Mock return for empty batch
            def mock_forward(x):
                return {"x_hat": torch.empty_like(x), "likelihoods": {"y": torch.empty(0, 3, 8, 8), "z": torch.empty(0, 3, 4, 4)}}

            mock_model.side_effect = mock_forward
            mock_factory.return_value = mock_model

            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5)

            # Create empty batch
            empty_batch = torch.empty(0, 3, 32, 32)

            # Should handle gracefully
            result = compressor.forward(empty_batch)
            assert result.shape == empty_batch.shape

    def test_single_image_processing(self, sample_image_batch):
        """Test processing of single images."""
        with patch("compressai.zoo.bmshj2018_factorized") as mock_factory:
            mock_model = Mock()
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model

            def mock_forward(x):
                return {"x_hat": torch.rand_like(x), "likelihoods": {"y": torch.ones(x.shape[0], 3, 8, 8) * 0.5, "z": torch.ones(x.shape[0], 3, 4, 4) * 0.5}}

            mock_model.side_effect = mock_forward
            mock_factory.return_value = mock_model

            compressor = NeuralCompressor(method="bmshj2018_factorized", quality=5, return_bits=True, collect_stats=True)

            single_image = sample_image_batch[0:1]  # Take first image only

            result = compressor.forward(single_image)

            assert isinstance(result, tuple)
            reconstructed, bits = result
            assert reconstructed.shape == single_image.shape
            assert bits.shape == (1,)

            stats = compressor.get_stats()
            assert len(stats["img_stats"]) == 1
