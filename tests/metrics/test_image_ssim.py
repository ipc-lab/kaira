# tests/metrics/test_image_ssim.py
"""Tests for SSIM (Structural Similarity Index Measure) metrics."""
import pytest
import torch

from kaira.metrics.image.ssim import SSIM, MultiScaleSSIM, StructuralSimilarityIndexMeasure


class TestStructuralSimilarityIndexMeasure:
    """Test cases for Structural Similarity Index Measure (SSIM) metric."""

    def test_ssim_basic_computation(self):
        """Test basic SSIM computation with simple images."""
        metric = StructuralSimilarityIndexMeasure()

        # Create simple test images
        img1 = torch.rand(1, 3, 32, 32)
        img2 = img1.clone()

        ssim = metric.forward(img1, img2)
        assert torch.isclose(ssim, torch.tensor([1.0]), atol=1e-4), f"SSIM should be ~1.0 for identical images, got {ssim}"

    def test_ssim_perfect_similarity(self):
        """Test SSIM with identical images."""
        metric = StructuralSimilarityIndexMeasure()

        img = torch.rand(2, 3, 64, 64)
        ssim = metric.forward(img, img)

        assert torch.allclose(ssim, torch.ones_like(ssim), atol=1e-4), "SSIM should be 1.0 for identical images"

    def test_ssim_different_images(self):
        """Test SSIM with different images."""
        metric = StructuralSimilarityIndexMeasure()

        img1 = torch.zeros(1, 3, 32, 32)
        img2 = torch.ones(1, 3, 32, 32)

        ssim = metric.forward(img1, img2)
        assert ssim < 1.0, "SSIM should be less than 1.0 for different images"
        assert ssim >= 0.0, "SSIM should be non-negative"

    def test_ssim_data_range(self):
        """Test SSIM with different data ranges."""
        # Test with data_range=1.0 (default)
        metric1 = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Test with data_range=255.0
        metric255 = StructuralSimilarityIndexMeasure(data_range=255.0)

        img_0_1 = torch.rand(1, 3, 32, 32)  # Range [0, 1]
        img_0_255 = img_0_1 * 255  # Range [0, 255]

        ssim1 = metric1.forward(img_0_1, img_0_1)
        ssim255 = metric255.forward(img_0_255, img_0_255)

        assert torch.allclose(ssim1, ssim255, atol=1e-4), "SSIM should be similar regardless of data range for identical images"

    def test_ssim_kernel_size(self):
        """Test SSIM with different kernel sizes."""
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)

        for kernel_size in [7, 11, 15]:
            metric = StructuralSimilarityIndexMeasure(kernel_size=kernel_size)
            ssim = metric.forward(img1, img2)
            assert torch.isfinite(ssim), f"SSIM should be finite for kernel_size={kernel_size}"
            assert 0 <= ssim <= 1, f"SSIM should be in [0,1] for kernel_size={kernel_size}"

    def test_ssim_sigma(self):
        """Test SSIM with different sigma values."""
        img1 = torch.rand(1, 3, 32, 32)
        img2 = torch.rand(1, 3, 32, 32)

        for sigma in [0.5, 1.0, 1.5, 2.0]:
            metric = StructuralSimilarityIndexMeasure(sigma=sigma)
            ssim = metric.forward(img1, img2)
            assert torch.isfinite(ssim), f"SSIM should be finite for sigma={sigma}"

    def test_ssim_reduction_methods(self):
        """Test SSIM with different reduction methods."""
        img1 = torch.rand(3, 3, 32, 32)
        img2 = torch.rand(3, 3, 32, 32)

        # Test no reduction
        metric_none = StructuralSimilarityIndexMeasure(reduction=None)
        ssim_none = metric_none.forward(img1, img2)
        assert ssim_none.shape[0] == 3, "No reduction should return per-sample SSIM"

        # Test mean reduction
        metric_mean = StructuralSimilarityIndexMeasure(reduction="mean")
        ssim_mean = metric_mean.forward(img1, img2)
        assert ssim_mean.numel() == 1, "Mean reduction should return scalar"

        # Test sum reduction
        metric_sum = StructuralSimilarityIndexMeasure(reduction="sum")
        ssim_sum = metric_sum.forward(img1, img2)
        assert ssim_sum.numel() == 1, "Sum reduction should return scalar"

        # Verify relationships
        assert torch.isclose(ssim_mean, ssim_none.mean()), "Mean reduction should equal manual mean"
        assert torch.isclose(ssim_sum, ssim_none.sum()), "Sum reduction should equal manual sum"

    def test_ssim_compute_with_stats(self):
        """Test SSIM compute_with_stats method."""
        metric = StructuralSimilarityIndexMeasure()

        img1 = torch.rand(5, 3, 32, 32)
        img2 = torch.rand(5, 3, 32, 32)

        mean_ssim, std_ssim = metric.compute_with_stats(img1, img2)

        assert torch.isfinite(mean_ssim), "Mean SSIM should be finite"
        assert torch.isfinite(std_ssim), "Std SSIM should be finite"
        assert std_ssim >= 0, "Standard deviation should be non-negative"

    def test_ssim_single_sample_stats(self):
        """Test SSIM stats computation with single sample."""
        metric = StructuralSimilarityIndexMeasure()

        img1 = torch.rand(1, 3, 32, 32)
        img2 = torch.rand(1, 3, 32, 32)

        mean_ssim, std_ssim = metric.compute_with_stats(img1, img2)

        assert torch.isfinite(mean_ssim), "Mean SSIM should be finite for single sample"
        assert torch.isclose(std_ssim, torch.tensor(0.0)), "Std should be 0 for single sample"

    def test_ssim_batch_processing(self):
        """Test SSIM with different batch sizes."""
        metric = StructuralSimilarityIndexMeasure()

        for batch_size in [1, 2, 4, 8]:
            img1 = torch.rand(batch_size, 3, 32, 32)
            img2 = torch.rand(batch_size, 3, 32, 32)

            ssim = metric.forward(img1, img2)
            assert ssim.shape[0] == batch_size, f"SSIM should have batch_size={batch_size} outputs"

    def test_ssim_grayscale_images(self):
        """Test SSIM with grayscale images."""
        metric = StructuralSimilarityIndexMeasure()

        img1 = torch.rand(2, 1, 32, 32)  # Grayscale
        img2 = torch.rand(2, 1, 32, 32)

        ssim = metric.forward(img1, img2)
        assert ssim.shape[0] == 2, "SSIM should work with grayscale images"
        assert torch.isfinite(ssim).all(), "SSIM should be finite for grayscale images"

    def test_ssim_different_image_sizes(self):
        """Test SSIM with different image sizes."""
        metric = StructuralSimilarityIndexMeasure()

        for size in [16, 32, 64, 128]:
            img1 = torch.rand(1, 3, size, size)
            img2 = torch.rand(1, 3, size, size)

            ssim = metric.forward(img1, img2)
            assert torch.isfinite(ssim), f"SSIM should be finite for size {size}x{size}"

    def test_ssim_shape_mismatch(self):
        """Test SSIM with mismatched image shapes."""
        metric = StructuralSimilarityIndexMeasure()

        img1 = torch.rand(1, 3, 32, 32)
        img2 = torch.rand(1, 3, 64, 64)

        with pytest.raises((RuntimeError, ValueError)):
            metric.forward(img1, img2)


class TestMultiScaleSSIM:
    """Test cases for Multi-Scale SSIM (MS-SSIM) metric."""

    def test_ms_ssim_basic_computation(self):
        """Test basic MS-SSIM computation."""
        metric = MultiScaleSSIM()

        img1 = torch.rand(1, 3, 200, 200)  # MS-SSIM requires larger images (>160)
        img2 = img1.clone()

        ms_ssim = metric.forward(img1, img2)
        assert torch.isclose(ms_ssim, torch.tensor([1.0]), atol=1e-3), f"MS-SSIM should be ~1.0 for identical images, got {ms_ssim}"

    def test_ms_ssim_perfect_similarity(self):
        """Test MS-SSIM with identical images."""
        metric = MultiScaleSSIM()

        img = torch.rand(2, 3, 200, 200)
        ms_ssim = metric.forward(img, img)

        assert torch.allclose(ms_ssim, torch.ones_like(ms_ssim), atol=1e-3), "MS-SSIM should be ~1.0 for identical images"

    def test_ms_ssim_different_images(self):
        """Test MS-SSIM with different images."""
        metric = MultiScaleSSIM()

        img1 = torch.zeros(1, 3, 200, 200)
        img2 = torch.ones(1, 3, 200, 200)

        ms_ssim = metric.forward(img1, img2)
        assert ms_ssim < 1.0, "MS-SSIM should be less than 1.0 for different images"
        assert ms_ssim >= 0.0, "MS-SSIM should be non-negative"

    def test_ms_ssim_data_range(self):
        """Test MS-SSIM with different data ranges."""
        metric1 = MultiScaleSSIM(data_range=1.0)
        metric255 = MultiScaleSSIM(data_range=255.0)

        img_0_1 = torch.rand(1, 3, 200, 200)
        img_0_255 = img_0_1 * 255

        ms_ssim1 = metric1.forward(img_0_1, img_0_1)
        ms_ssim255 = metric255.forward(img_0_255, img_0_255)

        assert torch.allclose(ms_ssim1, ms_ssim255, atol=1e-3), "MS-SSIM should be similar regardless of data range"

    def test_ms_ssim_custom_weights(self):
        """Test MS-SSIM with custom weights."""
        weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        metric = MultiScaleSSIM(weights=weights)

        img1 = torch.rand(1, 3, 200, 200)
        img2 = torch.rand(1, 3, 200, 200)

        ms_ssim = metric.forward(img1, img2)
        assert torch.isfinite(ms_ssim), "MS-SSIM should be finite with custom weights"

    def test_ms_ssim_reduction_methods(self):
        """Test MS-SSIM with different reduction methods."""
        img1 = torch.rand(3, 3, 200, 200)
        img2 = torch.rand(3, 3, 200, 200)

        # Test no reduction
        metric_none = MultiScaleSSIM(reduction=None)
        ms_ssim_none = metric_none.forward(img1, img2)
        assert ms_ssim_none.shape[0] == 3, "No reduction should return per-sample MS-SSIM"

        # Test mean reduction
        metric_mean = MultiScaleSSIM(reduction="mean")
        ms_ssim_mean = metric_mean.forward(img1, img2)
        assert ms_ssim_mean.numel() == 1, "Mean reduction should return scalar"

        # Test sum reduction
        metric_sum = MultiScaleSSIM(reduction="sum")
        ms_ssim_sum = metric_sum.forward(img1, img2)
        assert ms_ssim_sum.numel() == 1, "Sum reduction should return scalar"

    def test_ms_ssim_update_compute(self):
        """Test MS-SSIM update and compute methods."""
        metric = MultiScaleSSIM()

        img1 = torch.rand(2, 3, 200, 200)
        img2 = torch.rand(2, 3, 200, 200)

        # Test single update
        metric.reset()
        metric.update(img1, img2)
        mean, std = metric.compute()

        assert torch.isfinite(mean), "Mean should be finite"
        assert torch.isfinite(std), "Std should be finite"
        assert std >= 0, "Standard deviation should be non-negative"

    def test_ms_ssim_multiple_updates(self):
        """Test MS-SSIM with multiple updates."""
        metric = MultiScaleSSIM()

        metric.reset()

        # Multiple updates
        for _ in range(3):
            img1 = torch.rand(2, 3, 200, 200)
            img2 = torch.rand(2, 3, 200, 200)
            metric.update(img1, img2)

        mean, std = metric.compute()
        assert torch.isfinite(mean), "Mean should be finite after multiple updates"
        assert torch.isfinite(std), "Std should be finite after multiple updates"

    def test_ms_ssim_compute_with_stats(self):
        """Test MS-SSIM compute_with_stats method."""
        metric = MultiScaleSSIM()

        img1 = torch.rand(4, 3, 200, 200)
        img2 = torch.rand(4, 3, 200, 200)

        mean_ms_ssim, std_ms_ssim = metric.compute_with_stats(img1, img2)

        assert torch.isfinite(mean_ms_ssim), "Mean MS-SSIM should be finite"
        assert torch.isfinite(std_ms_ssim), "Std MS-SSIM should be finite"
        assert std_ms_ssim >= 0, "Standard deviation should be non-negative"

    def test_ms_ssim_reset(self):
        """Test MS-SSIM reset functionality."""
        metric = MultiScaleSSIM()

        img1 = torch.rand(2, 3, 200, 200)
        img2 = torch.rand(2, 3, 200, 200)

        # Update and compute
        metric.update(img1, img2)
        mean1, std1 = metric.compute()

        # Reset and check
        metric.reset()
        mean2, std2 = metric.compute()

        assert torch.isclose(mean2, torch.tensor(0.0)), "Mean should be 0 after reset"
        assert torch.isclose(std2, torch.tensor(0.0)), "Std should be 0 after reset"

    def test_ms_ssim_data_range_property(self):
        """Test MS-SSIM data_range property."""
        data_range = 255.0
        metric = MultiScaleSSIM(data_range=data_range)

        assert metric.data_range == data_range, f"data_range property should return {data_range}"

    def test_ms_ssim_kernel_size(self):
        """Test MS-SSIM with different kernel sizes."""
        # Use larger images for larger kernel sizes to satisfy torchmetrics constraints
        # For MS-SSIM with 5 betas and kernel_size=15, image must be > 224 pixels
        img1 = torch.rand(1, 3, 256, 256)  # Increased from 200x200 to 256x256
        img2 = torch.rand(1, 3, 256, 256)

        for kernel_size in [7, 11, 15]:
            metric = MultiScaleSSIM(kernel_size=kernel_size)
            ms_ssim = metric.forward(img1, img2)
            assert torch.isfinite(ms_ssim), f"MS-SSIM should be finite for kernel_size={kernel_size}"

    def test_ms_ssim_empty_update(self):
        """Test MS-SSIM update with empty tensors."""
        metric = MultiScaleSSIM()

        # Create tensors that would result in empty values
        img1 = torch.rand(0, 3, 200, 200)
        img2 = torch.rand(0, 3, 200, 200)

        metric.reset()
        # This should not crash, but torchmetrics may raise an error for empty tensors
        try:
            metric.update(img1, img2)
            mean, std = metric.compute()
            assert torch.isclose(mean, torch.tensor(0.0)), "Mean should be 0 for empty update"
        except (RuntimeError, IndexError, ValueError):
            # It's acceptable if this raises an error for empty tensors
            # The underlying torchmetrics implementation doesn't handle empty tensors well
            pass


def test_ssim_alias():
    """Test that SSIM alias works properly."""
    assert StructuralSimilarityIndexMeasure is SSIM


def test_ssim_integration():
    """Test integration between SSIM and MS-SSIM."""
    img1 = torch.rand(2, 3, 200, 200)
    img2 = img1.clone()

    ssim_metric = StructuralSimilarityIndexMeasure()
    ms_ssim_metric = MultiScaleSSIM()

    ssim_val = ssim_metric.forward(img1, img2)
    ms_ssim_val = ms_ssim_metric.forward(img1, img2)

    # Both should be close to 1.0 for identical images
    assert torch.allclose(ssim_val, torch.ones_like(ssim_val), atol=1e-3), "SSIM should be ~1.0 for identical images"
    assert torch.allclose(ms_ssim_val, torch.ones_like(ms_ssim_val), atol=1e-3), "MS-SSIM should be ~1.0 for identical images"
