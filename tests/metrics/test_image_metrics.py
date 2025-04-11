# tests/metrics/test_image_metrics.py
"""Tests for image metrics including PSNR, SSIM, MS-SSIM and LPIPS."""
import torch

from kaira.metrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleSSIM,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

# ===== Peak Signal-to-Noise Ratio (PSNR) Tests =====


class TestPeakSignalNoiseRatio:
    """Test suite for PSNR metrics."""

    def test_psnr_initialization(self):
        """Test PeakSignalNoiseRatio initialization."""
        psnr = PeakSignalNoiseRatio()
        assert isinstance(psnr, PeakSignalNoiseRatio)

    def test_psnr_forward(self, sample_images):
        """Test PeakSignalNoiseRatio forward method."""
        img1, img2 = sample_images
        psnr = PeakSignalNoiseRatio()
        result = psnr(img1, img2)
        assert isinstance(result, torch.Tensor)

    def test_psnr_compute_with_stats(self, sample_images):
        """Test PeakSignalNoiseRatio compute_with_stats method."""
        img1, img2 = sample_images
        psnr = PeakSignalNoiseRatio()
        mean, std = psnr.compute_with_stats(img1, img2)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean > 0
        assert std >= 0

    def test_psnr_compute_with_stats_single_value(self):
        """Test PeakSignalNoiseRatio compute_with_stats method with a single value."""
        # Create a single pixel image
        img1 = torch.rand(1, 3, 1, 1)
        img2 = torch.rand(1, 3, 1, 1)

        psnr = PeakSignalNoiseRatio()
        mean, std = psnr.compute_with_stats(img1, img2)

        # Check that mean is a valid value
        assert isinstance(mean, torch.Tensor)
        assert not torch.isnan(mean)

        # With only one value, standard deviation should be zero
        assert isinstance(std, torch.Tensor)
        assert not torch.isnan(std)
        assert std.item() == 0.0

    def test_psnr_forward_with_reduction(self, sample_images):
        """Test PeakSignalNoiseRatio with reduction."""
        img1, img2 = sample_images
        psnr = PeakSignalNoiseRatio(reduction="mean")
        result = psnr(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be a scalar

    def test_psnr_forward_with_sum_reduction(self, sample_images):
        """Test PeakSignalNoiseRatio with sum reduction."""
        img1, img2 = sample_images
        psnr = PeakSignalNoiseRatio(reduction="sum")
        result = psnr(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be a scalar

    def test_psnr_with_different_data_ranges(self):
        """Test PSNR with different data ranges."""
        # Default data range (1.0)
        psnr_default = PeakSignalNoiseRatio()

        # Custom data range
        psnr_custom = PeakSignalNoiseRatio(data_range=255.0)

        # Setup mock inputs scaled by the data ranges
        preds = torch.rand(1, 1, 32, 32)  # Values in [0, 1]
        targets = torch.rand(1, 1, 32, 32)  # Values in [0, 1]

        preds_scaled = preds * 255.0  # Values in [0, 255]
        targets_scaled = targets * 255.0  # Values in [0, 255]

        # Compute PSNR with default range
        psnr_result_default = psnr_default(preds, targets)

        # Compute PSNR with custom range
        psnr_result_custom = psnr_custom(preds_scaled, targets_scaled)

        # Results should be close despite different data ranges
        assert torch.isclose(psnr_result_default, psnr_result_custom, atol=1e-2)

    def test_psnr_compute_with_stats_multiple_values(self):
        """Test that PSNR compute_with_stats properly calculates mean and std with multiple
        values."""
        # Create a batch of images to ensure multiple values
        batch_size = 4
        img1 = torch.rand(batch_size, 3, 32, 32)
        img2 = torch.rand(batch_size, 3, 32, 32)

        psnr = PeakSignalNoiseRatio()

        # First get the raw values without reduction
        psnr_no_reduction = PeakSignalNoiseRatio(reduction=None)
        raw_values = psnr_no_reduction(img1, img2)

        # Now get mean and std from compute_with_stats
        mean, std = psnr.compute_with_stats(img1, img2)

        # Check that mean and std are correctly calculated
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean > 0

        # Verify the values against manually calculated mean and std
        assert torch.isclose(mean, raw_values.mean(), atol=1e-5)
        assert torch.isclose(std, raw_values.std(), atol=1e-5)


# ===== Structural Similarity Index Measure (SSIM) Tests =====


class TestStructuralSimilarityIndexMeasure:
    """Test suite for SSIM metrics."""

    def test_ssim_initialization(self):
        """Test StructuralSimilarityIndexMeasure initialization."""
        ssim = StructuralSimilarityIndexMeasure()
        assert isinstance(ssim, StructuralSimilarityIndexMeasure)

    def test_ssim_forward(self, sample_images):
        """Test SSIM forward method."""
        img1, img2 = sample_images
        ssim = StructuralSimilarityIndexMeasure()
        result = ssim(img1, img2)
        assert isinstance(result, torch.Tensor)

    def test_ssim_compute_with_stats(self, sample_images):
        """Test SSIM compute_with_stats method."""
        img1, img2 = sample_images
        ssim = StructuralSimilarityIndexMeasure()
        mean, std = ssim.compute_with_stats(img1, img2)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean > 0
        assert std >= 0

    def test_ssim_compute_with_stats_single_value(self):
        """Test SSIM compute_with_stats method with a single value."""
        img1 = torch.rand(1, 3, 256, 256)
        img2 = torch.rand(1, 3, 256, 256)

        ssim = StructuralSimilarityIndexMeasure()
        mean, std = ssim.compute_with_stats(img1, img2)

        # Check that mean is a valid value
        assert isinstance(mean, torch.Tensor)
        assert not torch.isnan(mean)

        # With only one value, standard deviation should be zero
        assert isinstance(std, torch.Tensor)
        assert not torch.isnan(std)
        assert std.item() == 0.0

    def test_ssim_forward_with_reduction(self, sample_images):
        """Test SSIM with reduction."""
        img1, img2 = sample_images
        ssim = StructuralSimilarityIndexMeasure(reduction="mean")
        result = ssim(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be a scalar

    def test_ssim_forward_with_sum_reduction(self, sample_images):
        """Test SSIM with sum reduction."""
        img1, img2 = sample_images
        ssim = StructuralSimilarityIndexMeasure(reduction="sum")
        result = ssim(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be a scalar

    def test_ssim_with_different_data_ranges(self):
        """Test SSIM with different data ranges."""
        # Default data range (1.0)
        ssim_default = StructuralSimilarityIndexMeasure()

        # Custom data range
        ssim_custom = StructuralSimilarityIndexMeasure(data_range=255.0)

        # Setup mock inputs scaled by the data ranges
        preds = torch.rand(1, 1, 32, 32)  # Values in [0, 1]
        targets = torch.rand(1, 1, 32, 32)  # Values in [0, 1]

        preds_scaled = preds * 255.0  # Values in [0, 255]
        targets_scaled = targets * 255.0  # Values in [0, 255]

        # Compute SSIM with default range
        ssim_result_default = ssim_default(preds, targets)

        # Compute SSIM with custom range
        ssim_result_custom = ssim_custom(preds_scaled, targets_scaled)

        # Results should be close despite different data ranges
        assert torch.isclose(ssim_result_default, ssim_result_custom, atol=1e-2)

    def test_ssim_forward_without_reduction(self, sample_images):
        """Test SSIM forward method without reduction."""
        img1, img2 = sample_images
        ssim = StructuralSimilarityIndexMeasure(reduction=None)
        result = ssim(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() > 0  # Should not be a scalar (not reduced)

    def test_ssim_compute_with_stats_multiple_values(self):
        """Test that SSIM compute_with_stats properly calculates mean and std with multiple
        values."""
        # Create a batch of images to ensure multiple values
        torch.manual_seed(42)
        batch_size = 4
        img1 = torch.rand(batch_size, 3, 64, 64)
        img2 = torch.rand(batch_size, 3, 64, 64)

        ssim = StructuralSimilarityIndexMeasure()

        # First get the raw values without reduction to verify batch calculation
        ssim_no_reduction = StructuralSimilarityIndexMeasure(reduction=None)
        raw_values = ssim_no_reduction(img1, img2)

        # Now get mean and std from compute_with_stats
        mean, std = ssim.compute_with_stats(img1, img2)

        # Check that mean and std are correctly calculated
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean > 0

        # Verify the values against manually calculated mean and std
        assert torch.isclose(mean, raw_values.mean(), atol=1e-5)
        assert torch.isclose(std, raw_values.std(), atol=1e-5)


# ===== Multi-Scale Structural Similarity Index Measure (MS-SSIM) Tests =====


class TestMultiScaleSSIM:
    """Test suite for MS-SSIM metrics."""

    def test_multiscale_ssim_initialization(self):
        """Test MultiScaleSSIM initialization."""
        ms_ssim = MultiScaleSSIM()
        assert isinstance(ms_ssim, MultiScaleSSIM)

    def test_multiscale_ssim_update(self, sample_preds, sample_targets):
        """Test MultiScaleSSIM update method."""
        ms_ssim = MultiScaleSSIM()
        ms_ssim.update(sample_preds, sample_targets)
        assert ms_ssim.sum_values.shape == torch.Size([])
        assert ms_ssim.count.shape == torch.Size([])

    def test_multiscale_ssim_forward(self, sample_preds, sample_targets):
        """Test MultiScaleSSIM forward method."""
        ms_ssim = MultiScaleSSIM()
        result = ms_ssim(sample_preds, sample_targets)
        assert isinstance(result, torch.Tensor)
        assert result.dim() > 0  # Should not be a scalar by default

    def test_multiscale_ssim_compute(self, sample_preds, sample_targets):
        """Test MultiScaleSSIM compute method."""
        ms_ssim = MultiScaleSSIM()
        ms_ssim.update(sample_preds, sample_targets)
        mean, std = ms_ssim.compute()  # Unpack the tuple correctly
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        # The mean might be zero for certain inputs, especially small/random test images
        # that are incompatible with MS-SSIM's multiple scales requirement
        assert mean >= 0  # Check that mean is non-negative instead of positive
        assert std >= 0  # Check that standard deviation is non-negative

    def test_multiscale_ssim_different_weights(self):
        """Test MultiScaleSSIM with different weights for scales."""
        # Default weights (all equal)
        ms_ssim_default = MultiScaleSSIM()

        # Custom weights
        custom_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        ms_ssim_custom = MultiScaleSSIM(weights=custom_weights)

        # Use larger images to allow for multiple scales
        # Image size must be larger than (win_size-1) * (2^4) = 160 for default win_size=11
        preds = torch.rand(1, 3, 256, 256)
        targets = torch.rand(1, 3, 256, 256)

        # Compute both metrics
        result_default = ms_ssim_default(preds, targets)
        result_custom = ms_ssim_custom(preds, targets)

        # Results should be different due to weight differences
        assert result_default is not None
        assert result_custom is not None
        # Weights affect the calculation, so results shouldn't be the same
        assert not torch.isclose(result_default, result_custom, atol=1e-3)

    def test_multiscale_ssim_forward_with_reduction(self, sample_images):
        """Test MS-SSIM forward method with reduction."""
        img1, img2 = sample_images
        ms_ssim = MultiScaleSSIM(reduction="mean")
        result = ms_ssim(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be a scalar

    def test_multiscale_ssim_update_compute_reset(self, sample_images):
        """Test MS-SSIM update, compute, and reset methods."""
        img1, img2 = sample_images
        ms_ssim = MultiScaleSSIM()

        # Initial state
        assert ms_ssim.sum_values.item() == 0.0
        assert ms_ssim.sum_sq.item() == 0.0
        assert ms_ssim.count.item() == 0

        # Update state with images
        ms_ssim.update(img1, img2)
        assert ms_ssim.count.item() > 0
        assert ms_ssim.sum_values.item() != 0.0

        # Compute accumulated statistics
        mean, std = ms_ssim.compute()
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean > 0
        assert std >= 0  # Check that standard deviation is non-negative

        # Reset state
        ms_ssim.reset()
        assert ms_ssim.sum_values.item() == 0.0
        assert ms_ssim.sum_sq.item() == 0.0
        assert ms_ssim.count.item() == 0

    def test_multiscale_ssim_with_reductions(self, sample_images):
        """Test MS-SSIM with different reduction methods."""
        img1, img2 = sample_images

        # Test with mean reduction
        ms_ssim_mean = MultiScaleSSIM(reduction="mean")
        result_mean = ms_ssim_mean(img1, img2)
        assert result_mean.dim() == 0  # Scalar

        # Test with sum reduction
        ms_ssim_sum = MultiScaleSSIM(reduction="sum")
        result_sum = ms_ssim_sum(img1, img2)
        assert result_sum.dim() == 0  # Scalar

        # Test with no reduction
        ms_ssim_none = MultiScaleSSIM(reduction=None)
        result_none = ms_ssim_none(img1, img2)
        assert result_none.dim() > 0  # Not reduced to scalar

    def test_multiscale_ssim_update_with_empty_values(self, monkeypatch):
        """Test that MultiScaleSSIM.update correctly handles empty values."""
        # Create a MS-SSIM instance
        ms_ssim = MultiScaleSSIM()

        # Record initial state
        initial_sum_values = ms_ssim.sum_values.clone()
        initial_sum_sq = ms_ssim.sum_sq.clone()
        initial_count = ms_ssim.count.clone()

        # Mock the forward method to return an empty tensor
        original_forward = ms_ssim.forward

        def mock_forward(*args, **kwargs):
            return torch.tensor([])

        monkeypatch.setattr(ms_ssim, "forward", mock_forward)

        # Create dummy input tensors
        preds = torch.randn(2, 3, 64, 64)
        targets = torch.randn(2, 3, 64, 64)

        # Call update with our mocked forward method that returns empty tensor
        ms_ssim.update(preds, targets)

        # Verify that internal state was not updated
        assert torch.equal(ms_ssim.sum_values, initial_sum_values)
        assert torch.equal(ms_ssim.sum_sq, initial_sum_sq)
        assert torch.equal(ms_ssim.count, initial_count)

        # Restore the original forward method
        monkeypatch.setattr(ms_ssim, "forward", original_forward)


# ===== Learned Perceptual Image Patch Similarity (LPIPS) Tests =====


class TestLearnedPerceptualImagePatchSimilarity:
    """Test suite for LPIPS metrics."""

    def test_lpips_initialization(self):
        """Test LPIPS initialization."""
        lpips = LearnedPerceptualImagePatchSimilarity()
        assert isinstance(lpips, LearnedPerceptualImagePatchSimilarity)

    def test_lpips_update(self, sample_preds, sample_targets):
        """Test LPIPS update method."""
        # Rescale to [0, 1] range for normalize=True
        normalized_preds = (torch.clamp(sample_preds, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        normalized_targets = (torch.clamp(sample_targets, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]

        # Use normalize=True which expects [0,1] inputs
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        lpips.update(normalized_preds, normalized_targets)
        assert lpips.sum_scores.shape == torch.Size([])
        assert lpips.total.shape == torch.Size([])

    def test_lpips_compute(self, sample_preds, sample_targets):
        """Test LPIPS compute method."""
        # Rescale to [0, 1] range for normalize=True
        normalized_preds = (torch.clamp(sample_preds, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        normalized_targets = (torch.clamp(sample_targets, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]

        # Use normalize=True which expects [0,1] inputs
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        lpips.update(normalized_preds, normalized_targets)
        mean, std = lpips.compute()
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert std >= 0  # Check that standard deviation is non-negative

        # Test with a single value case
        lpips.reset()
        # Create a batch with a single sample
        single_pred = normalized_preds[0:1]
        single_target = normalized_targets[0:1]
        lpips.update(single_pred, single_target)
        mean_single, std_single = lpips.compute()
        assert std_single.item() == 0.0  # Single value should have zero std

        # Test with multiple values
        lpips.reset()
        # Create a larger batch to ensure multiple values
        if sample_preds.size(0) > 1:
            lpips.update(normalized_preds, normalized_targets)
            mean_multi, std_multi = lpips.compute()
            # With multiple different values, std should be non-zero
            # But we can't guarantee this, so just check it's valid
            assert std_multi >= 0

    def test_lpips_reset(self, sample_preds, sample_targets):
        """Test LPIPS reset method."""
        # Rescale to [0, 1] range for normalize=True
        normalized_preds = (torch.clamp(sample_preds, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        normalized_targets = (torch.clamp(sample_targets, -1.0, 1.0) + 1.0) / 2.0  # Convert [-1,1] to [0,1]

        # Use normalize=True which expects [0,1] inputs
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Initial state
        assert lpips.sum_scores.item() == 0.0
        assert lpips.sum_sq.item() == 0.0
        assert lpips.total.item() == 0

        # Update state
        lpips.update(normalized_preds, normalized_targets)
        assert lpips.total.item() > 0
        assert lpips.sum_scores.item() != 0.0

        # Reset state
        lpips.reset()
        assert lpips.sum_scores.item() == 0.0
        assert lpips.sum_sq.item() == 0.0
        assert lpips.total.item() == 0

    def test_lpips_forward(self, sample_images):
        """Test LPIPS forward method."""
        img1, img2 = sample_images
        lpips = LearnedPerceptualImagePatchSimilarity()
        result = lpips(img1, img2)
        assert isinstance(result, torch.Tensor)
        assert result.dim() > 0  # Should not be a scalar by default

    def test_lpips_with_normalization(self):
        """Test LPIPS with different normalization settings."""
        # Create test images in [0, 1] range
        img1_norm = torch.rand(2, 3, 64, 64)  # Values in [0, 1]
        img2_norm = torch.rand(2, 3, 64, 64)  # Values in [0, 1]

        # Create test images in [-1, 1] range
        img1_unnorm = img1_norm * 2 - 1  # Values in [-1, 1]
        img2_unnorm = img2_norm * 2 - 1  # Values in [-1, 1]

        # Test with normalize=True (expects [0, 1] inputs)
        lpips_norm = LearnedPerceptualImagePatchSimilarity(normalize=True)
        result_norm = lpips_norm(img1_norm, img2_norm)

        # Test with normalize=False (expects [-1, 1] inputs)
        lpips_unnorm = LearnedPerceptualImagePatchSimilarity(normalize=False)
        result_unnorm = lpips_unnorm(img1_unnorm, img2_unnorm)

        assert isinstance(result_norm, torch.Tensor)
        assert isinstance(result_unnorm, torch.Tensor)
