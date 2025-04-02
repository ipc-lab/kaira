# tests/metrics/test_image_metrics.py
"""Tests for image metrics including PSNR, SSIM, MS-SSIM and LPIPS."""
import pytest
import torch

from kaira.metrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleSSIM,
    LearnedPerceptualImagePatchSimilarity,
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


# ===== Learned Perceptual Image Patch Similarity (LPIPS) Tests =====

class TestLearnedPerceptualImagePatchSimilarity:
    """Test suite for LPIPS metrics."""

    def test_lpips_initialization(self):
        """Test LPIPS initialization."""
        lpips = LearnedPerceptualImagePatchSimilarity()
        assert isinstance(lpips, LearnedPerceptualImagePatchSimilarity)

    def test_lpips_update(self, sample_preds, sample_targets):
        """Test LPIPS update method."""
        lpips = LearnedPerceptualImagePatchSimilarity()
        lpips.update(sample_preds, sample_targets)
        assert lpips.sum_scores.shape == torch.Size([])
        assert lpips.total.shape == torch.Size([])

    def test_lpips_compute(self, sample_preds, sample_targets):
        """Test LPIPS compute method."""
        lpips = LearnedPerceptualImagePatchSimilarity()
        lpips.update(sample_preds, sample_targets)
        mean, std = lpips.compute()
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)