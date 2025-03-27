import pytest
import torch
from kaira.metrics.image.ssim import StructuralSimilarityIndexMeasure, MultiScaleSSIM

@pytest.fixture
def sample_images():
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    return img1, img2

@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor."""
    return torch.randn(1, 3, 64, 64)

@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor."""
    return torch.randn(1, 3, 64, 64)

def test_ssim_initialization():
    ssim = StructuralSimilarityIndexMeasure()
    assert isinstance(ssim, StructuralSimilarityIndexMeasure)

def test_ssim_forward(sample_images):
    img1, img2 = sample_images
    ssim = StructuralSimilarityIndexMeasure()
    result = ssim(img1, img2)
    assert isinstance(result, torch.Tensor)

def test_ssim_compute_with_stats(sample_images):
    img1, img2 = sample_images
    ssim = StructuralSimilarityIndexMeasure()
    mean, std = ssim.compute_with_stats(img1, img2)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean > 0
    assert std >= 0

def test_ssim_forward_with_reduction(sample_images):
    img1, img2 = sample_images
    ssim = StructuralSimilarityIndexMeasure(reduction="mean")
    result = ssim(img1, img2)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Should be a scalar

def test_ssim_forward_with_sum_reduction(sample_images):
    img1, img2 = sample_images
    ssim = StructuralSimilarityIndexMeasure(reduction="sum")
    result = ssim(img1, img2)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Should be a scalar

def test_ssim_with_different_data_ranges():
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

def test_multiscale_ssim_different_weights():
    """Test MultiScaleSSIM with different weights for scales."""
    # Default weights (all equal)
    ms_ssim_default = MultiScaleSSIM()
    
    # Custom weights
    custom_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    ms_ssim_custom = MultiScaleSSIM(weights=custom_weights)
    
    # Use larger images to allow for multiple scales
    preds = torch.rand(1, 3, 128, 128)
    targets = torch.rand(1, 3, 128, 128)
    
    # Compute both metrics
    result_default = ms_ssim_default(preds, targets)
    result_custom = ms_ssim_custom(preds, targets)
    
    # Results should be different due to weight differences
    assert result_default is not None
    assert result_custom is not None
    # Weights affect the calculation, so results shouldn't be the same
    assert not torch.isclose(result_default, result_custom, atol=1e-3)
