import pytest
import torch
from kaira.metrics.image.ssim import StructuralSimilarityIndexMeasure

@pytest.fixture
def sample_images():
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    return img1, img2

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
