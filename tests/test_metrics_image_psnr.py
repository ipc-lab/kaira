import pytest
import torch
from kaira.metrics.image.psnr import PeakSignalNoiseRatio

@pytest.fixture
def sample_images():
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    return img1, img2

def test_psnr_initialization():
    psnr = PeakSignalNoiseRatio()
    assert isinstance(psnr, PeakSignalNoiseRatio)

def test_psnr_forward(sample_images):
    img1, img2 = sample_images
    psnr = PeakSignalNoiseRatio()
    result = psnr(img1, img2)
    assert isinstance(result, torch.Tensor)

def test_psnr_compute_with_stats(sample_images):
    img1, img2 = sample_images
    psnr = PeakSignalNoiseRatio()
    mean, std = psnr.compute_with_stats(img1, img2)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean > 0
    assert std >= 0

def test_psnr_forward_with_reduction(sample_images):
    img1, img2 = sample_images
    psnr = PeakSignalNoiseRatio(reduction="mean")
    result = psnr(img1, img2)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Should be a scalar

def test_psnr_forward_with_sum_reduction(sample_images):
    img1, img2 = sample_images
    psnr = PeakSignalNoiseRatio(reduction="sum")
    result = psnr(img1, img2)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Should be a scalar
