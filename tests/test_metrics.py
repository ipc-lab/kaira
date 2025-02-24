# tests/test_metrics.py
import pytest
import torch
from kaira.metrics import (
    MultiScaleSSIM,
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor."""
    return torch.randn(1, 3, 32, 32)

@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor."""
    return torch.randn(1, 3, 32, 32)

def test_multiscale_ssim_initialization():
    """Test MultiScaleSSIM initialization."""
    msssim = MultiScaleSSIM()
    assert isinstance(msssim, MultiScaleSSIM)

def test_multiscale_ssim_update(sample_preds, sample_targets):
    """Test MultiScaleSSIM update method."""
    msssim = MultiScaleSSIM()
    msssim.update(sample_preds, sample_targets)
    assert msssim.sum.shape == torch.Size([])
    assert msssim.total.shape == torch.Size([])

def test_multiscale_ssim_forward(sample_preds, sample_targets):
    """Test MultiScaleSSIM forward pass."""
    msssim = MultiScaleSSIM()
    msssim.update(sample_preds, sample_targets)
    value = msssim.compute()
    assert isinstance(value, torch.Tensor)

def test_lpips_initialization():
    """Test LearnedPerceptualImagePatchSimilarity initialization."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    assert isinstance(lpips, LearnedPerceptualImagePatchSimilarity)

def test_lpips_update(sample_preds, sample_targets):
    """Test LearnedPerceptualImagePatchSimilarity update method."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(sample_preds, sample_targets)
    assert lpips.sum_scores.shape == torch.Size([])
    assert lpips.total.shape == torch.Size([])

def test_lpips_compute(sample_preds, sample_targets):
    """Test LearnedPerceptualImagePatchSimilarity compute method."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(sample_preds, sample_targets)
    mean, std = lpips.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

def test_psnr_initialization():
    """Test PeakSignalNoiseRatio initialization."""
    psnr = PeakSignalNoiseRatio()
    assert isinstance(psnr, PeakSignalNoiseRatio)

def test_psnr_compute(sample_preds, sample_targets):
    """Test PeakSignalNoiseRatio compute method."""
    psnr = PeakSignalNoiseRatio()
    psnr.update(sample_preds, sample_targets)
    mean, std = psnr.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

def test_ssim_initialization():
    """Test StructuralSimilarityIndexMeasure initialization."""
    ssim = StructuralSimilarityIndexMeasure()
    assert isinstance(ssim, StructuralSimilarityIndexMeasure)

def test_ssim_compute(sample_preds, sample_targets):
    """Test StructuralSimilarityIndexMeasure compute method."""
    ssim = StructuralSimilarityIndexMeasure()
    ssim.update(sample_preds, sample_targets)
    mean, std = ssim.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

@pytest.mark.parametrize("kernel_size", [7, 11, 15])
def test_multiscale_ssim_different_kernel_sizes(sample_preds, sample_targets, kernel_size):
    """Test MultiScaleSSIM with different kernel sizes."""
    msssim = MultiScaleSSIM(kernel_size=kernel_size)
    msssim.update(sample_preds, sample_targets)
    value = msssim.compute()
    assert isinstance(value, torch.Tensor)

@pytest.mark.parametrize("net_type", ["vgg", "alex", "squeeze"])
def test_lpips_different_net_types(sample_preds, sample_targets, net_type):
    """Test LearnedPerceptualImagePatchSimilarity with different net_type values."""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
    lpips.update(sample_preds, sample_targets)
    mean, std = lpips.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
