import pytest
import torch
from kaira.metrics.image.lpips import LearnedPerceptualImagePatchSimilarity

@pytest.fixture
def sample_images():
    img1 = torch.rand(1, 3, 256, 256)
    img2 = torch.rand(1, 3, 256, 256)
    return img1, img2

def test_lpips_initialization():
    lpips = LearnedPerceptualImagePatchSimilarity()
    assert isinstance(lpips, LearnedPerceptualImagePatchSimilarity)

def test_lpips_forward(sample_images):
    img1, img2 = sample_images
    lpips = LearnedPerceptualImagePatchSimilarity()
    result = lpips(img1, img2)
    assert isinstance(result, torch.Tensor)

def test_lpips_update(sample_images):
    img1, img2 = sample_images
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(img1, img2)
    assert lpips.total > 0

def test_lpips_compute(sample_images):
    img1, img2 = sample_images
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(img1, img2)
    mean, std = lpips.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

def test_lpips_reset(sample_images):
    img1, img2 = sample_images
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(img1, img2)
    lpips.reset()
    assert lpips.total == 0
