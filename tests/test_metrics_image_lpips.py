import pytest
import torch
from kaira.metrics.image.lpips import LearnedPerceptualImagePatchSimilarity

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

def test_lpips_initialization_with_custom_net():
    """Test LPIPS initialization with custom net."""
    lpips = LearnedPerceptualImagePatchSimilarity(net='alex')
    assert lpips.net == 'alex'
    
    lpips = LearnedPerceptualImagePatchSimilarity(net='vgg')
    assert lpips.net == 'vgg'
    
    # Test invalid net
    with pytest.raises(ValueError):
        LearnedPerceptualImagePatchSimilarity(net='invalid_net')

def test_lpips_custom_initialization():
    """Test LPIPS custom initialization parameters."""
    # Test with non-default parameters
    lpips = LearnedPerceptualImagePatchSimilarity(
        net='alex',
        pnet_rand=True,  # Use random weights
        pnet_tune=True,  # Fine-tune network
        version='0.0',   # Legacy LPIPS
        lpips=False,     # Use L2 distance instead of LPIPS
        spatial=True,    # Return spatial map
        normalize=False, # Don't normalize by ImageNet mean/std
    )
    
    assert lpips.net == 'alex'
    # Other parameters would be passed to the model, so we can't directly test them
    # But we can verify the model was created successfully
    assert lpips.model is not None
