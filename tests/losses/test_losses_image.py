"""Tests for the image losses module with comprehensive coverage."""

import pytest
import torch
import torch.nn as nn

from kaira.losses.image import (
    MSELoss,
    CombinedLoss,
    MSELPIPSLoss,
    LPIPSLoss,
    SSIMLoss,
    MSSSIMLoss,
    L1Loss,
    VGGLoss,
    TotalVariationLoss,
    GradientLoss,
    PSNRLoss,
    FocalLoss,
    StyleLoss,
    ElasticLoss,
)
from kaira.losses.base import BaseLoss


@pytest.fixture
def sample_images():
    """Fixture for creating sample image tensor."""
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def sample_target_images():
    """Fixture for creating sample target image tensor."""
    return torch.rand(2, 3, 32, 32)


def test_mse_loss(sample_images, sample_target_images):
    """Test MSELoss functionality."""
    loss_fn = MSELoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Compare with PyTorch's built-in MSE loss
    torch_mse = nn.MSELoss()(sample_images, sample_target_images)
    assert torch.isclose(loss, torch_mse)


def test_combined_loss(sample_images, sample_target_images):
    """Test CombinedLoss functionality."""
    # Create individual loss functions
    mse_loss = MSELoss()
    l1_loss = L1Loss()
    
    # Create combined loss with custom weights
    combined_loss = CombinedLoss(losses=[mse_loss, l1_loss], weights=[0.7, 0.3])
    loss = combined_loss(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Manually calculate and compare
    mse_val = mse_loss(sample_images, sample_target_images)
    l1_val = l1_loss(sample_images, sample_target_images)
    manual_combined = 0.7 * mse_val + 0.3 * l1_val
    
    assert torch.isclose(loss, manual_combined)


def test_mselpips_loss(sample_images, sample_target_images, monkeypatch):
    """Test MSELPIPSLoss functionality."""
    # Mock the LPIPS loss calculation to return a known value
    class MockLPIPSLoss(BaseLoss):
        def forward(self, x, target):
            return torch.tensor(0.5)
    
    # Apply the mock
    monkeypatch.setattr("kaira.losses.image.LPIPSLoss", MockLPIPSLoss)
    
    # Create the loss with custom weights
    loss_fn = MSELPIPSLoss(mse_weight=0.6, lpips_weight=0.4)
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Manually calculate MSE component
    mse_val = nn.MSELoss()(sample_images, sample_target_images)
    lpips_val = torch.tensor(0.5)
    manual_combined = 0.6 * mse_val + 0.4 * lpips_val
    
    assert torch.isclose(loss, manual_combined)


def test_l1_loss(sample_images, sample_target_images):
    """Test L1Loss functionality."""
    loss_fn = L1Loss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Compare with PyTorch's built-in L1 loss
    torch_l1 = nn.L1Loss()(sample_images, sample_target_images)
    assert torch.isclose(loss, torch_l1)


def test_total_variation_loss(sample_images):
    """Test TotalVariationLoss functionality."""
    loss_fn = TotalVariationLoss()
    loss = loss_fn(sample_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Manual calculation of TV loss
    batch_size = sample_images.size()[0]
    h_diff = sample_images[:, :, 1:, :] - sample_images[:, :, :-1, :]
    w_diff = sample_images[:, :, :, 1:] - sample_images[:, :, :, :-1]
    
    h_tv = torch.pow(h_diff, 2).sum()
    w_tv = torch.pow(w_diff, 2).sum()
    manual_tv = (h_tv + w_tv) / batch_size
    
    assert torch.isclose(loss, manual_tv)


def test_gradient_loss(sample_images, sample_target_images):
    """Test GradientLoss functionality."""
    loss_fn = GradientLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0  # Loss should be non-negative


def test_psnr_loss(sample_images, sample_target_images):
    """Test PSNRLoss functionality."""
    # Test with default max_val=1.0
    loss_fn = PSNRLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # The loss is negative PSNR (to be minimized)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    
    # Manual calculation of PSNR
    mse = nn.MSELoss()(sample_images, sample_target_images)
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    
    assert torch.isclose(loss, -psnr)
    
    # Test with a different max_val
    loss_fn_255 = PSNRLoss(max_val=255.0)
    loss_255 = loss_fn_255(sample_images, sample_target_images)
    psnr_255 = 20 * torch.log10(torch.tensor(255.0) / torch.sqrt(mse))
    
    assert torch.isclose(loss_255, -psnr_255)


def test_focal_loss_binary():
    """Test FocalLoss for binary classification."""
    # Create binary inputs and targets
    inputs = torch.randn(5, 1)
    targets = torch.randint(0, 2, (5,)).float()
    
    # Test with default parameters
    loss_fn = FocalLoss()
    loss = loss_fn(inputs, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0  # Loss should be non-negative
    
    # Test with custom alpha and gamma
    loss_fn_custom = FocalLoss(alpha=0.75, gamma=1.0)
    loss_custom = loss_fn_custom(inputs, targets)
    
    assert isinstance(loss_custom, torch.Tensor)
    assert loss_custom.ndim == 0


def test_focal_loss_multiclass():
    """Test FocalLoss for multi-class classification."""
    # Create multi-class inputs and targets
    inputs = torch.randn(5, 3)  # 3 classes
    targets = torch.randint(0, 3, (5,))  # Class indices: 0, 1, or 2
    
    # Test with default parameters
    loss_fn = FocalLoss()
    loss = loss_fn(inputs, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0  # Loss should be non-negative
    
    # Test with custom alpha (weight per class)
    alpha = torch.tensor([0.5, 0.3, 0.2])
    loss_fn_custom = FocalLoss(alpha=alpha, gamma=2.0)
    loss_custom = loss_fn_custom(inputs, targets)
    
    assert isinstance(loss_custom, torch.Tensor)
    assert loss_custom.ndim == 0


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_focal_loss_reduction_modes(reduction):
    """Test FocalLoss with different reduction modes."""
    inputs = torch.randn(5, 3)
    targets = torch.randint(0, 3, (5,))
    
    loss_fn = FocalLoss(reduction=reduction)
    loss = loss_fn(inputs, targets)
    
    if reduction == "none":
        assert loss.shape == (5,)  # One loss value per sample
    else:  # "mean" or "sum"
        assert loss.ndim == 0  # Scalar tensor


@pytest.mark.parametrize("identical", [True, False])
def test_losses_with_identical_inputs(identical):
    """Test that losses are zero when inputs are identical (if applicable)."""
    # Create a single tensor to use as both input and target
    tensor = torch.rand(2, 3, 32, 32)
    
    if identical:
        x = tensor
        y = tensor
    else:
        x = tensor
        y = torch.rand(2, 3, 32, 32)
    
    # Test various losses
    mse_loss = MSELoss()(x, y)
    l1_loss = L1Loss()(x, y)
    psnr_loss = PSNRLoss()(x, y)
    
    if identical:
        assert torch.isclose(mse_loss, torch.tensor(0.0))
        assert torch.isclose(l1_loss, torch.tensor(0.0))
        assert psnr_loss == float('-inf')  # -PSNR is -∞ when MSE=0


def test_lpips_loss(sample_images, sample_target_images, monkeypatch):
    """Test LPIPSLoss functionality."""
    # Mock the LPIPS metric to return a known value
    class MockLPIPS:
        def __call__(self, x, y):
            return torch.tensor([0.25])  # Returning a 1D tensor to match actual implementation
    
    # Apply the mock
    monkeypatch.setattr("kaira.metrics.LearnedPerceptualImagePatchSimilarity", MockLPIPS)
    
    # Create the loss function
    loss_fn = LPIPSLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a tensor with expected value
    assert isinstance(loss, torch.Tensor)
    # The loss might be returned as a 1D tensor with a single value
    if loss.ndim > 0:
        assert loss.numel() == 1
        loss_value = loss.item()
    else:
        loss_value = loss.item()
    
    assert abs(loss_value - 0.25) < 1e-5


def test_ssim_loss(sample_images, sample_target_images, monkeypatch):
    """Test SSIMLoss functionality."""
    # Mock the SSIM metric to return a known value
    class MockSSIM:
        def __init__(self, **kwargs):
            pass
            
        def __call__(self, x, y):
            # Return a scalar tensor to match expected behavior
            return torch.tensor(0.8, dtype=torch.float32)
    
    # Apply the mock
    monkeypatch.setattr("kaira.metrics.StructuralSimilarityIndexMeasure", MockSSIM)
    
    # Create the loss function
    loss_fn = SSIMLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a tensor
    assert isinstance(loss, torch.Tensor)
    # Handle either scalar or 1D tensor output
    if loss.ndim > 0:
        assert loss.numel() == 1
        loss_value = loss.item()
    else:
        loss_value = loss.item()
    
    # Loss should be 1 - SSIM
    expected_value = 1 - 0.8
    assert abs(loss_value - expected_value) < 1e-5
    
    # Test with different kernel size
    loss_fn_k7 = SSIMLoss(kernel_size=7)
    loss_k7 = loss_fn_k7(sample_images, sample_target_images)
    if loss_k7.ndim > 0:
        assert loss_k7.numel() == 1
        loss_k7_value = loss_k7.item()
    else:
        loss_k7_value = loss_k7.item()
    assert abs(loss_k7_value - expected_value) < 1e-5


def test_ms_ssim_loss(sample_images, sample_target_images, monkeypatch):
    """Test MSSSIMLoss functionality."""
    # Mock the SSIM metric (used by MS-SSIM) to return a known value
    class MockSSIM:
        def __init__(self, **kwargs):
            pass
            
        def __call__(self, x, y):
            # Return a scalar tensor to match expected behavior
            return torch.tensor(0.75, dtype=torch.float32)
    
    # Apply the mock
    monkeypatch.setattr("kaira.metrics.StructuralSimilarityIndexMeasure", MockSSIM)
    
    # Create the loss function
    loss_fn = MSSSIMLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a tensor
    assert isinstance(loss, torch.Tensor)
    # Handle either scalar or 1D tensor output
    if loss.ndim > 0:
        assert loss.numel() == 1
        loss_value = loss.item()
    else:
        loss_value = loss.item()
    
    # Loss should be 1 - MS_SSIM
    expected_value = 1 - 0.75
    assert abs(loss_value - expected_value) < 1e-5


@pytest.mark.parametrize("with_input", [True, False])
def test_vgg_loss(sample_images, sample_target_images, monkeypatch, with_input):
    """Test VGGLoss functionality."""
    # Create a simple mock for the VGG network
    class MockModule:
        def __init__(self):
            self._modules = {"3": None, "8": None, "15": None, "22": None}
        
        def __call__(self, x):
            return x  # Just return the input for testing
            
    class MockVGG:
        def __init__(self, **kwargs):
            self.features = MockModule()

    # Mock the required imports
    monkeypatch.setattr("torchvision.models.vgg16", lambda **kwargs: MockVGG())
    monkeypatch.setattr("torchvision.models.VGG16_Weights.DEFAULT", None)
    
    # Create custom weights for each layer
    layer_weights = {"conv1_2": 0.1, "conv2_2": 0.2, "conv3_3": 0.3, "conv4_3": 0.4}
    
    # Create the loss function
    loss_fn = VGGLoss(layer_weights=layer_weights)
    
    if with_input:
        # Actual test with input/target images
        loss = loss_fn(sample_images, sample_target_images)
        
        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0  # Loss should be non-negative
    else:
        # Just verify initialization works properly
        assert loss_fn.layer_weights == layer_weights


def test_style_loss(sample_images, sample_target_images, monkeypatch):
    """Test StyleLoss functionality."""
    # Create a simple mock for the VGG network
    class MockSequential(nn.Module):
        def __init__(self):
            super().__init__()
            self.children_list = [nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU()]
        
        def children(self):
            return self.children_list
            
        def add_module(self, name, module):
            pass
            
        def __call__(self, x):
            return x  # Just return the input for testing
            
    class MockVGG:
        def __init__(self, **kwargs):
            self.features = MockSequential()
            
    # Mock the required imports
    monkeypatch.setattr("torchvision.models.vgg16", lambda **kwargs: MockVGG())
    monkeypatch.setattr("torchvision.models.VGG16_Weights.DEFAULT", None)
    
    # Test with default parameters
    loss_fn = StyleLoss()
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check that the loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0  # Loss should be non-negative
    
    # Test with apply_gram=False
    loss_fn_no_gram = StyleLoss(apply_gram=False)
    loss_no_gram = loss_fn_no_gram(sample_images, sample_target_images)
    
    assert isinstance(loss_no_gram, torch.Tensor)
    assert loss_no_gram.ndim == 0
    
    # Test with normalize=True
    loss_fn_norm = StyleLoss(normalize=True)
    loss_norm = loss_fn_norm(sample_images, sample_target_images)
    
    assert isinstance(loss_norm, torch.Tensor)
    assert loss_norm.ndim == 0
    
    # Test gram matrix calculation directly
    gram = loss_fn.gram_matrix(sample_images)
    batch_size, channels = sample_images.size(0), sample_images.size(1)
    assert gram.shape == (batch_size, channels, channels)


@pytest.mark.parametrize("alpha,beta,reduction", [
    (0.5, 1.0, "mean"),  # Default parameters
    (0.0, 1.0, "mean"),  # Pure L2 loss
    (1.0, 1.0, "mean"),  # Pure L1 loss
    (0.5, 0.5, "sum"),   # Different beta and sum reduction
    (0.5, 1.0, "none"),  # No reduction
])
def test_elastic_loss(sample_images, sample_target_images, alpha, beta, reduction):
    """Test ElasticLoss functionality with various parameters."""
    loss_fn = ElasticLoss(alpha=alpha, beta=beta, reduction=reduction)
    loss = loss_fn(sample_images, sample_target_images)
    
    # Check loss dimensions based on reduction mode
    if reduction == "none":
        assert loss.shape == sample_images.shape
    else:
        assert loss.ndim == 0
    
    # Test with identical inputs
    if reduction != "none":
        identical_loss = loss_fn(sample_images, sample_images)
        assert torch.isclose(identical_loss, torch.tensor(0.0))
    
    # Test edge cases
    tiny_diff = torch.ones_like(sample_images) * 1e-10
    tiny_loss = loss_fn(sample_images, sample_images + tiny_diff)
    assert tiny_loss.sum() > 0  # Should be positive but small
    
    # Test larger differences
    large_diff = torch.ones_like(sample_images) * 10.0
    large_loss = loss_fn(sample_images, sample_images + large_diff)
    assert large_loss.sum() > tiny_loss.sum()  # Should be significantly larger


def test_elastic_loss_implementation(sample_images, sample_target_images):
    """Test specific implementation details of ElasticLoss."""
    # Create a simple case with known differences
    x = torch.zeros(2, 3, 4, 4)
    target = torch.zeros(2, 3, 4, 4)
    
    # Set specific differences: some below beta, some above
    target[0, 0, 0, 0] = 0.5  # diff = 0.5 (below beta=1.0)
    target[0, 0, 1, 1] = 2.0  # diff = 2.0 (above beta=1.0)
    
    # Test with alpha=0.5, beta=1.0 (default)
    loss_fn = ElasticLoss(alpha=0.5, beta=1.0, reduction="none")
    point_losses = loss_fn(x, target)
    
    # For diff=0.5 (below beta), should use alpha * 0.5 * diff²/beta
    expected_below = 0.5 * 0.5 * 0.5**2 / 1.0
    assert torch.isclose(point_losses[0, 0, 0, 0], torch.tensor(expected_below))
    
    # For diff=2.0 (above beta), should use (1-alpha)*|diff| + alpha*beta/2
    expected_above = (1-0.5)*2.0 + 0.5*1.0/2.0
    assert torch.isclose(point_losses[0, 0, 1, 1], torch.tensor(expected_above))