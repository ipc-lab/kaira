import pytest
import torch

from kaira.losses.image import (
    CombinedLoss,
    ElasticLoss,
    FocalLoss,
    GradientLoss,
    L1Loss,
    LPIPSLoss,
    MSELoss,
    MSELPIPSLoss,
    MSSSIMLoss,
    PSNRLoss,
    SSIMLoss,
    StyleLoss,
    TotalVariationLoss,
    VGGLoss,
)


@pytest.fixture
def sample_preds():
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def sample_targets():
    return torch.randn(1, 3, 32, 32)


def test_mse_loss_forward(sample_preds, sample_targets):
    mse_loss = MSELoss()
    loss = mse_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_combined_loss_forward(sample_preds, sample_targets):
    loss1 = MSELoss()
    loss2 = LPIPSLoss()
    weights = [0.5, 0.5]
    combined_loss = CombinedLoss(losses=[loss1, loss2], weights=weights)
    loss = combined_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_mselpip_loss_forward(sample_preds, sample_targets):
    mselpip_loss = MSELPIPSLoss(lpips_weight=0.5)
    loss = mselpip_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_lpips_loss_forward(sample_preds, sample_targets):
    lpips_loss = LPIPSLoss()
    loss = lpips_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_ssim_loss_forward(sample_preds, sample_targets):
    ssim_loss = SSIMLoss()
    loss = ssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_msssim_loss_forward(sample_preds, sample_targets):
    msssim_loss = MSSSIMLoss()
    loss = msssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_l1_loss_forward(sample_preds, sample_targets):
    l1_loss = L1Loss()
    loss = l1_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_vgg_loss_forward(sample_preds, sample_targets):
    vgg_loss = VGGLoss()
    loss = vgg_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_total_variation_loss_forward(sample_preds):
    total_variation_loss = TotalVariationLoss()
    loss = total_variation_loss(sample_preds)
    assert isinstance(loss, torch.Tensor)


def test_gradient_loss_forward(sample_preds, sample_targets):
    gradient_loss = GradientLoss()
    loss = gradient_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_psnr_loss_forward(sample_preds, sample_targets):
    psnr_loss = PSNRLoss()
    loss = psnr_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_style_loss_forward(sample_preds, sample_targets):
    style_loss = StyleLoss()
    loss = style_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_focal_loss_forward(sample_preds, sample_targets):
    focal_loss = FocalLoss()
    loss = focal_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_elastic_loss_forward(sample_preds, sample_targets):
    elastic_loss = ElasticLoss()
    loss = elastic_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_total_variation_loss_different_dimensions():
    """Test total variation loss with different image dimensions."""
    # This will test line 278 in TotalVariationLoss (checking larger dimensions)
    x = torch.randn(2, 3, 64, 64)
    total_variation_loss = TotalVariationLoss()
    loss = total_variation_loss(x)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar


def test_focal_loss_with_alpha_multiclass():
    """Test focal loss with alpha parameter for multiclass case."""
    # Tests lines 476-479
    num_classes = 5
    batch_size = 8
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create alpha tensor for multiclass case
    alpha = torch.ones(num_classes) * 0.5
    
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    loss = focal_loss(inputs, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar


def test_elastic_loss_different_reductions():
    """Test elastic loss with different reduction modes."""
    # Tests lines 568-575 and part of 581-594
    x = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    
    # Test 'none' reduction
    elastic_loss = ElasticLoss(reduction="none")
    loss = elastic_loss(x, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == x.shape  # No reduction, so shape should match input
    
    # Test 'sum' reduction
    elastic_loss = ElasticLoss(reduction="sum")
    loss = elastic_loss(x, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar


def test_elastic_loss_beta_transition():
    """Test elastic loss transition from L1 to L2 based on beta parameter."""
    # Tests the remaining lines in 581-594
    x = torch.zeros(1, 3, 16, 16)
    
    # Create a target with varying differences
    target = torch.zeros(1, 3, 16, 16)
    target[0, 0, 0, 0] = 0.1  # Small difference (< beta)
    target[0, 0, 0, 1] = 2.0  # Large difference (> beta)
    
    beta = 0.5
    elastic_loss = ElasticLoss(beta=beta, alpha=0.5)
    loss = elastic_loss(x, target)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar
    
    # Check behavior with custom alpha
    elastic_loss = ElasticLoss(beta=beta, alpha=0.75)
    loss_more_l2 = elastic_loss(x, target)
    
    elastic_loss = ElasticLoss(beta=beta, alpha=0.25)
    loss_more_l1 = elastic_loss(x, target)
    
    # Higher alpha means more L2 contribution (smoother behavior)
    # so for same input, it should handle large differences differently
    assert loss_more_l2 != loss_more_l1


def test_style_loss_with_parameters():
    """Test style loss with different parameter configurations."""
    # Tests lines 645-648
    x = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    
    # Test with normalization
    style_loss = StyleLoss(normalize=True)
    loss_normalized = style_loss(x, target)
    assert isinstance(loss_normalized, torch.Tensor)
    
    # Test with custom layer weights
    layer_weights = {f"layer_{i}": 0.2 * i for i in range(5)}
    style_loss = StyleLoss(layer_weights=layer_weights)
    loss_weighted = style_loss(x, target)
    assert isinstance(loss_weighted, torch.Tensor)
    
    # Test without applying gram matrix (directly compare features)
    style_loss = StyleLoss(apply_gram=False)
    loss_no_gram = style_loss(x, target)
    assert isinstance(loss_no_gram, torch.Tensor)
