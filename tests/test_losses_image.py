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
