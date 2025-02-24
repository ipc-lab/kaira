# tests/test_utils/test_losses.py
import pytest
import torch

from kaira.utils.losses import (
    CombinedLoss,
    LPIPSLoss,
    MSELoss,
    MSELPIPSLoss,
    MSSSIMLoss,
    SSIMLoss,
)


@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor."""
    return torch.randn(1, 3, 32, 32)


def test_mse_loss_forward(sample_preds, sample_targets):
    """Test MSELoss forward pass."""
    mse_loss = MSELoss()
    loss = mse_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_combined_loss_forward(sample_preds, sample_targets):
    """Test CombinedLoss forward pass."""
    loss1 = MSELoss()
    loss2 = LPIPSLoss()
    weights = [0.5, 0.5]
    combined_loss = CombinedLoss(losses=[loss1, loss2], weights=weights)
    loss = combined_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_mselpip_loss_forward(sample_preds, sample_targets):
    """Test MSELPIPSLoss forward pass."""
    mselpip_loss = MSELPIPSLoss(lpips_weight=0.5)
    loss = mselpip_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_lpips_loss_forward(sample_preds, sample_targets):
    """Test LPIPSLoss forward pass."""
    lpips_loss = LPIPSLoss()
    loss = lpips_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_ssim_loss_forward(sample_preds, sample_targets):
    """Test SSIMLoss forward pass."""
    ssim_loss = SSIMLoss()
    loss = ssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


def test_msssim_loss_forward(sample_preds, sample_targets):
    """Test MSSSIMLoss forward pass."""
    msssim_loss = MSSSIMLoss()
    loss = msssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize("kernel_size", [7, 11, 15])
def test_ssim_loss_different_kernel_sizes(sample_preds, sample_targets, kernel_size):
    """Test SSIMLoss with different kernel sizes."""
    ssim_loss = SSIMLoss(kernel_size=kernel_size)
    loss = ssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize("kernel_size", [7, 11, 15])
def test_msssim_loss_different_kernel_sizes(sample_preds, sample_targets, kernel_size):
    """Test MSSSIMLoss with different kernel sizes."""
    msssim_loss = MSSSIMLoss(kernel_size=kernel_size)
    loss = msssim_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)
