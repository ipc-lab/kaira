import pytest
import torch
from kaira.losses.composite import CompositeLoss
from kaira.losses.image import MSELoss, SSIMLoss

@pytest.fixture
def sample_preds():
    return torch.randn(1, 3, 32, 32)

@pytest.fixture
def sample_targets():
    return torch.randn(1, 3, 32, 32)

def test_composite_loss_forward(sample_preds, sample_targets):
    mse_loss = MSELoss()
    ssim_loss = SSIMLoss()
    losses = {"mse": mse_loss, "ssim": ssim_loss}
    weights = {"mse": 0.7, "ssim": 0.3}
    composite_loss = CompositeLoss(losses=losses, weights=weights)
    loss = composite_loss(sample_preds, sample_targets)
    assert isinstance(loss, torch.Tensor)

def test_composite_loss_individual(sample_preds, sample_targets):
    mse_loss = MSELoss()
    ssim_loss = SSIMLoss()
    losses = {"mse": mse_loss, "ssim": ssim_loss}
    weights = {"mse": 0.7, "ssim": 0.3}
    composite_loss = CompositeLoss(losses=losses, weights=weights)
    individual_losses = composite_loss.compute_individual(sample_preds, sample_targets)
    assert isinstance(individual_losses, dict)
    assert "mse" in individual_losses
    assert "ssim" in individual_losses
    assert isinstance(individual_losses["mse"], torch.Tensor)
    assert isinstance(individual_losses["ssim"], torch.Tensor)
