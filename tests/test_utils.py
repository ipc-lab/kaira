# tests/test_utils.py
import pytest
import torch
import numpy as np
from kaira.utils import (
    snr_linear_to_db,
    snr_db_to_linear,
    to_tensor,
    calculate_num_filters_image,
    losses
)

@pytest.mark.parametrize("snr_linear", [1.0, 10.0, 100.0])
def test_snr_linear_to_db(snr_linear):
    """Test snr_linear_to_db function."""
    snr_db = snr_linear_to_db(snr_linear)
    assert isinstance(snr_db, torch.Tensor)
    assert torch.isclose(snr_db, torch.tensor(10 * np.log10(snr_linear)))

@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_snr_db_to_linear(snr_db):
    """Test snr_db_to_linear function."""
    snr_linear = snr_db_to_linear(snr_db)
    assert isinstance(snr_linear, float)
    assert np.isclose(snr_linear, 10 ** (snr_db / 10))

@pytest.mark.parametrize(
    "x, expected_type",
    [
        (torch.tensor([1, 2, 3]), torch.Tensor),
        (1, torch.Tensor),
        (1.0, torch.Tensor),
        ([1, 2, 3], torch.Tensor),
        (np.array([1, 2, 3]), torch.Tensor),
    ],
)
def test_to_tensor(x, expected_type):
    """Test to_tensor function with various inputs."""
    tensor = to_tensor(x)
    assert isinstance(tensor, expected_type)

def test_to_tensor_device():
    """Test to_tensor function with device argument."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = to_tensor([1, 2, 3], device=device)
        assert tensor.device == device

@pytest.mark.parametrize("num_strided_layers, bw_ratio", [(1, 1.0), (2, 2.0)])
def test_calculate_num_filters_image(num_strided_layers, bw_ratio):
    """Test calculate_num_filters_image function."""
    num_filters = calculate_num_filters_image(num_strided_layers, bw_ratio)
    assert isinstance(num_filters, float)
    expected_filters = 2 * 3 * (2 ** (2 * num_strided_layers)) * bw_ratio
    assert np.isclose(num_filters, expected_filters)

def test_to_tensor_unsupported_type():
    """Test to_tensor function with unsupported type."""
    with pytest.raises(TypeError):
        to_tensor("string")

# tests/test_utils/test_losses.py
import pytest
import torch
import torch.nn as nn
from kaira.utils import losses
from kaira.utils.losses import (
    MSELoss,
    CombinedLoss,
    MSELPIPSLoss,
    LPIPSLoss,
    SSIMLoss,
    MSSSIMLoss,
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
