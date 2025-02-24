# tests/test_utils.py
import numpy as np
import pytest
import torch

from kaira.utils import (
    calculate_num_filters_image,
    snr_db_to_linear,
    snr_linear_to_db,
    to_tensor,
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
