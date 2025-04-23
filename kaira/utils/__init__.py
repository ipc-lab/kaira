"""General utility functions for the Kaira library."""

import os
import random
from typing import Any, Union

import numpy as np
import torch

from .capacity_analyzer import CapacityAnalyzer
from .snr import (
    add_noise_for_snr,
    calculate_snr,
    estimate_signal_power,
    noise_power_to_snr,
    snr_db_to_linear,
    snr_linear_to_db,
    snr_to_noise_power,
)


def to_tensor(x: Any, device: Union[str, torch.device, None] = None) -> torch.Tensor:
    """Convert an input data into a torch.Tensor, with an option to move it to a specific device.

    Args:
        x (Any): The data to be converted. Acceptable types are:
            - torch.Tensor: Returned as is (optionally moved to the specified device).
            - int or float: Converted to a scalar tensor.
            - list or numpy.ndarray: Converted to a tensor.
        device (Union[str, torch.device, None]): The target device for the tensor
            (for example, 'cpu' or 'cuda'). Default is None.

    Returns:
        torch.Tensor: The input data converted to a tensor on the specified device if provided.

    Raises:
        TypeError: If the input type is not supported for conversion.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    elif isinstance(x, (int, float)):
        return torch.tensor(x, device=device)
    elif isinstance(x, (list, np.ndarray)):
        return torch.tensor(x, device=device)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def calculate_num_filters_factor_image(num_strided_layers, bw_ratio, channels=3, is_complex_transmission=False):
    """Calculate the number of filters in an image based on the number of strided layers and
    bandwidth ratio.

    Args:
        num_strided_layers (int): The number of strided layers in the network. These
            layers typically reduce the spatial dimensions of the input image.
        bw_ratio (float): The bandwidth ratio, which is the ratio of the number of
            transmitted filters to the number of filters in the image.
        channels (int, optional): The number of channels in the input image. Defaults to 3.
        is_complex_transmission (bool, optional): If True, indicates that the transmission
            is complex. Defaults to False.

    Returns:
        int: The calculated number of filters in an image.
    """

    # The formula according to the test cases:
    base_filters = channels * (2 ** (2 * num_strided_layers))
    res = base_filters * bw_ratio

    if is_complex_transmission:
        res *= 2

    assert res.is_integer(), f"Result {res} is not an integer"

    return int(res)


def seed_everything(seed: int, cudnn_benchmark: bool = False, cudnn_deterministic: bool = True):
    """Seed all random number generators to make runs reproducible.

    Args:
        seed (int): The seed value for random number generators.
        cudnn_benchmark (bool): If True, allows the use of CuDNN's auto-tuner to find the best algorithm for your hardware. Setting this False might have performance implications.
        cudnn_deterministic (bool): If True, makes CuDNN operations deterministic. Setting this False might have performance implications.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


__all__ = [
    "CapacityAnalyzer",
    "to_tensor",
    "calculate_num_filters_factor_image",
    "snr_db_to_linear",
    "snr_linear_to_db",
    "snr_to_noise_power",
    "noise_power_to_snr",
    "calculate_snr",
    "add_noise_for_snr",
    "estimate_signal_power",
]
