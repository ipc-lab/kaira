from typing import Any, Union

import numpy as np
import torch

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


def calculate_num_filters_image(num_strided_layers, bw_ratio):
    """The function calculates the number of filters in an image based on the number of strided
    layers and a black and white ratio.

    Args:
        num_strided_layers (int): The number of strided layers in the network. These
            layers typically reduce the spatial dimensions of the input image.
        bw_ratio (float): The bandwidth ratio, which is the ratio of the number of
            filters in the current layer to the number of filters in the previous layer.

    Returns:
        int: The calculated number of filters in an image.
    """
    res = 2 * 3 * (2 ** (2 * num_strided_layers)) * bw_ratio

    assert res.is_integer()

    return res


__all__ = [
    "to_tensor",
    "calculate_num_filters_image",
    "snr_db_to_linear",
    "snr_linear_to_db",
    "snr_to_noise_power",
    "noise_power_to_snr",
    "calculate_snr",
    "add_noise_for_snr",
    "estimate_signal_power",
]
