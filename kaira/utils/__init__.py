from typing import Any, Union

import numpy as np
import torch

from . import losses


def snr_linear_to_db(snr_linear):
    """The function converts a signal-to-noise ratio from linear scale to decibel scale.

    Parameters
    ----------
    snr_linear
        The parameter `snr_linear` represents the signal-to-noise ratio (SNR) in linear scale.

    Returns
    -------
        the SNR (Signal-to-Noise Ratio) in decibels (dB) given the SNR in linear scale.
    """

    return 10 * torch.log10(torch.tensor(snr_linear))


def snr_db_to_linear(snr_db):
    """The function converts a signal-to-noise ratio value from decibels to linear scale.

    Parameters
    ----------
    snr_db
        The parameter "snr_db" represents the signal-to-noise ratio in decibels.

    Returns
    -------
        the linear value of the signal-to-noise ratio (SNR) in decibels.
    """

    return 10 ** (snr_db / 10)


def to_tensor(x: Any, device: Union[str, torch.device, None] = None) -> torch.Tensor:
    """Convert an input data into a torch.Tensor, with an option to move it to a specific device.

    Parameters
    ----------
    x : Any
        The data to be converted. Acceptable types are:
            - torch.Tensor: Returned as is (optionally moved to the specified device).
            - int or float: Converted to a scalar tensor.
            - list or numpy.ndarray: Converted to a tensor.
    device : Union[str, torch.device, None], optional
        The target device for the tensor (for example, 'cpu' or 'cuda'). Default is None.

    Returns
    -------
    torch.Tensor
        The input data converted to a tensor on the specified device if provided.

    Raises
    ------
    TypeError
        If the input type is not supported for conversion.
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

    Parameters
    ----------
    num_strided_layers
        The parameter "num_strided_layers" represents the number of strided layers in the network. These
    layers typically reduce the spatial dimensions of the input image.
    bw_ratio
        The `bw_ratio` parameter represents the bandwidth ratio, which is the ratio of the number of
    filters in the current layer to the number of filters in the previous layer.

    Returns
    -------
        the calculated number of filters in an image.
    """
    res = 2 * 3 * (2 ** (2 * num_strided_layers)) * bw_ratio

    assert res.is_integer()

    return res


__all__ = [
    "snr_linear_to_db",
    "snr_db_to_linear",
    "to_tensor",
    "calculate_num_filters_image",
    "losses",
]
