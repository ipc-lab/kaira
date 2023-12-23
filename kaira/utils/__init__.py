import torch

def snr_linear_to_db(snr_linear):
    '''The function converts a signal-to-noise ratio from linear scale to decibel scale.
    
    Parameters
    ----------
    snr_linear
        The parameter `snr_linear` represents the signal-to-noise ratio (SNR) in linear scale.
    
    Returns
    -------
        the SNR (Signal-to-Noise Ratio) in decibels (dB) given the SNR in linear scale.
    
    '''

    return 10 * torch.log10(torch.tensor(snr_linear))

def snr_db_to_linear(snr_db):
    '''The function converts a signal-to-noise ratio value from decibels to linear scale.
    
    Parameters
    ----------
    snr_db
        The parameter "snr_db" represents the signal-to-noise ratio in decibels.
    
    Returns
    -------
        the linear value of the signal-to-noise ratio (SNR) in decibels.
    
    '''

    return 10 ** (snr_db / 10)

def to_tensor(x, device=None):
    '''The function `to_tensor` converts a given input `x` to a PyTorch tensor, and optionally assigns it
    to a specified device.
    
    Parameters
    ----------
    x
        The parameter `x` can be any input value that you want to convert to a PyTorch tensor. It can be a
    scalar value, a list, a numpy array, or any other supported data type.
    device
        The `device` parameter is an optional argument that specifies the device (e.g., CPU or GPU) on
    which the tensor should be created. If `device` is not provided, the tensor will be created on the
    default device.
    
    Returns
    -------
        The function `to_tensor` returns a torch.Tensor object.
    
    '''
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, float):
        return torch.tensor(x, device=device)
    else:
        raise TypeError("Unsupported type: {}".format(type(x)))

def calculate_num_filters_image(num_strided_layers, bw_ratio):
    '''The function calculates the number of filters in an image based on the number of strided layers and
    a black and white ratio.
    
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
    
    '''
    res = 2 * 3 * (2 ** (2 * num_strided_layers)) * bw_ratio
    
    assert res.is_integer()
    
    return res
