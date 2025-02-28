from typing import Optional, Tuple, Union

import torch


def snr_db_to_linear(snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
    """Convert Signal-to-Noise Ratio from decibel to linear scale.

    Args:
        snr_db (Union[float, torch.Tensor]): Signal-to-Noise Ratio in decibels (dB).

    Returns:
        torch.Tensor: Signal-to-Noise Ratio in linear scale.
    """
    if isinstance(snr_db, float):
        snr_db = torch.tensor(snr_db)
    return 10 ** (snr_db / 10.0)


def snr_linear_to_db(snr_linear: Union[float, torch.Tensor]) -> torch.Tensor:
    """Convert Signal-to-Noise Ratio from linear scale to decibels.

    Args:
        snr_linear (Union[float, torch.Tensor]): SNR in linear scale.

    Returns:
        torch.Tensor: SNR in decibel (dB) scale.

    Raises:
        ValueError: If snr_linear contains zero or negative values.
    """
    if isinstance(snr_linear, float):
        snr_linear = torch.tensor(snr_linear)

    if torch.any(snr_linear <= 0):
        raise ValueError("SNR in linear scale must be positive for dB conversion")

    return 10 * torch.log10(snr_linear)


def snr_to_noise_power(
    signal_power: Union[float, torch.Tensor], snr_db: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Convert SNR in dB to noise power given a signal power.

    Args:
        signal_power (Union[float, torch.Tensor]): Power of the signal.
        snr_db (Union[float, torch.Tensor]): Signal-to-Noise Ratio in decibels (dB).

    Returns:
        torch.Tensor: Corresponding noise power for the specified SNR.
    """
    snr_linear = snr_db_to_linear(snr_db)
    return signal_power / snr_linear


def noise_power_to_snr(
    signal_power: Union[float, torch.Tensor], noise_power: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Calculate SNR in dB given signal and noise power.

    Args:
        signal_power (Union[float, torch.Tensor]): Power of the signal.
        noise_power (Union[float, torch.Tensor]): Power of the noise.

    Returns:
        torch.Tensor: Signal-to-Noise Ratio in decibels (dB).

    Raises:
        ValueError: If noise_power contains zero values (would result in infinite SNR).
    """
    if isinstance(signal_power, float):
        signal_power = torch.tensor(signal_power)
    if isinstance(noise_power, float):
        noise_power = torch.tensor(noise_power)

    if torch.any(noise_power == 0):
        raise ValueError("Noise power cannot be zero (would result in infinite SNR)")

    snr_linear = signal_power / noise_power
    return snr_linear_to_db(snr_linear)


def calculate_snr(
    original_signal: torch.Tensor,
    noisy_signal: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    """Calculate the SNR between original and noisy signals.

    Args:
        original_signal (torch.Tensor): The original clean signal.
        noisy_signal (torch.Tensor): The noisy signal (original signal plus noise).
        dim (Optional[Union[int, Tuple[int, ...]]]): Dimensions to reduce when calculating power.
            If None, uses all dimensions.
        keepdim (bool): Whether to keep the reduced dimensions in the output. Default is False.

    Returns:
        torch.Tensor: SNR in decibels (dB).

    Raises:
        ValueError: If original and noisy signals have different shapes.
    """
    # Ensure tensors have the same shape
    if original_signal.shape != noisy_signal.shape:
        raise ValueError("Original and noisy signals must have the same shape")

    # Extract noise component
    noise = noisy_signal - original_signal

    # Calculate powers based on signal type
    if torch.is_complex(original_signal):
        original_power = torch.mean(torch.abs(original_signal) ** 2, dim=dim, keepdim=keepdim)
        noise_power = torch.mean(torch.abs(noise) ** 2, dim=dim, keepdim=keepdim)
    else:
        original_power = torch.mean(original_signal**2, dim=dim, keepdim=keepdim)
        noise_power = torch.mean(noise**2, dim=dim, keepdim=keepdim)

    # Handle zero noise case
    eps = torch.finfo(original_power.dtype).eps
    noise_power = torch.clamp(noise_power, min=eps)

    # Calculate SNR in dB
    return 10 * torch.log10(original_power / noise_power)


def add_noise_for_snr(
    signal: torch.Tensor,
    target_snr_db: Union[float, torch.Tensor],
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to achieve a target Signal-to-Noise Ratio.

    Args:
        signal (torch.Tensor): The original clean signal.
        target_snr_db (Union[float, torch.Tensor]): Target Signal-to-Noise Ratio in decibels (dB).
        dim (Optional[Union[int, Tuple[int, ...]]]): Dimensions to reduce when calculating power.
            If None, uses all dimensions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Noisy signal (original signal with added noise)
            - The generated noise component
    """
    # Calculate signal power
    if torch.is_complex(signal):
        signal_power = torch.mean(torch.abs(signal) ** 2, dim=dim, keepdim=True)
    else:
        signal_power = torch.mean(signal**2, dim=dim, keepdim=True)

    # Calculate required noise power
    noise_power = snr_to_noise_power(signal_power, target_snr_db)

    # Generate noise with the right power
    if torch.is_complex(signal):
        # For complex signals, generate complex noise
        noise_std = torch.sqrt(noise_power / 2)
        real_noise = torch.randn_like(signal.real) * noise_std
        imag_noise = torch.randn_like(signal.imag) * noise_std
        noise = torch.complex(real_noise, imag_noise)
    else:
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std

    noisy_signal = signal + noise

    return noisy_signal, noise


def estimate_signal_power(
    signal: torch.Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
) -> torch.Tensor:
    """Estimate the power of a signal.

    Args:
        signal (torch.Tensor): The input signal (real or complex).
        dim (Optional[Union[int, Tuple[int, ...]]]): Dimensions to reduce when calculating power.
            If None, uses all dimensions.
        keepdim (bool): Whether to keep the reduced dimensions in the output. Default is False.

    Returns:
        torch.Tensor: Signal power estimation.
    """
    if torch.is_complex(signal):
        return torch.mean(torch.abs(signal) ** 2, dim=dim, keepdim=keepdim)
    else:
        return torch.mean(signal**2, dim=dim, keepdim=keepdim)
