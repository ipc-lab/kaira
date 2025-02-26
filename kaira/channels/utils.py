"""Utility Functions for Communication Channels."""

import torch
import math
from typing import Union, Tuple

def snr_to_noise_power(signal_power: Union[float, torch.Tensor], 
                       snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
    """Convert SNR in dB to noise power.

    Args:
        signal_power (float or torch.Tensor): Power of the signal
        snr_db (float or torch.Tensor): Signal-to-Noise Ratio in dB

    Returns:
        torch.Tensor: Corresponding noise power
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Calculate noise power: signal_power / snr
    noise_power = signal_power / snr_linear
    
    return noise_power


def noise_power_to_snr(signal_power: Union[float, torch.Tensor], 
                       noise_power: Union[float, torch.Tensor]) -> torch.Tensor:
    """Convert noise power to SNR in dB.

    Args:
        signal_power (float or torch.Tensor): Power of the signal
        noise_power (float or torch.Tensor): Power of the noise

    Returns:
        torch.Tensor: Signal-to-Noise Ratio in dB
    """
    # Calculate SNR in linear scale
    snr_linear = signal_power / noise_power
    
    # Convert to dB
    snr_db = 10 * torch.log10(snr_linear)
    
    return snr_db


def calculate_snr(original_signal: torch.Tensor, 
                  noisy_signal: torch.Tensor) -> Union[float, torch.Tensor]:
    """Calculate the SNR between original and noisy signals.

    Args:
        original_signal (torch.Tensor): The original clean signal
        noisy_signal (torch.Tensor): The noisy signal

    Returns:
        float or torch.Tensor: SNR in dB
    """
    # Ensure tensors have the same shape
    if original_signal.shape != noisy_signal.shape:
        raise ValueError("Original and noisy signals must have the same shape")
    
    # Complex signals need special handling
    if torch.is_complex(original_signal):
        original_signal_power = torch.mean(torch.abs(original_signal) ** 2)
        noise = noisy_signal - original_signal
        noise_power = torch.mean(torch.abs(noise) ** 2)
    else:
        original_signal_power = torch.mean(original_signal ** 2)
        noise = noisy_signal - original_signal
        noise_power = torch.mean(noise ** 2)
    
    # Handle zero noise case
    if noise_power == 0:
        return torch.tensor(float('inf'))
    
    # Calculate SNR in dB
    snr_db = 10 * torch.log10(original_signal_power / noise_power)
    
    return snr_db


def evaluate_ber(transmitted_bits: torch.Tensor,
                received_bits: torch.Tensor) -> torch.Tensor:
    """Calculate the Bit Error Rate (BER) between transmitted and received bits.
    
    Args:
        transmitted_bits (torch.Tensor): Original transmitted bits (0s and 1s)
        received_bits (torch.Tensor): Received/detected bits (0s and 1s)
        
    Returns:
        torch.Tensor: Bit Error Rate (between 0 and 1)
    """
    if transmitted_bits.shape != received_bits.shape:
        raise ValueError("Transmitted and received bits must have the same shape")
    
    # Count bit errors
    errors = torch.sum(transmitted_bits != received_bits)
    
    # Total bits
    total_bits = transmitted_bits.numel()
    
    # Calculate BER
    ber = errors.float() / total_bits
    
    return ber
