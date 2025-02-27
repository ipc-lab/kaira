"""Signal-to-Noise Ratio (SNR) metric."""

from typing import Tuple, Optional

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import register_metric


@register_metric("snr")
class SignalToNoiseRatio(BaseMetric):
    """Signal-to-Noise Ratio (SNR) metric.
    
    SNR measures the ratio of signal power to noise power, expressed in decibels (dB).
    Higher values indicate better signal quality.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the SNR metric.
        
        Args:
            name (Optional[str]): Optional name for the metric
        """
        super().__init__(name=name or "SNR")

    def forward(self, signal: Tensor, noisy_signal: Tensor) -> Tensor:
        """Calculate SNR between original signal and noisy signal.
        
        Args:
            signal (Tensor): Original clean signal
            noisy_signal (Tensor): Noisy version of the signal
            
        Returns:
            Tensor: SNR values in decibels (dB)
        """
        # Calculate noise
        noise = noisy_signal - signal
        
        # Calculate power of signal and noise
        signal_power = torch.mean(signal ** 2, dim=-1)
        noise_power = torch.mean(noise ** 2, dim=-1)
        
        # Avoid division by zero
        eps = torch.finfo(signal.dtype).eps
        
        # Calculate SNR in dB: 10 * log10(signal_power / noise_power)
        snr = 10 * torch.log10(signal_power / (noise_power + eps))
        
        return snr
    
    def compute_with_stats(self, signal: Tensor, noisy_signal: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute SNR with mean and standard deviation.
        
        Args:
            signal (Tensor): Original clean signal
            noisy_signal (Tensor): Noisy version of the signal
            
        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of SNR values
        """
        snr_values = self.forward(signal, noisy_signal)
        return snr_values.mean(), snr_values.std()


# Alias for backward compatibility
SNR = SignalToNoiseRatio
