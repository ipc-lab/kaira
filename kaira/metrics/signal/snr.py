"""Signal-to-Noise Ratio (SNR) metric.

SNR is a fundamental measure for quantifying the quality of a signal in the presence of noise,
widely used in communications and signal processing :cite:`goldsmith2005wireless` :cite:`sklar2001digital`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import register_metric


@register_metric("snr")
class SignalToNoiseRatio(BaseMetric):
    """Signal-to-Noise Ratio (SNR) metric.

    SNR measures the ratio of signal power to noise power, expressed in decibels (dB). Higher
    values indicate better signal quality :cite:`sklar2001digital`. This metric is fundamental
    in determining the performance limits of communication systems as discussed in 
    :cite:`shannon1948mathematical`.
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
        signal_power = torch.mean(signal**2, dim=-1)
        noise_power = torch.mean(noise**2, dim=-1)

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
