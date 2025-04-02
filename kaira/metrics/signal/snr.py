"""Signal-to-Noise Ratio (SNR) metric.

SNR is a fundamental measure for quantifying the quality of a signal in the presence of noise,
widely used in communications and signal processing :cite:`goldsmith2005wireless` :cite:`sklar2001digital`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("snr")
class SignalToNoiseRatio(BaseMetric):
    """Signal-to-Noise Ratio (SNR) metric.

    SNR measures the ratio of signal power to noise power, expressed in decibels (dB). Higher
    values indicate better signal quality :cite:`sklar2001digital`. This metric is fundamental
    in determining the performance limits of communication systems as discussed in
    :cite:`shannon1948mathematical`.
    """

    def __init__(self, name: Optional[str] = None, mode: str = "db"):
        """Initialize the SNR metric.

        Args:
            name (Optional[str]): Optional name for the metric
            mode (str): Output mode - "db" for decibels or "linear" for linear ratio
        """
        super().__init__(name=name or "SNR")
        self.mode = mode.lower()
        if self.mode not in ["db", "linear"]:
            raise ValueError("Mode must be either 'db' or 'linear'")

    def forward(self, signal: Tensor, noisy_signal: Tensor) -> Tensor:
        """Calculate SNR between original signal and noisy signal.

        Args:
            signal (Tensor): Original clean signal
            noisy_signal (Tensor): Noisy version of the signal

        Returns:
            Tensor: SNR values in decibels (dB) or linear ratio based on mode
        """
        # Check for batch dimension
        is_batched = signal.dim() > 1 and signal.size(0) > 1

        # Calculate noise
        noise = noisy_signal - signal

        if is_batched:
            # Process each sample in the batch independently
            result = []
            for i in range(signal.size(0)):
                # Handle complex signals
                if torch.is_complex(signal):
                    signal_power = torch.mean(torch.abs(signal[i]) ** 2)
                    noise_power = torch.mean(torch.abs(noise[i]) ** 2)
                else:
                    # Calculate power of signal and noise
                    signal_power = torch.mean(signal[i] ** 2)
                    noise_power = torch.mean(noise[i] ** 2)

                # Avoid division by zero
                eps = torch.finfo(torch.float32).eps

                # For perfect signal (no noise), return very high value approaching infinity
                if noise_power < eps:
                    if self.mode == "db":
                        result.append(torch.tensor(float("inf")))
                    else:
                        result.append(torch.tensor(float("inf")))
                else:
                    # Calculate SNR
                    snr_linear = signal_power / (noise_power + eps)
                    if self.mode == "db":
                        # Convert to dB: 10 * log10(signal_power / noise_power)
                        snr = 10 * torch.log10(snr_linear)
                    else:
                        # Return linear ratio
                        snr = snr_linear
                    result.append(snr)

            return torch.stack(result)
        else:
            # Handle complex signals
            if torch.is_complex(signal):
                signal_power = torch.mean(torch.abs(signal) ** 2)
                noise_power = torch.mean(torch.abs(noise) ** 2)
            else:
                # Calculate power of signal and noise
                signal_power = torch.mean(signal**2)
                noise_power = torch.mean(noise**2)

            # Avoid division by zero
            eps = torch.finfo(torch.float32).eps

            # For perfect signal (no noise), return very high value approaching infinity
            if noise_power < eps:
                return torch.tensor(float("inf"))

            # Calculate SNR in linear form
            snr_linear = signal_power / (noise_power + eps)
            
            # Convert to dB if needed
            if self.mode == "db":
                snr = 10 * torch.log10(snr_linear)
            else:
                snr = snr_linear

            # Return scalar tensor
            return snr.squeeze()

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

    def reset(self) -> None:
        """Reset accumulated statistics.

        For SNR, there are no accumulated statistics to reset as it's a direct computation.
        """
        pass


# Alias for backward compatibility
SNR = SignalToNoiseRatio
