"""Signal-to-Noise Ratio (SNR) metric.

SNR is a fundamental measure for quantifying the quality of a signal in the presence of noise,
widely used in communications and signal processing :cite:`goldsmith2005wireless` :cite:`sklar2001digital`.
"""

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("snr")
class SignalToNoiseRatio(BaseMetric):
    """Signal-to-Noise Ratio (SNR) metric.

    SNR measures the ratio of signal power to noise power, often expressed in decibels (dB).
    Higher values indicate better signal quality. It's a fundamental metric in signal processing
    and communications :cite:`goldsmith2005wireless` :cite:`sklar2001digital`.

    Attributes:
        mode (str): Output mode - "db" for decibels or "linear" for linear ratio.
    """

    def __init__(self, name: Optional[str] = None, mode: str = "db", *args: Any, **kwargs: Any):
        """Initialize the SNR metric.

        Args:
            name (Optional[str]): Optional name for the metric
            mode (str): Output mode - "db" for decibels or "linear" for linear ratio
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "SNR")  # Pass only name
        self.mode = mode.lower()
        if self.mode not in ["db", "linear"]:
            raise ValueError("Mode must be either 'db' or 'linear'")

    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the Signal-to-Noise Ratio (SNR).

        Args:
            x (Tensor): The original (clean) signal tensor.
            y (Tensor): The noisy signal tensor.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tensor: The computed SNR value(s). If input is batched, returns SNR per batch element.
        """
        # Ensure inputs are tensors
        if not isinstance(x, Tensor) or not isinstance(y, Tensor):
            raise TypeError(f"Inputs must be torch.Tensor, got {type(x)} and {type(y)}")

        # Ensure inputs have the same shape
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Calculate noise
        noise = y - x

        # Check for batch dimension (assuming dim > 1 implies batching)
        is_batched = x.dim() > 1 and x.shape[0] > 1

        if is_batched:
            result = []
            for i in range(x.size(0)):
                # Handle complex signals
                if torch.is_complex(x):
                    signal_power = torch.mean(torch.abs(x[i]) ** 2)
                    noise_power = torch.mean(torch.abs(noise[i]) ** 2)
                else:
                    # Calculate power of signal and noise
                    signal_power = torch.mean(x[i] ** 2)
                    noise_power = torch.mean(noise[i] ** 2)

                # Avoid division by zero
                eps = torch.finfo(torch.float32).eps

                # For perfect signal (no noise), return very high value approaching infinity
                if noise_power < eps:
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
            if torch.is_complex(x):
                signal_power = torch.mean(torch.abs(x) ** 2)
                noise_power = torch.mean(torch.abs(noise) ** 2)
            else:
                # Calculate power of signal and noise
                signal_power = torch.mean(x**2)
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

    def compute_with_stats(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """Compute SNR with mean and standard deviation across batches.

        Args:
            x (Tensor): The original (clean) signal tensor (batched).
            y (Tensor): The noisy signal tensor (batched).
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of the SNR values across the batch.
        """
        values = self.forward(x, y, *args, **kwargs)
        # Handle potential inf values before calculating stats
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            # Return NaN if all values were inf or input was empty
            return torch.tensor(float("nan")), torch.tensor(float("nan"))
        return values.mean(), values.std()

    def reset(self) -> None:
        """Reset accumulated statistics.

        For SNR, there are no accumulated statistics to reset as it's a direct computation.
        """
        pass


# Alias for backward compatibility
SNR = SignalToNoiseRatio
