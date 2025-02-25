"""Communication Channel Models for Signal Processing.

This module provides various channel models used in communication systems and signal
processing applications. Each channel model simulates different types of signal
distortion and noise that occur in real-world communication systems.

Available Channels:
    - AWGNChannel: Additive White Gaussian Noise channel
    - ComplexAWGNChannel: Complex-valued AWGN channel
    - PerfectChannel: Ideal channel with no distortion

Each channel implements a forward() method that takes an input tensor and returns
the output tensor after applying the channel effects.

Example:
    >>> from kaira.channels import AWGNChannel
    >>> channel = AWGNChannel(avg_noise_power=0.1)
    >>> input_signal = torch.randn(32, 1, 16, 16)
    >>> noisy_signal = channel(input_signal)
"""

import torch

from kaira.core import BaseChannel
from kaira.utils import to_tensor

__all__ = [
    "PerfectChannel",
    "AWGNChannel",
    "ComplexAWGNChannel",
]


class AWGNChannel(BaseChannel):
    """Additive White Gaussian Noise (AWGN) Channel.

    This channel adds real-valued Gaussian noise to the input signal.
    The noise follows the distribution N(0, σ²) where σ² is the average noise power.

    Mathematical Model:
        y = x + n
        where n ~ N(0, σ²)

    Args:
        avg_noise_power (float): The average noise power σ² (variance of the Gaussian noise)

    Example:
        >>> channel = AWGNChannel(avg_noise_power=0.1)
        >>> x = torch.ones(10, 1)  # Input signal
        >>> y = channel(x)  # Noisy output
    """

    def __init__(self, avg_noise_power: float):
        """Initialize the AWGNChannel object.

        Args:
            avg_noise_power (float): The average noise power.

        Returns:
            None
        """
        super().__init__()
        self.avg_noise_power = to_tensor(avg_noise_power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AWGN (Additive White Gaussian Noise) to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape BxCxWxH.

        Returns:
            torch.Tensor: The output tensor after adding AWGN.
        """
        awgn = torch.randn_like(x) * torch.sqrt(self.avg_noise_power)
        x = x + awgn
        return x


class ComplexAWGNChannel(BaseChannel):
    """Complex Additive White Gaussian Noise (AWGN) Channel.

    This channel adds complex-valued Gaussian noise to the input signal.
    The noise follows CN(0, σ²) where σ² is split between real and imaginary components.

    Mathematical Model:
        y = x + n
        where n ~ CN(0, σ²) = N(0, σ²/2) + jN(0, σ²/2)

    Args:
        avg_noise_power (float): The total average noise power σ² (split between real/imaginary)

    Example:
        >>> channel = ComplexAWGNChannel(avg_noise_power=0.1)
        >>> x = torch.complex(torch.ones(10, 1), torch.zeros(10, 1))
        >>> y = channel(x)  # Complex noisy output
    """

    def __init__(self, avg_noise_power: float):
        super().__init__()
        self.avg_noise_power = to_tensor(avg_noise_power) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ComplexAWGNChannel.

        Args:
            x (torch.Tensor): The input signal tensor.

        Returns:
            torch.Tensor: The output signal tensor after adding complex Gaussian noise (equivalent to standard domain noise, but in complex domain).
        """
        awgn = torch.randn_like(x) * torch.sqrt(
            torch.tensor(self.avg_noise_power, device=x.device)
        )
        x = x + awgn
        return x


class PerfectChannel(BaseChannel):
    """Perfect (identity) channel that passes signals unchanged.

    This channel represents an ideal communication medium with no distortion,
    noise, or interference. It simply returns the input signal as is.

    Mathematical Model:
        y = x

    Example:
        >>> channel = PerfectChannel()
        >>> x = torch.randn(10, 1)
        >>> y = channel(x)  # y is identical to x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass of the perfect channel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor without any modification.
        """
        return x
