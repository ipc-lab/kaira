"""Additive White Gaussian Noise Channel Implementations."""

import torch

from kaira.core import BaseChannel
from kaira.utils import to_tensor

from .utils import snr_to_noise_power


class AWGNChannel(BaseChannel):
    """Additive White Gaussian Noise (AWGN) Channel.

    This channel adds real-valued Gaussian noise to the input signal.
    The noise follows the distribution N(0, σ²) where σ² is the average noise power.

    Mathematical Model:
        y = x + n
        where n ~ N(0, σ²)

    Args:
        avg_noise_power (float): The average noise power σ² (variance of the Gaussian noise)
        snr_db (float, optional): Alternative way to specify noise as SNR in dB. If provided,
            it overrides avg_noise_power.

    Example:
        >>> channel = AWGNChannel(avg_noise_power=0.1)
        >>> x = torch.ones(10, 1)  # Input signal
        >>> y = channel(x)  # Noisy output

        >>> # Alternatively, specify SNR in dB
        >>> channel = AWGNChannel(snr_db=20)  # 20dB SNR
        >>> y = channel(x)
    """

    def __init__(self, avg_noise_power=None, snr_db=None):
        """Initialize the AWGNChannel object.

        Args:
            avg_noise_power (float, optional): The average noise power.
            snr_db (float, optional): Signal-to-noise ratio in dB.

        Returns:
            None
        """
        super().__init__()

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None  # Will be calculated per input in forward
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply AWGN (Additive White Gaussian Noise) to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape BxCxWxH.

        Returns:
            torch.Tensor: The output tensor after adding AWGN.
        """
        # If SNR was specified, calculate noise power based on input signal power
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(x) ** 2)
            noise_power = snr_to_noise_power(signal_power, self.snr_db)

        awgn = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + awgn


class ComplexAWGNChannel(BaseChannel):
    """Complex Additive White Gaussian Noise (AWGN) Channel.

    This channel adds complex-valued Gaussian noise to the input signal.
    The noise follows CN(0, σ²) where σ² is split between real and imaginary components.

    Mathematical Model:
        y = x + n
        where n ~ CN(0, σ²) = N(0, σ²/2) + jN(0, σ²/2)

    Args:
        avg_noise_power (float, optional): The total average noise power σ² (split between real/imaginary)
        snr_db (float, optional): Alternative way to specify noise as SNR in dB. If provided,
            it overrides avg_noise_power.

    Example:
        >>> channel = ComplexAWGNChannel(avg_noise_power=0.1)
        >>> x = torch.complex(torch.ones(10, 1), torch.zeros(10, 1))
        >>> y = channel(x)  # Complex noisy output
    """

    def __init__(self, avg_noise_power=None, snr_db=None):
        super().__init__()

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None  # Will be calculated per input in forward
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power) * 0.5  # Split between real/imag
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ComplexAWGNChannel.

        Args:
            x (torch.Tensor): The input signal tensor (complex).

        Returns:
            torch.Tensor: The output signal tensor after adding complex Gaussian noise.
        """
        # If SNR was specified, calculate noise power based on input signal power
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(x) ** 2)
            noise_power = (
                snr_to_noise_power(signal_power, self.snr_db) * 0.5
            )  # Split between real/imag

        noise_real = torch.randn_like(x.real) * torch.sqrt(noise_power)
        noise_imag = torch.randn_like(x.imag) * torch.sqrt(noise_power)
        noise = torch.complex(noise_real, noise_imag)

        return x + noise
