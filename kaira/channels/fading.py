"""Fading Channel Models for Wireless Communications."""

import torch
import torch.nn.functional as F

from kaira.core import BaseChannel
from kaira.utils import to_tensor

from .utils import snr_to_noise_power


class RayleighChannel(BaseChannel):
    """Rayleigh Fading Channel.

    Models a wireless channel with Rayleigh fading, which occurs when there is no
    line-of-sight path between the transmitter and receiver. The channel coefficients
    follow a Rayleigh distribution.

    Mathematical Model:
        y = h*x + n
        where h ~ CN(0,1) (complex normal) and n ~ CN(0,σ²)

    Args:
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        normalize_energy (bool): Whether to normalize the fading coefficients to preserve energy

    Example:
        >>> channel = RayleighChannel(avg_noise_power=0.1)
        >>> x = torch.randn(32, 1, 16)
        >>> y = channel(x)
    """

    def __init__(self, avg_noise_power=None, snr_db=None, normalize_energy=True):
        super().__init__()

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

        self.normalize_energy = normalize_energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rayleigh fading and AWGN to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying fading and noise.
        """
        # Generate Rayleigh fading coefficients (complex Gaussian)
        h_real = torch.randn_like(x)
        h_imag = torch.randn_like(x)

        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        h = torch.complex(h_real, h_imag)

        # Normalize to preserve average signal energy if requested
        if self.normalize_energy:
            h = h / torch.sqrt(torch.tensor(2.0, device=h.device))

        # Apply fading
        y = h * x

        # Add noise
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(y) ** 2)
            noise_power = (
                snr_to_noise_power(signal_power, self.snr_db) * 0.5
            )  # Split between real/imag
        else:
            noise_power = self.avg_noise_power * 0.5  # Split between real/imag

        noise_real = torch.randn_like(y.real) * torch.sqrt(noise_power)
        noise_imag = torch.randn_like(y.imag) * torch.sqrt(noise_power)
        noise = torch.complex(noise_real, noise_imag)

        return y + noise


class RicianChannel(BaseChannel):
    """Rician Fading Channel.

    Models a wireless channel with Rician fading, which occurs when there is a
    line-of-sight path between the transmitter and receiver, along with multiple
    reflected paths. The K-factor represents the ratio between the power in the
    line-of-sight component and the power in the scattered paths.

    Mathematical Model:
        y = h*x + n
        where h = sqrt(K/(K+1))*direct + sqrt(1/(K+1))*scattered
        direct is deterministic, scattered ~ CN(0,1), and n ~ CN(0,σ²)

    Args:
        k_factor (float): Rician K-factor (ratio of line-of-sight to scattered power)
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        normalize_energy (bool): Whether to normalize the fading coefficients

    Example:
        >>> channel = RicianChannel(k_factor=4, avg_noise_power=0.1)
        >>> x = torch.randn(32, 1, 16)
        >>> y = channel(x)
    """

    def __init__(self, k_factor, avg_noise_power=None, snr_db=None, normalize_energy=True):
        super().__init__()
        self.k_factor = k_factor

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

        self.normalize_energy = normalize_energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rician fading and AWGN to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying fading and noise.
        """
        # Convert to complex if not already
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        k = self.k_factor

        # Direct component (line of sight)
        los_magnitude = torch.sqrt(torch.tensor(k / (k + 1), device=x.device))
        los = los_magnitude * torch.ones_like(x)  # Assuming phase=0 for los

        # Scattered component (similar to Rayleigh)
        scattered_magnitude = torch.sqrt(torch.tensor(1 / (k + 1), device=x.device))
        h_real = torch.randn_like(x.real) * scattered_magnitude
        h_imag = torch.randn_like(x.imag) * scattered_magnitude
        scattered = torch.complex(h_real, h_imag)

        # Combined channel
        h = los + scattered

        # Apply fading
        y = h * x

        # Add noise
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(y) ** 2)
            noise_power = (
                snr_to_noise_power(signal_power, self.snr_db) * 0.5
            )  # Split between real/imag
        else:
            noise_power = self.avg_noise_power * 0.5  # Split between real/imag

        noise_real = torch.randn_like(y.real) * torch.sqrt(noise_power)
        noise_imag = torch.randn_like(y.imag) * torch.sqrt(noise_power)
        noise = torch.complex(noise_real, noise_imag)

        return y + noise


class FrequencySelectiveChannel(BaseChannel):
    """Frequency-Selective Fading Channel.

    Models a wireless channel with frequency-selective fading caused by multipath
    propagation. The channel is implemented as a convolution with a random impulse
    response followed by additive noise.

    Mathematical Model:
        y = h * x + n (where * represents convolution)
        where h is the channel impulse response and n ~ CN(0,σ²)

    Args:
        tap_count (int): Number of channel taps representing multipath components
        delay_spread (float): RMS delay spread of the channel (controls power decay)
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> channel = FrequencySelectiveChannel(tap_count=5, delay_spread=1.0, snr_db=20)
        >>> x = torch.randn(32, 1, 128)
        >>> y = channel(x)
    """

    def __init__(self, tap_count, delay_spread, avg_noise_power=None, snr_db=None):
        super().__init__()
        self.tap_count = tap_count
        self.delay_spread = delay_spread

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def _generate_channel_taps(self, batch_size, device):
        """Generate exponentially decaying channel taps with Rayleigh fading."""
        # Power delay profile (exponential decay)
        delays = torch.arange(self.tap_count, device=device).float()
        power_profile = torch.exp(-delays / self.delay_spread)

        # Generate complex Gaussian taps
        h_real = torch.randn(batch_size, 1, self.tap_count, device=device)
        h_imag = torch.randn(batch_size, 1, self.tap_count, device=device)

        # Scale by power profile and normalize
        h = torch.complex(h_real, h_imag) * torch.sqrt(power_profile / 2)

        # Normalize to preserve energy
        norm_factor = torch.sqrt(torch.sum(torch.abs(h) ** 2, dim=2, keepdim=True))
        h = h / norm_factor

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-selective fading and AWGN to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, channels, samples).

        Returns:
            torch.Tensor: The output tensor after applying fading and noise.
        """
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        batch_size = x.shape[0]
        device = x.device

        # Generate channel impulse response
        h = self._generate_channel_taps(batch_size, device)

        # Apply convolution separately for each batch and channel
        y = torch.zeros_like(x)
        for b in range(batch_size):
            y[b] = F.conv1d(
                x[b].unsqueeze(0).real, h[b].real, padding=self.tap_count - 1
            ) + 1j * F.conv1d(x[b].unsqueeze(0).imag, h[b].real, padding=self.tap_count - 1)

        # Remove padding
        y = y[:, :, : x.shape[2]]

        # Add noise
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(y) ** 2)
            noise_power = snr_to_noise_power(signal_power, self.snr_db) * 0.5
        else:
            noise_power = self.avg_noise_power * 0.5

        noise_real = torch.randn_like(y.real) * torch.sqrt(noise_power)
        noise_imag = torch.randn_like(y.imag) * torch.sqrt(noise_power)
        noise = torch.complex(noise_real, noise_imag)

        return y + noise
