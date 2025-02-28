"""Analog Channel Implementations for Continuous-Input Signals.

This module provides implementations of channels with continuous inputs, supporting both real and
complex-valued signals. These channels represent various types of noise and distortions found in
analog communication systems.
"""

import numpy as np
import torch

from kaira.utils import snr_to_noise_power, to_tensor

from .base import BaseChannel


def _apply_noise(x: torch.Tensor, noise_power=None, snr_db=None) -> torch.Tensor:
    """Add Gaussian noise to a signal with specified power or SNR.

    Automatically handles both real and complex signals by adding
    appropriate noise to each component.

    Args:
        x (torch.Tensor): The input signal (real or complex)
        noise_power (float, optional): The noise power to apply
        snr_db (float, optional): The SNR in dB (alternative to noise_power)

    Returns:
        torch.Tensor: The signal with added noise
    """
    # Calculate noise power if SNR specified
    if snr_db is not None:
        signal_power = torch.mean(torch.abs(x) ** 2)
        noise_power = snr_to_noise_power(signal_power, snr_db)

    # Add appropriate noise type
    if torch.is_complex(x):
        # For complex signals, split noise power between real/imag components
        noise_power_component = noise_power * 0.5
        noise_real = torch.randn_like(x.real) * torch.sqrt(noise_power_component)
        noise_imag = torch.randn_like(x.imag) * torch.sqrt(noise_power_component)
        noise = torch.complex(noise_real, noise_imag)
    else:
        # For real signals, apply all noise power
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

    return x + noise


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert a real tensor to complex by adding zero imaginary part.

    If the input is already complex, it is returned unchanged.

    Args:
        x (torch.Tensor): Input tensor (real or complex)

    Returns:
        torch.Tensor: Complex tensor
    """
    if torch.is_complex(x):
        return x
    else:
        return torch.complex(x, torch.zeros_like(x))


class AWGNChannel(BaseChannel):
    """Additive white Gaussian noise (AWGN) channel for signal transmission.

    This channel adds Gaussian noise to the input signal, supporting both real
    and complex-valued inputs automatically. For complex inputs, noise is added
    to both real and imaginary components.

    Mathematical Model:
        y = x + n
        where n ~ N(0, σ²) for real inputs
        or n ~ CN(0, σ²) for complex inputs

    Args:
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> # For real-valued signals
        >>> channel = AWGNChannel(avg_noise_power=0.1)
        >>> x_real = torch.ones(10, 1)
        >>> y_real = channel(x_real)  # Real noisy output

        >>> # For complex-valued signals (same channel works)
        >>> x_complex = torch.complex(torch.ones(10, 1), torch.zeros(10, 1))
        >>> y_complex = channel(x_complex)  # Complex noisy output
    """

    def __init__(self, avg_noise_power=None, snr_db=None):
        super().__init__()

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the input signal.

        Automatically handles both real and complex-valued inputs.

        Args:
            x (torch.Tensor): The input tensor (real or complex).

        Returns:
            torch.Tensor: The output tensor after adding noise.
        """
        # Use centralized noise application function
        if self.snr_db is not None:
            return _apply_noise(x, snr_db=self.snr_db)
        else:
            return _apply_noise(x, noise_power=self.avg_noise_power)


GaussianChannel = AWGNChannel


class LaplacianChannel(BaseChannel):
    """Channel with additive Laplacian (double-exponential) noise.

    Models a channel with noise following the Laplacian distribution, which has
    heavier tails than Gaussian noise. This channel supports both real and
    complex-valued inputs.

    Mathematical Model:
        y = x + n
        where n follows a Laplacian distribution

    Args:
        scale (float, optional): Scale parameter of the Laplacian distribution
        avg_noise_power (float, optional): The average noise power
        snr_db (float, optional): SNR in dB (alternative to scale or avg_noise_power)
    """

    def __init__(self, scale=None, avg_noise_power=None, snr_db=None):
        super().__init__()

        # Handle different parameter specifications
        if scale is not None:
            self.scale = to_tensor(scale)
            self.avg_noise_power = None
            self.snr_db = None
        elif snr_db is not None:
            self.snr_db = snr_db
            self.scale = None
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.scale = None
            self.snr_db = None
        else:
            raise ValueError("Either scale, avg_noise_power, or snr_db must be provided")

    def _get_laplacian_noise(self, shape, device):
        """Generate Laplacian distributed noise."""
        u = torch.rand(shape, device=device)
        exp1 = -torch.log(u)
        exp2 = -torch.log(1 - u)
        return exp1 - exp2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add Laplacian noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor (real or complex)

        Returns:
            torch.Tensor: The output tensor with additive Laplacian noise
        """
        # Determine noise parameters
        scale = self.scale

        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(x) ** 2)
            target_noise_power = snr_to_noise_power(signal_power, self.snr_db)
            # For Laplacian distribution with zero mean, variance = 2*scale²
            scale = torch.sqrt(target_noise_power / 2)
        elif self.avg_noise_power is not None:
            scale = torch.sqrt(self.avg_noise_power / 2)

        # Handle complex input
        if torch.is_complex(x):
            noise_real = self._get_laplacian_noise(x.real.shape, x.device) * scale
            noise_imag = self._get_laplacian_noise(x.imag.shape, x.device) * scale
            noise = torch.complex(noise_real, noise_imag)
        else:
            noise = self._get_laplacian_noise(x.shape, x.device) * scale

        return x + noise


class PoissonChannel(BaseChannel):
    r"""Channel with signal-dependent Poisson noise.

    Models a channel where the output follows a Poisson distribution with
    mean proportional to the input. This is commonly used to model photon
    counting systems. For complex inputs, the magnitude is used to generate
    Poisson counts, and the phase is preserved.

    Mathematical Model:
        y ~ Poisson(λ·\|x\|)

    Args:
        rate_factor (float): Scaling factor λ for the Poisson rate
        normalize (bool): Whether to normalize output back to input scale
    """

    def __init__(self, rate_factor=1.0, normalize=False):
        super().__init__()
        if rate_factor <= 0:
            raise ValueError("Rate factor must be positive")
        self.rate_factor = to_tensor(rate_factor)
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Poisson channel to the input signal.

        Args:
            x (torch.Tensor): The input tensor (must be non-negative if real,
                              or will use magnitude if complex)

        Returns:
            torch.Tensor: The output tensor following Poisson distribution
        """
        # Handle complex input
        if torch.is_complex(x):
            magnitude = torch.abs(x)
            phase = torch.angle(x)

            # Check if magnitude is non-negative (should always be true)
            if torch.min(magnitude) < 0:
                raise ValueError("Complex magnitude should be non-negative")

            # Apply Poisson noise to magnitude
            rate = self.rate_factor * magnitude
            noisy_magnitude = torch.poisson(rate)

            # Normalize if requested
            if self.normalize:
                noisy_magnitude = noisy_magnitude / self.rate_factor

            # Reconstruct complex signal preserving phase
            return noisy_magnitude * torch.exp(1j * phase)
        else:
            if torch.min(x) < 0:
                raise ValueError("Input to PoissonChannel must be non-negative")

            # Scale the input to get the Poisson rate
            rate = self.rate_factor * x

            # Generate Poisson random values
            y = torch.poisson(rate)

            # Normalize back to input scale if requested
            if self.normalize:
                y = y / self.rate_factor

            return y


class PhaseNoiseChannel(BaseChannel):
    """Channel that introduces random phase noise.

    Models a channel where the phase of the signal is perturbed by random noise,
    which is common in oscillator circuits and synchronization.

    Mathematical Model:
        y = x * exp(j·θ)
        where θ ~ N(0, σ²) is the phase noise

    Args:
        phase_noise_std (float): Standard deviation of phase noise in radians
    """

    def __init__(self, phase_noise_std):
        super().__init__()
        if phase_noise_std < 0:
            raise ValueError("Phase noise standard deviation must be non-negative")
        self.phase_noise_std = to_tensor(phase_noise_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random phase noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor (real or complex)

        Returns:
            torch.Tensor: The output tensor with phase noise
        """
        # Convert real signal to complex if needed
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        # Generate random phase noise
        phase_noise = torch.randn_like(x.real) * self.phase_noise_std

        # Apply phase noise
        return x * torch.exp(1j * phase_noise)


class FlatFadingChannel(BaseChannel):
    """Flat fading channel with configurable distribution and coherence time.

    Models a wireless channel where the fading coefficient remains constant over
    a specified coherence time and then changes to a new independent realization.
    This represents blockwise fading commonly used in communications analysis.

    Mathematical Model:
        y[i] = h[⌊i/L⌋] * x[i] + n[i]
        where L is the coherence length, h follows a specified distribution,
        and n ~ CN(0,σ²)

    Args:
        fading_type (str): Distribution type for fading coefficients
            ('rayleigh', 'rician', or 'lognormal')
        coherence_time (int): Number of samples over which the fading coefficient
            remains constant
        k_factor (float, optional): Rician K-factor (ratio of direct to scattered power),
            used only when fading_type='rician'
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        shadow_sigma_db (float, optional): Standard deviation in dB for log-normal shadowing,
            used only when fading_type='lognormal'

    Example:
        >>> # Create a flat Rayleigh fading channel with coherence time of 10 samples
        >>> channel = FlatFadingChannel('rayleigh', coherence_time=10, snr_db=15)
        >>> x = torch.complex(torch.ones(100), torch.zeros(100))
        >>> y = channel(x)  # Output with block fading effects
    """

    def __init__(
        self,
        fading_type,
        coherence_time,
        k_factor=None,
        avg_noise_power=None,
        snr_db=None,
        shadow_sigma_db=None,
    ):
        super().__init__()

        # Validate and store fading type
        valid_types = ["rayleigh", "rician", "lognormal"]
        if fading_type not in valid_types:
            raise ValueError(f"Fading type must be one of {valid_types}")
        self.fading_type = fading_type

        # Store fading parameters
        self.coherence_time = coherence_time
        self.k_factor = to_tensor(k_factor) if k_factor is not None else None
        self.shadow_sigma_db = to_tensor(shadow_sigma_db) if shadow_sigma_db is not None else None

        # Verify required parameters based on fading type
        if fading_type == "rician" and k_factor is None:
            raise ValueError("K-factor must be provided for Rician fading")
        if fading_type == "lognormal" and shadow_sigma_db is None:
            raise ValueError("shadow_sigma_db must be provided for lognormal fading")

        # Store noise parameters
        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def _generate_fading_coefficients(self, batch_size, seq_length, device):
        """Generate fading coefficients based on the specified distribution.

        Args:
            batch_size (int): Number of independent channel realizations
            seq_length (int): Length of the input sequence
            device (torch.device): Device to create tensors on

        Returns:
            torch.Tensor: Complex fading coefficients of shape (batch_size, blocks)
                where blocks = ceil(seq_length / coherence_time)
        """
        # Calculate number of fading blocks needed
        num_blocks = (seq_length + self.coherence_time - 1) // self.coherence_time

        if self.fading_type == "rayleigh":
            # Complex Gaussian distribution for Rayleigh fading
            h_real = torch.randn(batch_size, num_blocks, device=device)
            h_imag = torch.randn(batch_size, num_blocks, device=device)
            h = torch.complex(h_real, h_imag) / np.sqrt(2)

        elif self.fading_type == "rician":
            # Rician fading with K factor
            k = self.k_factor

            # Direct component (line of sight)
            los_magnitude = torch.sqrt(k / (k + 1))
            los = los_magnitude * torch.ones(batch_size, num_blocks, device=device)

            # Scattered component
            scattered_magnitude = torch.sqrt(1 / (k + 1)) / np.sqrt(2)
            h_real = torch.randn(batch_size, num_blocks, device=device) * scattered_magnitude
            h_imag = torch.randn(batch_size, num_blocks, device=device) * scattered_magnitude
            scattered = torch.complex(h_real, h_imag)

            # Combined Rician fading
            h = torch.complex(los, torch.zeros_like(los)) + scattered

        elif self.fading_type == "lognormal":
            # Log-normal shadowing combined with Rayleigh fading
            # First generate Rayleigh component
            h_real = torch.randn(batch_size, num_blocks, device=device)
            h_imag = torch.randn(batch_size, num_blocks, device=device)
            h_rayleigh = torch.complex(h_real, h_imag) / np.sqrt(2)

            # Generate log-normal shadowing in linear scale
            sigma_ln = self.shadow_sigma_db * (np.log(10) / 10)  # Convert from dB to natural log
            ln_mean = -(sigma_ln**2) / 2  # Ensure unit mean
            shadow = torch.exp(
                torch.randn(batch_size, num_blocks, device=device) * sigma_ln + ln_mean
            )

            # Apply shadowing to fast fading component
            h = h_rayleigh * torch.complex(shadow, torch.zeros_like(shadow))

        return h

    def _expand_coefficients(self, h, seq_length):
        """Expand block fading coefficients to match input sequence length.

        Args:
            h (torch.Tensor): Block fading coefficients of shape (batch_size, num_blocks)
            seq_length (int): Target sequence length

        Returns:
            torch.Tensor: Expanded coefficients of shape (batch_size, seq_length)
        """
        batch_size = h.shape[0]
        device = h.device

        # Create indices for each position in the sequence
        block_indices = torch.arange(seq_length, device=device) // self.coherence_time

        # Expand block fading coefficients to full sequence length
        h_expanded = torch.zeros(batch_size, seq_length, dtype=h.dtype, device=device)

        for b in range(batch_size):
            h_expanded[b] = h[b, block_indices]

        return h_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply flat fading channel effects to the input signal.

        Applies block fading coefficients that remain constant over the coherence time
        and then adds complex Gaussian noise.

        Args:
            x (torch.Tensor): The input signal tensor of shape (batch_size, seq_length)
                or (batch_size, channels, seq_length)

        Returns:
            torch.Tensor: The output signal after applying fading and noise
        """
        # Handle different input shapes
        original_shape = x.shape
        if len(original_shape) > 2:
            # Reshape to (batch_size, seq_length) for processing
            x = x.reshape(original_shape[0], -1)

        # Ensure input is complex
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        batch_size, seq_length = x.shape
        device = x.device

        # Generate fading coefficients
        h_blocks = self._generate_fading_coefficients(batch_size, seq_length, device)

        # Expand to match sequence length
        h = self._expand_coefficients(h_blocks, seq_length)

        # Apply fading
        y = h * x

        # Add noise
        noise_power = self.avg_noise_power
        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(y) ** 2)
            noise_power = (
                snr_to_noise_power(signal_power, self.snr_db) * 0.5
            )  # Split between real/imag

        noise_real = torch.randn_like(y.real) * torch.sqrt(noise_power)
        noise_imag = torch.randn_like(y.imag) * torch.sqrt(noise_power)
        noise = torch.complex(noise_real, noise_imag)

        y = y + noise

        # Reshape to original dimensions if needed
        if len(original_shape) > 2:
            y = y.reshape(*original_shape)

        return y


class NonlinearChannel(BaseChannel):
    """General nonlinear channel with configurable transfer function.

    Models various nonlinear effects by applying a user-specified nonlinear function
    to the input signal, optionally followed by additive noise. Handles both real and
    complex-valued signals.

    Mathematical Model:
        y = f(x) + n
        where f is a nonlinear function and n is optional noise

    Args:
        nonlinear_fn (callable): A function that implements the nonlinear transformation
        add_noise (bool): Whether to add noise after the nonlinear operation
        avg_noise_power (float, optional): The average noise power if add_noise is True
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        complex_mode (str, optional): How to handle complex inputs: 'direct' (default)
            passes the complex signal directly to nonlinear_fn, 'cartesian' applies
            the function separately to real and imaginary parts, 'polar' applies to
            magnitude and preserves phase

    Example:
        >>> # Create a channel with cubic nonlinearity for real signals
        >>> channel = NonlinearChannel(lambda x: x + 0.2 * x**3)
        >>> x = torch.linspace(-1, 1, 100)
        >>> y = channel(x)  # Output with cubic distortion

        >>> # For complex signals, using polar mode (apply nonlinearity to magnitude only)
        >>> def mag_distortion(x): return x * (1 - 0.1 * x)  # compression
        >>> channel = NonlinearChannel(mag_distortion, complex_mode='polar')
        >>> x = torch.complex(torch.randn(100), torch.randn(100))
        >>> y = channel(x)  # Output with magnitude distortion, phase preserved
    """

    def __init__(
        self,
        nonlinear_fn,
        add_noise=False,
        avg_noise_power=None,
        snr_db=None,
        complex_mode="direct",
    ):
        super().__init__()
        self.nonlinear_fn = nonlinear_fn
        self.add_noise = add_noise
        self.complex_mode = complex_mode

        if complex_mode not in ["direct", "cartesian", "polar"]:
            raise ValueError("complex_mode must be 'direct', 'cartesian', or 'polar'")

        if add_noise:
            if snr_db is not None:
                self.snr_db = snr_db
                self.avg_noise_power = None
            elif avg_noise_power is not None:
                self.avg_noise_power = avg_noise_power
                self.snr_db = None
            else:
                raise ValueError(
                    "If add_noise=True, either avg_noise_power or snr_db must be provided"
                )
        else:
            self.avg_noise_power = None
            self.snr_db = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear transformation and optional noise to the input.

        Handles both real and complex inputs based on the complex_mode setting.

        Args:
            x (torch.Tensor): The input tensor (real or complex)

        Returns:
            torch.Tensor: The output tensor after nonlinear transformation and optional noise
        """
        # Handle complex inputs according to specified mode
        if torch.is_complex(x):
            if self.complex_mode == "direct":
                # Pass complex tensor directly to the function
                y = self.nonlinear_fn(x)

            elif self.complex_mode == "cartesian":
                # Apply nonlinearity separately to real and imaginary parts
                y_real = self.nonlinear_fn(x.real)
                y_imag = self.nonlinear_fn(x.imag)
                y = torch.complex(y_real, y_imag)

            elif self.complex_mode == "polar":
                # Apply nonlinearity to magnitude, preserve phase
                magnitude = torch.abs(x)
                phase = torch.angle(x)

                # Apply nonlinearity to magnitude
                new_magnitude = self.nonlinear_fn(magnitude)

                # Reconstruct complex signal
                y = new_magnitude * torch.exp(1j * phase)
        else:
            # For real inputs, just apply the function
            y = self.nonlinear_fn(x)

        # Add noise if requested
        if self.add_noise:
            if self.snr_db is not None:
                y = _apply_noise(y, snr_db=self.snr_db)
            else:
                y = _apply_noise(y, noise_power=self.avg_noise_power)

        return y
