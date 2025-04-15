"""Analog Channel Implementations for Continuous-Input Signals.

This module provides implementations of channels with continuous inputs, supporting both real and
complex-valued signals. These channels represent various types of noise and distortions found in
analog communication systems.

For a comprehensive overview of analog channel models, see :cite:`goldsmith2005wireless` and :cite:`proakis2007digital`.
"""

from typing import Any, Optional, Union  # Add Union

import numpy as np
import torch

from kaira.utils import snr_to_noise_power

from .base import BaseChannel
from .registry import ChannelRegistry


# Change type hints to accept torch.Tensor or float
def _apply_noise(x: torch.Tensor, noise_power: Optional[Union[float, torch.Tensor]] = None, snr_db: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
    """Add Gaussian noise to a signal with specified power or SNR.

    Automatically handles both real and complex signals by adding
    appropriate noise to each component.

    Args:
        x (torch.Tensor): The input signal (real or complex)
        noise_power (Optional[Union[float, torch.Tensor]]): The noise power to apply
        snr_db (Optional[Union[float, torch.Tensor]]): The SNR in dB (alternative to noise_power)

    Returns:
        torch.Tensor: The signal with added noise
    """
    # Calculate noise power if SNR specified
    if snr_db is not None:
        signal_power = torch.mean(torch.abs(x) ** 2)
        # Ensure snr_db is float for snr_to_noise_power if it's a tensor
        snr_db_float = snr_db.item() if isinstance(snr_db, torch.Tensor) else snr_db
        noise_power = snr_to_noise_power(signal_power, snr_db_float)

    # Validate that at least one of noise_power or snr_db was provided
    if noise_power is None:
        raise ValueError("Either noise_power or snr_db must be provided")

    # Ensure noise_power is a tensor
    if not isinstance(noise_power, torch.Tensor):
        noise_power = torch.tensor(noise_power, device=x.device, dtype=x.dtype if not torch.is_complex(x) else x.real.dtype)

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


@ChannelRegistry.register_channel()
class AWGNChannel(BaseChannel):
    """Additive white Gaussian noise (AWGN) channel for signal transmission.

    This channel adds Gaussian noise to the input signal, supporting both real
    and complex-valued inputs automatically. For complex inputs, noise is added
    to both real and imaginary components. AWGN channels are fundamental in communication
    theory and commonly used as a baseline model :cite:`proakis2007digital`.

    Mathematical Model:
        y = x + n
        where n ~ N(0, σ²) for real inputs
        or n ~ CN(0, σ²) for complex inputs

    Args:
        avg_noise_power (float, optional): The average noise power σ².
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power).

    Example:
        >>> # For real-valued signals
        >>> channel = AWGNChannel(avg_noise_power=0.1)
        >>> x_real = torch.ones(10, 1)
        >>> y_real = channel(x_real)  # Real noisy output

        >>> # For complex-valued signals (same channel works)
        >>> x_complex = torch.complex(torch.ones(10, 1), torch.zeros(10, 1))
        >>> y_complex = channel(x_complex)  # Complex noisy output
    """

    avg_noise_power: Optional[float]
    snr_db: Optional[float]

    def __init__(self, avg_noise_power: Optional[float] = None, snr_db: Optional[float] = None, *args: Any, **kwargs: Any):
        """Initialize the AWGN channel.

        Args:
            avg_noise_power (float, optional): The average noise power σ².
            snr_db (float, optional): SNR in dB (alternative to avg_noise_power).
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = avg_noise_power
            self.snr_db = None
        else:
            raise ValueError("Either avg_noise_power or snr_db must be provided")

    def forward(self, x: torch.Tensor, *args: Any, csi=None, noise=None, **kwargs: Any) -> torch.Tensor:
        """Apply AWGN to the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments (unused).
            csi (Optional[torch.Tensor]): Channel state information (unused in AWGN).
            noise (Optional[torch.Tensor]): Pre-generated noise tensor. If provided,
                this noise will be added instead of generating new noise.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor with AWGN added.
        """
        # If pre-generated noise is provided, use it
        if noise is not None:
            return x + noise

        return _apply_noise(x, snr_db=self.snr_db, noise_power=self.avg_noise_power)


GaussianChannel = AWGNChannel


@ChannelRegistry.register_channel()
class LaplacianChannel(BaseChannel):
    """Channel with additive Laplacian (double-exponential) noise.

    Models a channel with noise following the Laplacian distribution, which has
    heavier tails than Gaussian noise. This channel supports both real and
    complex-valued inputs. Laplacian noise is often used to model impulsive noise
    environments :cite:`middleton1977statistical`.

    Mathematical Model:
        y = x + n
        where n follows a Laplacian distribution

    Args:
        scale (float, optional): Scale parameter of the Laplacian distribution.
        avg_noise_power (float, optional): The average noise power.
        snr_db (float, optional): SNR in dB (alternative to scale or avg_noise_power).

    Example:
        >>> # Create a Laplacian channel with scale=0.5
        >>> channel = LaplacianChannel(scale=0.5)
        >>> x = torch.ones(10, 1)
        >>> y = channel(x)  # Output with Laplacian noise
    """

    scale: Optional[float]
    avg_noise_power: Optional[float]
    snr_db: Optional[float]

    def __init__(self, scale: Optional[float] = None, avg_noise_power: Optional[float] = None, snr_db: Optional[float] = None, *args: Any, **kwargs: Any):
        """Initialize the Laplacian channel.

        Args:
            scale (float, optional): Scale parameter of the Laplacian distribution.
            avg_noise_power (float, optional): The average noise power.
            snr_db (float, optional): SNR in dB (alternative to scale or avg_noise_power).
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # Handle different parameter specifications
        if scale is not None:
            self.scale = scale
            self.avg_noise_power = None
            self.snr_db = None
        elif snr_db is not None:
            self.snr_db = snr_db
            self.scale = None
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = avg_noise_power
            self.scale = None
            self.snr_db = None
        else:
            raise ValueError("Either scale, avg_noise_power, or snr_db must be provided")

    def _get_laplacian_noise(self, shape, device):
        """Generate Laplacian distributed noise."""
        u = torch.rand(shape, device=device)
        # Transform uniformly distributed samples to Laplacian distribution
        # using the inverse CDF method: sign(u-0.5) * -ln(1-2|u-0.5|)
        shifted_u = u - 0.5
        sign = torch.sign(shifted_u)
        abs_shifted_u = torch.abs(shifted_u)
        # Handle edge case to avoid log(0)
        safe_abs_shifted_u = torch.clamp(2 * abs_shifted_u, max=0.999999)
        raw_laplacian = sign * (-torch.log(1 - safe_abs_shifted_u))

        return raw_laplacian

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply Laplacian noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor with Laplacian noise added.
        """
        # Determine noise parameters
        scale = self.scale
        target_noise_power: Optional[Union[float, torch.Tensor]] = None  # Define target_noise_power

        if self.snr_db is not None:
            signal_power = torch.mean(torch.abs(x) ** 2)
            # Ensure snr_db is float
            snr_db_float = self.snr_db.item() if isinstance(self.snr_db, torch.Tensor) else self.snr_db
            target_noise_power = snr_to_noise_power(signal_power, snr_db_float)
            # For Laplacian distribution with zero mean, variance = 2*scale²
            scale = torch.sqrt(target_noise_power / 2)
        elif self.avg_noise_power is not None:
            # For Laplacian distribution with zero mean, variance = 2*scale²
            # Ensure avg_noise_power is float or tensor
            avg_noise_power_val = self.avg_noise_power.item() if isinstance(self.avg_noise_power, torch.Tensor) else self.avg_noise_power
            scale = torch.sqrt(torch.tensor(avg_noise_power_val / 2, device=x.device))  # Convert to tensor

        # Make sure scale is a tensor for calculations
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=x.device, dtype=x.dtype if not torch.is_complex(x) else x.real.dtype)

        # Handle complex input
        if torch.is_complex(x):
            noise_real = self._get_laplacian_noise(x.real.shape, x.device) * scale
            noise_imag = self._get_laplacian_noise(x.imag.shape, x.device) * scale
            noise = torch.complex(noise_real, noise_imag)
        else:
            noise = self._get_laplacian_noise(x.shape, x.device) * scale

        return x + noise


@ChannelRegistry.register_channel()
class PoissonChannel(BaseChannel):
    r"""Channel with signal-dependent Poisson noise.

    Models a channel where the output follows a Poisson distribution with
    mean proportional to the input. This is commonly used to model photon
    counting systems and optical communication channels :cite:`middleton1977statistical`.

    Mathematical Model:
        y ~ Poisson(λ·\|x\|)

    Args:
        rate_factor (float): Scaling factor λ for the Poisson rate.
        normalize (bool): Whether to normalize output back to input scale.

    Example:
        >>> # Create a Poisson channel with rate_factor=0.1
        >>> channel = PoissonChannel(rate_factor=0.1)
        >>> x = torch.ones(10, 1)
        >>> y = channel(x)  # Output with Poisson noise
    """

    rate_factor: float

    def __init__(self, rate_factor: float = 1.0, normalize: bool = False, *args: Any, **kwargs: Any):
        """Initialize the Poisson channel.

        Args:
            rate_factor (float): Scaling factor λ for the Poisson rate.
            normalize (bool): Whether to normalize output back to input scale.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        if rate_factor <= 0:
            raise ValueError("Rate factor must be positive")
        self.rate_factor = rate_factor
        self.normalize = normalize

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply Poisson channel to the input signal.

        Args:
            x (torch.Tensor): The input tensor (must be non-negative if real,
                              or will use magnitude if complex)
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor following Poisson distribution
        """
        # Ensure rate_factor is tensor for calculations
        rate_factor_tensor = torch.tensor(self.rate_factor, device=x.device, dtype=x.dtype if not torch.is_complex(x) else x.real.dtype)

        # Handle complex input
        if torch.is_complex(x):
            magnitude = torch.abs(x)
            # Store the phase to ensure we preserve it exactly
            phase = torch.angle(x)

            # Apply Poisson noise to magnitude
            rate = rate_factor_tensor * magnitude
            noisy_magnitude = torch.poisson(rate)

            # Normalize if requested
            if self.normalize:
                noisy_magnitude = noisy_magnitude / rate_factor_tensor

            # Reconstruct complex signal preserving exact phase
            return torch.polar(noisy_magnitude, phase)  # Uses polar form with exact phase preservation
        else:
            if torch.any(x < 0):
                raise ValueError("Input to PoissonChannel must be non-negative")

            # Scale the input to get the Poisson rate
            rate = rate_factor_tensor * x

            # Generate Poisson random values
            y = torch.poisson(rate)

            # Normalize back to input scale if requested
            if self.normalize:
                y = y / rate_factor_tensor

            return y


@ChannelRegistry.register_channel()
class PhaseNoiseChannel(BaseChannel):
    """Channel that introduces random phase noise.

    Models a channel where the phase of the signal is perturbed by random noise,
    which is common in oscillator circuits and synchronization :cite:`demir2000phase`.

    Mathematical Model:
        y = x * exp(j·θ)
        where θ ~ N(0, σ²) is the phase noise

    Args:
        phase_noise_std (float): Standard deviation of phase noise in radians.
    """

    phase_noise_std: float

    def __init__(self, phase_noise_std: float, *args: Any, **kwargs: Any):
        """Initialize the Phase Noise channel.

        Args:
            phase_noise_std (float): Standard deviation of phase noise in radians.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        if phase_noise_std < 0:
            raise ValueError("Phase noise standard deviation must be non-negative")
        self.phase_noise_std = phase_noise_std

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply phase noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor (must be complex).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor with phase noise applied.
        """
        # Ensure phase_noise_std is tensor for calculations
        phase_noise_std_tensor = torch.tensor(self.phase_noise_std, device=x.device, dtype=x.real.dtype if torch.is_complex(x) else x.dtype)

        # Convert real signal to complex if needed
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        # Generate random phase noise with controlled standard deviation
        phase_noise = torch.randn_like(x.real) * phase_noise_std_tensor

        return x * torch.exp(1j * phase_noise)


@ChannelRegistry.register_channel()
class FlatFadingChannel(BaseChannel):
    """Flat fading channel with configurable distribution and coherence time.

    Models a wireless channel where the fading coefficient remains constant over
    a specified coherence time and then changes to a new independent realization.
    This represents blockwise fading commonly used in communications analysis
    :cite:`tse2005fundamentals` :cite:`rappaport2024wireless`.

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

    k_factor: Optional[float]
    avg_noise_power: Optional[float]
    snr_db: Optional[float]
    shadow_sigma_db: Optional[float]

    def __init__(
        self,
        fading_type: str,
        coherence_time: int,
        k_factor: Optional[float] = None,
        avg_noise_power: Optional[float] = None,
        snr_db: Optional[float] = None,
        shadow_sigma_db: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Flat Fading channel.

        Args:
            fading_type (str): Distribution type ('rayleigh', 'rician', 'lognormal').
            coherence_time (int): Samples over which fading is constant.
            k_factor (float, optional): Rician K-factor (for 'rician').
            avg_noise_power (float, optional): Average noise power σ².
            snr_db (float, optional): SNR in dB (alternative to avg_noise_power).
            shadow_sigma_db (float, optional): Shadowing std dev in dB (for 'lognormal').
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # Validate and store fading type
        valid_types = ["rayleigh", "rician", "lognormal"]
        if fading_type not in valid_types:
            raise ValueError(f"Fading type must be one of {valid_types}")
        self.fading_type = fading_type

        # Store fading parameters
        self.coherence_time = coherence_time
        self.k_factor = k_factor
        self.shadow_sigma_db = shadow_sigma_db

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
            self.avg_noise_power = avg_noise_power
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
            # Ensure k_factor is tensor for calculations
            if self.k_factor is None:
                raise ValueError("K-factor must be provided for Rician fading")
            k = torch.tensor(self.k_factor, device=device)

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
            # Ensure shadow_sigma_db is tensor for calculations
            if self.shadow_sigma_db is None:
                raise ValueError("shadow_sigma_db must be provided for lognormal fading")
            shadow_sigma_db_tensor = torch.tensor(self.shadow_sigma_db, device=device)
            sigma_ln = shadow_sigma_db_tensor * (np.log(10) / 10)  # Convert from dB to natural log
            ln_mean = -(sigma_ln**2) / 2  # Ensure unit mean
            shadow = torch.exp(torch.randn(batch_size, num_blocks, device=device) * sigma_ln + ln_mean)

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

    def forward(self, x: torch.Tensor, *args: Any, csi=None, noise=None, **kwargs: Any) -> torch.Tensor:
        """Apply flat fading and noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments (unused).
            csi (Optional[torch.Tensor]): Pre-computed channel state information (fading coefficients).
                                         If provided, these coefficients are used instead of generating new ones.
            noise (Optional[torch.Tensor]): Pre-generated noise tensor. If provided, this noise is added.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor after applying fading and noise.
        """
        # Handle different input shapes
        original_shape = x.shape
        is_1d = len(original_shape) == 1

        if is_1d:
            # Handle 1D inputs by adding a batch dimension
            x = x.unsqueeze(0)

        if len(x.shape) > 2:
            # Reshape to (batch_size, seq_length) for processing
            x = x.reshape(x.shape[0], -1)

        # Ensure input is complex
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        batch_size, seq_length = x.shape
        device = x.device

        # Use provided CSI if available, otherwise generate fading coefficients
        if csi is not None:
            # Use the provided CSI
            h = csi
        else:
            # Generate fading coefficients
            h_blocks = self._generate_fading_coefficients(batch_size, seq_length, device)
            # Expand to match sequence length
            h = self._expand_coefficients(h_blocks, seq_length)

        # Apply fading
        y = h * x

        # Add noise if provided, otherwise generate it
        if noise is not None:
            y = y + noise
        else:
            # Determine noise power
            noise_power_val: Union[float, torch.Tensor]  # Type hint for clarity
            if self.snr_db is not None:
                signal_power = torch.mean(torch.abs(y) ** 2)
                # self.snr_db is guaranteed to be float by __init__
                noise_power_val = snr_to_noise_power(signal_power, self.snr_db)
            elif self.avg_noise_power is not None:
                # self.avg_noise_power is guaranteed to be float by __init__
                noise_power_val = self.avg_noise_power
            else:
                # This case should be prevented by __init__ validation
                raise ValueError("Noise parameters not properly initialized.")  # Should not happen

            # Ensure noise_power is tensor for calculations
            if not isinstance(noise_power_val, torch.Tensor):
                noise_power_tensor = torch.tensor(noise_power_val, device=device, dtype=y.dtype if not torch.is_complex(y) else y.real.dtype)
            else:
                noise_power_tensor = noise_power_val  # Already a tensor

            # Split noise power between real and imaginary components
            component_noise_power = noise_power_tensor * 0.5
            noise_real = torch.randn_like(y.real) * torch.sqrt(component_noise_power)
            noise_imag = torch.randn_like(y.imag) * torch.sqrt(component_noise_power)
            noise = torch.complex(noise_real, noise_imag)
            y = y + noise

        # Reshape to original dimensions if needed
        if len(original_shape) > 2:
            y = y.reshape(*original_shape)
        elif is_1d:
            # Remove the batch dimension we added for 1D inputs
            y = y.squeeze(0)

        return y


@ChannelRegistry.register_channel()
class NonlinearChannel(BaseChannel):
    """General nonlinear channel with configurable transfer function.

    Models various nonlinear effects by applying a user-specified nonlinear function
    to the input signal, optionally followed by additive noise. Handles both real and
    complex-valued signals. Common nonlinear models include the Saleh model for traveling-wave
    tube amplifiers :cite:`saleh1981frequency`.

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

    avg_noise_power: Optional[float]
    snr_db: Optional[float]

    def __init__(
        self,
        nonlinear_fn,
        add_noise=False,
        avg_noise_power: Optional[float] = None,
        snr_db: Optional[float] = None,
        complex_mode="direct",
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Nonlinear channel.

        Args:
            nonlinear_fn (callable): The nonlinear transformation function.
            add_noise (bool): Whether to add noise after nonlinearity.
            avg_noise_power (float, optional): Average noise power if add_noise=True.
            snr_db (float, optional): SNR in dB (alternative if add_noise=True).
            complex_mode (str): How to handle complex inputs ('direct', 'cartesian', 'polar').
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.nonlinear_fn = nonlinear_fn
        self.add_noise = add_noise
        self.complex_mode = complex_mode

        if complex_mode not in ["direct", "cartesian", "polar"]:
            raise ValueError("complex_mode must be 'direct', 'cartesian', or 'polar'")

        if add_noise:
            if snr_db is not None and avg_noise_power is not None:
                raise ValueError("Cannot specify both snr_db and avg_noise_power")
            elif snr_db is not None:
                self.snr_db = snr_db
                self.avg_noise_power = None
            elif avg_noise_power is not None:
                self.avg_noise_power = avg_noise_power
                self.snr_db = None
            else:
                raise ValueError("If add_noise=True, either avg_noise_power or snr_db must be provided")
        else:
            self.avg_noise_power = None
            self.snr_db = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the nonlinear function and optional noise to the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor after applying nonlinearity and noise.
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
            # Check if avg_noise_power or snr_db is None before passing to _apply_noise
            # _apply_noise already handles the case where one is None
            y = _apply_noise(y, snr_db=self.snr_db, noise_power=self.avg_noise_power)

        return y


@ChannelRegistry.register_channel()
class RayleighFadingChannel(FlatFadingChannel):
    """Specialized channel for Rayleigh fading in wireless communications.

    This is a convenience class that creates a FlatFadingChannel with the
    fading_type set to "rayleigh" to model Rayleigh fading, which is common in
    non-line-of-sight wireless propagation environments.

    Mathematical Model:
        y[i] = h[⌊i/L⌋] * x[i] + n[i]
        where L is the coherence length, h follows a Rayleigh distribution,
        and n ~ CN(0,σ²)

    Args:
        coherence_time (int, optional): Number of samples over which the fading coefficient
            remains constant. Defaults to 1.
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> # Create a Rayleigh fading channel with coherence time of 10 samples
        >>> channel = RayleighFadingChannel(coherence_time=10, snr_db=15)
        >>> x = torch.complex(torch.ones(100), torch.zeros(100))
        >>> y = channel(x)  # Output with Rayleigh fading
    """

    def __init__(
        self,
        coherence_time=1,
        avg_noise_power: Optional[float] = None,
        snr_db: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Rayleigh Fading channel.

        Args:
            coherence_time (int, optional): Samples over which fading is constant. Defaults to 1.
            avg_noise_power (float, optional): Average noise power σ².
            snr_db (float, optional): SNR in dB (alternative to avg_noise_power).
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        kwargs = kwargs.copy()
        kwargs["coherence_time"] = coherence_time
        kwargs["avg_noise_power"] = avg_noise_power
        kwargs["snr_db"] = snr_db
        kwargs["fading_type"] = "rayleigh"
        super().__init__(*args, **kwargs)


@ChannelRegistry.register_channel()
class RicianFadingChannel(FlatFadingChannel):
    """Rician fading channel with configurable K-factor and coherence time.

    A specialized version of FlatFadingChannel that uses Rician fading.
    Suitable for modeling wireless channels with a dominant direct path plus
    multiple weaker reflection paths.

    Mathematical Model:
        y = h*x + n
        where h follows a Rician distribution with K-factor and n ~ CN(0,σ²)

    The K-factor represents the ratio of power in the direct path to the power
    in the scattered paths. Higher K values indicate a stronger line-of-sight component.

    Args:
        k_factor (float): Rician K-factor (ratio of direct to scattered power)
        coherence_time (int): Number of samples over which the fading coefficient
            remains constant
        avg_noise_power (float, optional): The average noise power
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> # Create a Rician channel with K=5 (strong direct path)
        >>> channel = RicianFadingChannel(k_factor=5, coherence_time=10, snr_db=15)
        >>> x = torch.complex(torch.ones(100), torch.zeros(100))
        >>> y = channel(x)  # Output with Rician fading
    """

    def __init__(
        self,
        k_factor: float = 1.0,
        coherence_time=1,
        avg_noise_power: Optional[float] = None,
        snr_db: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Rician Fading channel.

        Args:
            k_factor (float): Rician K-factor. Defaults to 1.0.
            coherence_time (int): Samples over which fading is constant. Defaults to 1.
            avg_noise_power (float, optional): Average noise power.
            snr_db (float, optional): SNR in dB (alternative to avg_noise_power).
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        # Validate k_factor is non-negative before passing to parent
        if k_factor < 0:
            raise ValueError("K-factor must be non-negative")

        kwargs = kwargs.copy()
        kwargs["k_factor"] = k_factor
        kwargs["coherence_time"] = coherence_time
        kwargs["avg_noise_power"] = avg_noise_power
        kwargs["snr_db"] = snr_db
        kwargs["fading_type"] = "rician"
        super().__init__(*args, **kwargs)


@ChannelRegistry.register_channel()
class LogNormalFadingChannel(FlatFadingChannel):
    """Log-normal fading channel with configurable shadowing standard deviation.

    A specialized version of FlatFadingChannel that uses log-normal fading.
    Suitable for modeling large-scale shadowing effects in wireless channels
    where obstacles like buildings, terrain, and foliage cause signal power variations.

    Mathematical Model:
        y = h*x + n
        where h includes log-normal shadowing and n ~ CN(0,σ²)

    The shadowing standard deviation (shadow_sigma_db) controls the variability
    of the fading. Higher values lead to more severe shadowing effects.

    Args:
        shadow_sigma_db (float): Standard deviation in dB for log-normal shadowing
        coherence_time (int): Number of samples over which the fading coefficient
            remains constant
        avg_noise_power (float, optional): The average noise power
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> # Create a log-normal shadowing channel with 8 dB standard deviation
        >>> channel = LogNormalFadingChannel(shadow_sigma_db=8.0, coherence_time=100, snr_db=15)
        >>> x = torch.complex(torch.ones(1000), torch.zeros(1000))
        >>> y = channel(x)  # Output with log-normal shadowing
    """

    def __init__(
        self,
        shadow_sigma_db: float = 4.0,
        coherence_time=100,
        avg_noise_power: Optional[float] = None,
        snr_db: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Log-Normal Fading channel.

        Args:
            shadow_sigma_db (float): Shadowing std dev in dB. Defaults to 4.0.
            coherence_time (int): Samples over which fading is constant. Defaults to 100.
            avg_noise_power (float, optional): Average noise power.
            snr_db (float, optional): SNR in dB (alternative to avg_noise_power).
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        # Validate shadow_sigma_db is non-negative
        if shadow_sigma_db < 0:
            raise ValueError("shadow_sigma_db must be non-negative")

        kwargs = kwargs.copy()
        kwargs["shadow_sigma_db"] = shadow_sigma_db
        kwargs["coherence_time"] = coherence_time
        kwargs["avg_noise_power"] = avg_noise_power
        kwargs["snr_db"] = snr_db
        kwargs["fading_type"] = "lognormal"
        super().__init__(*args, **kwargs)
