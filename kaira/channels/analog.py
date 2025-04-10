"""Analog Channel Implementations for Continuous-Input Signals.

This module provides implementations of channels with continuous inputs, supporting both real and
complex-valued signals. These channels represent various types of noise and distortions found in
analog communication systems.

For a comprehensive overview of analog channel models, see :cite:`goldsmith2005wireless` and :cite:`proakis2007digital`.
"""

import numpy as np
import torch

from kaira.utils import snr_to_noise_power, to_tensor

from .base import BaseChannel
from .registry import ChannelRegistry


def _apply_noise(x: torch.Tensor, noise_power=None, snr_db=None) -> torch.Tensor:
    """Add AWGN to a signal with specified noise power or SNR.

    Args:
        x: Input signal tensor
        noise_power: Noise power (variance)
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Signal with noise added

    Raises:
        ValueError: If neither noise_power nor snr_db is specified
    """
    if noise_power is None and snr_db is None:
        raise ValueError("Either noise_power or snr_db must be specified")

    # If snr_db is provided, calculate the corresponding noise power
    if snr_db is not None:
        signal_power = torch.mean(torch.abs(x) ** 2)
        noise_power = snr_to_noise_power(signal_power, snr_db)

    # Generate and add noise based on whether the input is real or complex
    if torch.is_complex(x):
        # For complex signals, split noise power between real and imaginary parts
        component_noise_power = noise_power * 0.5
        noise_real = torch.randn_like(x.real) * torch.sqrt(component_noise_power)
        noise_imag = torch.randn_like(x.imag) * torch.sqrt(component_noise_power)
        noise = torch.complex(noise_real, noise_imag)
    else:
        # For real signals, use the full noise power
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

    return x + noise


@ChannelRegistry.register_channel()
class AWGNChannel(BaseChannel):
    """Additive White Gaussian Noise (AWGN) channel.

    Adds Gaussian noise to the input signal. The noise power can be specified
    directly or through the SNR in dB. Supports both real and complex-valued signals.

    Mathematical Model:
        y = x + n
        where n ~ N(0,σ²) for real signals or n ~ CN(0,σ²) for complex signals

    Args:
        avg_noise_power (float, optional): The average noise power (variance)
        snr_db (float, optional): The signal-to-noise ratio in dB

    Example:
        >>> # For real-valued signals
        >>> channel = AWGNChannel(snr_db=15)
        >>> x = torch.ones(10, 1)
        >>> y = channel(x)  # Noisy output

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

    def forward(self, x: torch.Tensor, csi=None, noise=None) -> torch.Tensor:
        """Pass signal through the AWGN channel.

        Args:
            x: Input signal
            csi: Channel state information (unused in AWGN)
            noise: Optional pre-generated noise to add instead of generating new noise

        Returns:
            Noisy signal y = x + n
        """
        if noise is not None:
            # Use provided noise
            return x + noise

        # Otherwise, generate and add noise using the helper function
        if self.avg_noise_power is not None:
            return _apply_noise(x, noise_power=self.avg_noise_power)
        else:
            return _apply_noise(x, snr_db=self.snr_db)

    def __repr__(self):
        if self.avg_noise_power is not None:
            return f"AWGNChannel(avg_noise_power={self.avg_noise_power})"
        else:
            return f"AWGNChannel(snr_db={self.snr_db})"


# Register GaussianChannel as an alias for AWGNChannel
GaussianChannel = AWGNChannel


@ChannelRegistry.register_channel()
class LaplacianChannel(BaseChannel):
    """Additive Laplacian Noise channel.

    Adds Laplacian (double exponential) noise to the input signal. The noise scale
    can be specified directly or through the average noise power or SNR in dB.
    Supports both real and complex-valued signals.

    Mathematical Model:
        y = x + n
        where n follows a Laplacian distribution with mean 0 and scale parameter b

    Args:
        scale (float, optional): The scale parameter of the Laplacian distribution
        avg_noise_power (float, optional): The average noise power
        snr_db (float, optional): The signal-to-noise ratio in dB

    Example:
        >>> # For real-valued signals
        >>> channel = LaplacianChannel(scale=0.1)
        >>> x = torch.ones(10, 1)
        >>> y = channel(x)  # Noisy output

        >>> # Using SNR specification
        >>> channel = LaplacianChannel(snr_db=15)
        >>> x = torch.randn(10, 1)
        >>> y = channel(x)  # Noisy output with specified SNR
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
        """Generate Laplacian distributed noise of a specified shape.

        Uses the difference of two exponential random variables method.
        """
        u1 = torch.rand(shape, device=device)
        u2 = torch.rand(shape, device=device)
        return torch.log(u1) - torch.log(u2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass signal through the Laplacian noise channel.

        Args:
            x: Input signal

        Returns:
            Noisy signal y = x + n, where n follows a Laplacian distribution
        """
        # Determine scale parameter based on initialization method
        if self.scale is not None:
            scale = self.scale
        elif self.avg_noise_power is not None:
            # For Laplacian distribution, variance = 2*scale^2
            scale = torch.sqrt(self.avg_noise_power / 2)
        elif self.snr_db is not None:
            # Calculate scale based on the signal power and desired SNR
            signal_power = torch.mean(torch.abs(x) ** 2)
            noise_power = snr_to_noise_power(signal_power, self.snr_db)
            scale = torch.sqrt(noise_power / 2)

        # Generate and apply Laplacian noise based on signal type
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
        """Pass signal through the Poisson channel.

        Args:
            x: Input signal

        Returns:
            Output following a Poisson distribution with rate proportional to input

        Raises:
            ValueError: If input contains negative values
        """
        with torch.no_grad():
            # Check for negative values
            if (x < 0).any():
                raise ValueError("Input to PoissonChannel must be non-negative")

            # Scale the input to get the Poisson rate
            rate = self.rate_factor * x

            # Generate Poisson random values
            y = torch.poisson(rate)

            # Normalize back to input scale if requested
            if self.normalize:
                y = y / self.rate_factor

            return y


@ChannelRegistry.register_channel()
class PhaseNoiseChannel(BaseChannel):
    """Channel with phase noise.

    Models a channel that adds phase noise to complex signals, representing
    impairments in oscillators and radio frequency components :cite:`demir2000phase`.
    Supports various phase noise models including Gaussian, Wiener process,
    and von Mises distributions.

    Mathematical Model:
        y = x·exp(jθ)
        where θ follows a specified distribution

    Args:
        std (float): Standard deviation of phase noise (radians)
        model (str): Phase noise model ('gaussian', 'wiener', 'vonmises')
        correlation (float): Parameter controlling temporal correlation (0-1)
            for the Wiener process model
    """

    def __init__(self, std=0.1, model="gaussian", correlation=None):
        super().__init__()
        self.std = to_tensor(std)
        
        valid_models = ["gaussian", "wiener", "vonmises"]
        if model not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        self.model = model
        
        if model == "wiener" and correlation is None:
            correlation = 0.9  # Default correlation for Wiener model
        self.correlation = to_tensor(correlation) if correlation is not None else None
        
        # State for Wiener process model
        self._prev_phase = None

    def _generate_gaussian_phase_noise(self, shape, device):
        """Generate Gaussian distributed phase noise."""
        return torch.randn(shape, device=device) * self.std

    def _generate_wiener_phase_noise(self, shape, device):
        """Generate Wiener process phase noise with temporal correlation."""
        # Initialize or reset state if shape changes
        if self._prev_phase is None or self._prev_phase.shape != shape:
            self._prev_phase = torch.zeros(shape, device=device)
            
        # Generate new phase based on previous phase and innovation
        innovation = torch.randn(shape, device=device) * self.std * torch.sqrt(1 - self.correlation**2)
        new_phase = self.correlation * self._prev_phase + innovation
        
        # Update state
        self._prev_phase = new_phase
        
        return new_phase

    def _generate_vonmises_phase_noise(self, shape, device):
        """Generate von Mises distributed phase noise."""
        # For simplicity, approximate via rejection sampling from wrapped normal
        return self._generate_gaussian_phase_noise(shape, device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass signal through the phase noise channel.

        Args:
            x: Input signal

        Returns:
            Signal with phase noise: y = x·exp(jθ)

        Raises:
            ValueError: If input is not a complex tensor
        """
        if not torch.is_complex(x):
            raise ValueError("PhaseNoiseChannel requires complex input")
            
        # Generate phase noise based on selected model
        if self.model == "gaussian":
            phase_noise = self._generate_gaussian_phase_noise(x.shape, x.device)
        elif self.model == "wiener":
            phase_noise = self._generate_wiener_phase_noise(x.shape, x.device)
        elif self.model == "vonmises":
            phase_noise = self._generate_vonmises_phase_noise(x.shape, x.device)
            
        # Apply phase noise to complex signal
        phase_term = torch.exp(1j * phase_noise)
        return x * phase_term


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
        """Generate fading coefficients according to the specified distribution.

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            device: Device for tensor creation

        Returns:
            Complex tensor of fading coefficients with shape [batch_size, num_blocks]
        """
        # Calculate number of fading blocks
        num_blocks = (seq_length + self.coherence_time - 1) // self.coherence_time

        # Generate appropriate fading coefficients based on type
        if self.fading_type == "rayleigh":
            # Rayleigh fading - complex Gaussian with zero mean and unit variance
            h_real = torch.randn(batch_size, num_blocks, device=device) / np.sqrt(2)
            h_imag = torch.randn(batch_size, num_blocks, device=device) / np.sqrt(2)
            h = torch.complex(h_real, h_imag)

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
            shadow = torch.exp(torch.randn(batch_size, num_blocks, device=device) * sigma_ln + ln_mean)

            # Apply shadowing to fast fading component
            h = h_rayleigh * torch.complex(shadow, torch.zeros_like(shadow))

        return h

    def _expand_coefficients(self, h, seq_length):
        """Expand block fading coefficients to match sequence length.

        Args:
            h: Fading coefficients tensor [batch_size, num_blocks]
            seq_length: Desired sequence length

        Returns:
            Expanded coefficients tensor [batch_size, seq_length]
        """
        batch_size, num_blocks = h.shape
        
        # Create indices for each sample in the sequence
        block_idx = torch.div(
            torch.arange(seq_length, device=h.device), 
            self.coherence_time,
            rounding_mode='floor'
        )

        # Limit indices to available blocks (in case seq_length requires more blocks)
        block_idx = torch.clamp(block_idx, max=num_blocks-1)
        
        # Expand fading coefficients to full sequence length
        h_expanded = torch.zeros((batch_size, seq_length), device=h.device, dtype=torch.complex64)
        for i in range(batch_size):
            h_expanded[i] = h[i, block_idx]
        
        return h_expanded

    def forward(self, x: torch.Tensor, csi=None, noise=None) -> torch.Tensor:
        """Pass signal through the fading channel.

        Args:
            x: Input signal
            csi: Optional channel state information (fading coefficients)
            noise: Optional pre-generated noise to add

        Returns:
            Faded and noisy signal y = h*x + n
        """
        # Handle different input formats
        is_1d = x.dim() == 1
        original_shape = x.shape
        
        # Convert 1D inputs to have a batch dimension for consistent processing
        if is_1d:
            x = x.unsqueeze(0)
            
        # Reshape to (batch_size, seq_length) for processing
        if x.dim() > 2:
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
            noise_power = self.avg_noise_power
            if self.snr_db is not None:
                signal_power = torch.mean(torch.abs(y) ** 2)
                noise_power = snr_to_noise_power(signal_power, self.snr_db)
            
            # Split noise power between real and imaginary components
            component_noise_power = noise_power * 0.5
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass signal through the nonlinear channel.

        Args:
            x: Input signal

        Returns:
            Nonlinearly transformed signal, optionally with noise
        """
        # Apply the nonlinear function based on input type and complex mode
        if torch.is_complex(x):
            if self.complex_mode == "direct":
                # Directly apply the function to complex input
                y = self.nonlinear_fn(x)
            elif self.complex_mode == "cartesian":
                # Apply to real and imaginary parts separately
                real_part = self.nonlinear_fn(x.real)
                imag_part = self.nonlinear_fn(x.imag)
                y = torch.complex(real_part, imag_part)
            elif self.complex_mode == "polar":
                # Apply to magnitude, preserve phase
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


@ChannelRegistry.register_channel()
class RayleighFadingChannel(FlatFadingChannel):
    """Rayleigh fading channel with configurable coherence time.

    A specialized version of FlatFadingChannel that uses Rayleigh fading.
    Suitable for modeling wireless channels in non-line-of-sight environments
    where multiple reflective paths exist.

    Mathematical Model:
        y = h*x + n
        where h follows a Rayleigh distribution and n ~ CN(0,σ²)

    Args:
        coherence_time (int): Number of samples over which the fading coefficient
            remains constant
        avg_noise_power (float, optional): The average noise power
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)

    Example:
        >>> channel = RayleighFadingChannel(coherence_time=10, snr_db=15)
        >>> x = torch.complex(torch.ones(100), torch.zeros(100))
        >>> y = channel(x)  # Output with Rayleigh fading
    """
    
    def __init__(
        self,
        coherence_time=1,
        avg_noise_power=None,
        snr_db=None,
    ):
        # If neither noise parameter is provided, use a default SNR of 15 dB
        if avg_noise_power is None and snr_db is None:
            snr_db = 15.0
            
        super().__init__(
            fading_type="rayleigh",
            coherence_time=coherence_time,
            avg_noise_power=avg_noise_power,
            snr_db=snr_db
        )


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
        k_factor=1.0,
        coherence_time=1,
        avg_noise_power=None,
        snr_db=None,
    ):
        # If neither noise parameter is provided, use a default SNR of 15 dB
        if avg_noise_power is None and snr_db is None:
            snr_db = 15.0
            
        super().__init__(
            fading_type="rician",
            coherence_time=coherence_time,
            k_factor=k_factor,
            avg_noise_power=avg_noise_power,
            snr_db=snr_db
        )


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
        shadow_sigma_db=4.0,
        coherence_time=100,
        avg_noise_power=None,
        snr_db=None,
    ):
        # If neither noise parameter is provided, use a default SNR of 15 dB
        if avg_noise_power is None and snr_db is None:
            snr_db = 15.0
            
        super().__init__(
            fading_type="lognormal",
            coherence_time=coherence_time,
            shadow_sigma_db=shadow_sigma_db,
            avg_noise_power=avg_noise_power,
            snr_db=snr_db
        )
