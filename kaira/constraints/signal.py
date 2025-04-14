"""Signal characteristic constraints for communication systems.

This module provides constraints related to signal characteristics such as amplitude limitations
and spectral properties. These constraints are essential for ensuring that transmitted signals
comply with hardware limitations and regulatory requirements :cite:`han2005overview` :cite:`armstrong2002peak`.
"""

import torch

from .base import BaseConstraint
from .registry import ConstraintRegistry


@ConstraintRegistry.register_constraint()
class PeakAmplitudeConstraint(BaseConstraint):
    """Enforces maximum signal amplitude by clipping values that exceed threshold.

    Limits the maximum amplitude of the signal to prevent clipping in digital-to-analog
    converters (DACs) and power amplifiers. This constraint applies a hard clipping
    operation to ensure signal values remain within the specified bounds.
    Peak amplitude constraints are critical for practical communication systems as discussed
    in :cite:`armstrong2002peak` and :cite:`jiang2008overview`.

    Attributes:
        max_amplitude (float): Maximum allowed amplitude value
    """

    def __init__(self, max_amplitude: float, *args, **kwargs) -> None:
        """Initialize the peak amplitude constraint.

        Args:
            max_amplitude (float): Maximum allowed amplitude. Signal values exceeding
                this threshold (positive or negative) will be clipped.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.max_amplitude = max_amplitude

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply peak amplitude constraint.

        Clips the input signal to ensure all values fall within the range
        [-max_amplitude, max_amplitude].

        Args:
            x (torch.Tensor): Input tensor of any shape
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Amplitude-constrained signal with the same shape as input
        """
        # Simple clipping approach
        return torch.clamp(x, -self.max_amplitude, self.max_amplitude)


@ConstraintRegistry.register_constraint()
class SpectralMaskConstraint(BaseConstraint):
    """Restricts signal frequency components to comply with regulatory spectral masks.

    Ensures the signal's spectrum complies with regulatory requirements by limiting
    the power spectral density at specific frequencies. This is particularly important
    for preventing interference with adjacent channels or frequency bands
    :cite:`weiss2004spectrum` :cite:`fcc2002revision`.

    The constraint works in the frequency domain by applying a scaling operation
    to frequency components that exceed the mask.

    Attributes:
        mask (torch.Tensor): Spectral mask defining maximum power per frequency bin
    """

    def __init__(self, mask: torch.Tensor, *args, **kwargs) -> None:
        """Initialize the spectral mask constraint.

        Args:
            mask (torch.Tensor): Spectral mask tensor defining maximum power per frequency bin.
                The shape of this tensor should match the last dimension of the input signal
                after FFT transformation.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply spectral mask constraint.

        Transforms the signal to the frequency domain, applies the spectral mask by
        scaling frequency components that exceed the mask, then transforms back to
        the time domain.

        Args:
            x (torch.Tensor): Input tensor in time domain
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Spectral mask constrained signal in time domain with the
                same shape as the input

        Note:
            This operation preserves the signal phase while scaling the magnitude
            to comply with the mask.
        """
        x_freq = torch.fft.fft(x, dim=-1)

        # Calculate power in frequency domain
        power_spectrum = torch.abs(x_freq) ** 2

        # Apply mask by scaling where needed
        excess_indices = power_spectrum > self.mask.expand_as(power_spectrum)

        if torch.any(excess_indices):
            # Scale frequency components to meet the mask
            scale_factor = torch.sqrt(self.mask / (power_spectrum + 1e-8))
            scale_factor = torch.where(excess_indices, scale_factor, torch.ones_like(scale_factor))
            x_freq = x_freq * scale_factor

        # Convert back to time domain
        return torch.fft.ifft(x_freq, dim=-1).real
