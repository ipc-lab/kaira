"""Identity modulation module.

This module provides identity modulators and demodulators that simply pass data through without
modification. These are useful as placeholders or for testing pipelines without actual modulation.
"""

import torch

from .base import BaseDemodulator, BaseModulator
from .registry import ModulationRegistry
from .utils import plot_constellation as utils_plot_constellation


@ModulationRegistry.register_modulator()
class IdentityModulator(BaseModulator):
    """Identity modulator that passes input data through unchanged.

    This modulator implements the BaseModulator interface but doesn't perform
    any actual modulation. It's useful as a no-op placeholder in pipelines
    or for testing.

    Attributes:
        constellation (torch.Tensor): Trivial constellation points [0, 1]
    """

    def __init__(self, *args, **kwargs):
        """Initialize the identity modulator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.constellation = self._create_constellation()

        self._bits_per_symbol = 1  # Always 1 for identity modulation

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bits to symbols as required by BaseModulator.

        Args:
            x: Input tensor of bits with shape (..., N)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The same tensor, unchanged, with shape (..., N)
        """
        return self.modulate(x)

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """Pass input bits through unchanged.

        Args:
            bits: Input tensor of bits with any shape

        Returns:
            The same tensor, unchanged
        """
        return bits

    def _create_constellation(self) -> torch.Tensor:
        """Create a trivial constellation (just 0 and 1).

        Returns:
            Complex-valued tensor of constellation points [0, 1]
        """
        return torch.tensor([0.0, 1.0], dtype=torch.complex64)

    def plot_constellation(self, **kwargs):
        """Plot the constellation diagram.

        Args:
            **kwargs: Additional arguments for plotting

        Returns:
            Matplotlib figure object
        """
        return utils_plot_constellation(self.constellation, title="Identity Constellation", labels=["0", "1"], **kwargs)


@ModulationRegistry.register_demodulator()
class IdentityDemodulator(BaseDemodulator):
    """Identity demodulator that passes input data through unchanged.

    This demodulator implements the BaseDemodulator interface but doesn't perform
    any actual demodulation. It's useful as a no-op placeholder in pipelines
    or for testing.

    Attributes:
        constellation (torch.Tensor): Trivial constellation points [0, 1]
    """

    def __init__(self, *args, **kwargs):
        """Initialize the identity demodulator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.constellation = torch.tensor([0.0, 1.0], dtype=torch.complex64)

        self._bits_per_symbol = 1  # Always 1 for identity demodulation

    def forward(self, y: torch.Tensor, noise_var=None, *args, **kwargs) -> torch.Tensor:
        """Demodulate symbols to bits as required by BaseDemodulator.

        Args:
            y: Received symbols with shape (..., N)
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns soft values;
            otherwise, returns hard bit decisions
        """
        if noise_var is not None:
            return self.soft_demodulate(y, noise_var)
        return self.demodulate(y)

    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """Pass input symbols through unchanged.

        Args:
            symbols: Input tensor with any shape

        Returns:
            The same tensor, unchanged
        """
        return symbols

    def soft_demodulate(self, symbols: torch.Tensor, noise_var: float) -> torch.Tensor:
        """Pass input symbols through unchanged, ignoring noise variance.

        For true soft demodulation, the implementation would calculate LLRs.
        This implementation simply returns the symbols unchanged.

        Args:
            symbols: Input tensor with any shape
            noise_var: Noise variance (ignored in this implementation)

        Returns:
            The same tensor, unchanged
        """
        return symbols
