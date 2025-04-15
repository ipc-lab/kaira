"""Base classes for modulation and demodulation schemes."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn


class BaseModulator(nn.Module, ABC):
    """Abstract base class for all modulators.

    A modulator maps bit sequences to complex symbols according to a specific
    modulation scheme.

    Attributes:
        constellation: Complex-valued tensor of constellation points
    """

    def __init__(self, bits_per_symbol: Optional[int] = None, *args, **kwargs) -> None:
        """Initialize the modulator.

        Args:
            bits_per_symbol: Number of bits to encode in each symbol
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)  # Pass *args and **kwargs to parent
        self._bits_per_symbol = bits_per_symbol

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per symbol."""
        if self._bits_per_symbol is None:
            raise NotImplementedError("bits_per_symbol must be defined in subclass")
        return self._bits_per_symbol

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bits to symbols.

        Args:
            x: Input tensor of bits with shape (..., K*N), where K is bits_per_symbol
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Modulated symbols with shape (..., N)
        """
        pass

    def plot_constellation(self, **kwargs):
        """Plot the constellation diagram.

        Args:
            **kwargs: Additional arguments for plotting

        Returns:
            Matplotlib figure object
        """
        raise NotImplementedError("plot_constellation must be implemented in subclass")

    def reset_state(self) -> None:
        """Reset any stateful components.

        For modulators with memory (like differential schemes).
        """
        pass  # Default implementation does nothing


class BaseDemodulator(nn.Module, ABC):
    """Abstract base class for all demodulators.

    A demodulator maps received complex symbols back to bit sequences according to a specific
    demodulation scheme, which may include soft or hard decisions.
    """

    def __init__(self, bits_per_symbol: Optional[int] = None, *args, **kwargs) -> None:
        """Initialize the demodulator.

        Args:
            bits_per_symbol: Number of bits encoded in each symbol
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)  # Pass *args and **kwargs to parent
        self._bits_per_symbol = bits_per_symbol

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per symbol."""
        if self._bits_per_symbol is None:
            raise NotImplementedError("bits_per_symbol must be defined in subclass")
        return self._bits_per_symbol

    @abstractmethod
    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate symbols to bits or LLRs.

        Args:
            y: Received symbols with shape (..., N)
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
            with shape (..., N*bits_per_symbol)
        """
        pass

    def reset_state(self) -> None:
        """Reset any stateful components.

        For demodulators with memory.
        """
        pass  # Default implementation does nothing
