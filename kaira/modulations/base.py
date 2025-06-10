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

    def forward_soft(self, x: torch.Tensor, temp: float = 1.0, *args, **kwargs) -> torch.Tensor:
        """Modulate soft bits to symbols in a differentiable manner.

        This method enables differentiability through the modulator using soft bit
        probabilities as input. Default implementation calls forward, but subclasses
        should override for true differentiability.

        Args:
            x: Input tensor of soft bit probabilities with shape (..., K*N),
               where K is bits_per_symbol. Values should be in [0, 1] range,
               representing P(bit=1).
            temp: Temperature parameter for soft decisions (lower = harder)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Modulated symbols with shape (..., N)
        """
        # Default implementation just calls forward with hard decisions
        # Subclasses should override this for true differentiability
        hard_bits = (x > 0.5).float()
        return self.forward(hard_bits, *args, **kwargs)

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

    def forward_soft(self, y: torch.Tensor, noise_var: Union[float, torch.Tensor], temp: float = 1.0, *args, **kwargs) -> torch.Tensor:
        """Demodulate symbols to soft bit probabilities in a differentiable manner.

        This method enables differentiability through the demodulator. The default
        implementation converts LLRs to probabilities, but subclasses should override
        this method if a more efficient implementation is available.

        Args:
            y: Received symbols with shape (..., N)
            noise_var: Noise variance (required)
            temp: Temperature parameter for controlling softness of decisions
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Soft bit probabilities with shape (..., N*bits_per_symbol)
            Values are in [0, 1] range, representing P(bit=1)
        """
        # Default implementation converts LLRs to probabilities
        # Subclasses can override for more efficient implementations
        llrs = self.forward(y, noise_var, *args, **kwargs)
        # Convert LLRs to probabilities with temperature scaling
        # P(bit=1) = 1 / (1 + exp(LLR / temp))
        probs = torch.sigmoid(-llrs / temp)
        return probs

    def reset_state(self) -> None:
        """Reset any stateful components.

        For demodulators with memory.
        """
        pass  # Default implementation does nothing
