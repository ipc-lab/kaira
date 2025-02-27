from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModulator(nn.Module, ABC):
    """Abstract base class for all modulators.

    A modulator maps bit sequences to complex symbols according to a specific
    modulation scheme. This class defines the interface that all modulator
    implementations must follow.

    Attributes:
        bits_per_symbol: Number of bits encoded in each symbol
        constellation: Complex-valued tensor of constellation points
    """

    def __init__(self, bits_per_symbol: int) -> None:
        """Initialize the modulator.

        Args:
            bits_per_symbol: Number of bits to encode in each symbol
        """
        self.bits_per_symbol = bits_per_symbol
        self.constellation = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after processing by the model.
        """
        pass


class BaseDemodulator(ABC):
    """Abstract base class for all demodulators.

    A demodulator maps received complex symbols back to bit sequences according
    to a specific demodulation scheme, which may include soft or hard decisions.

    Attributes:
        bits_per_symbol: Number of bits encoded in each symbol
        constellation: Complex-valued tensor of constellation points
    """

    def __init__(self, bits_per_symbol: int, is_soft: bool) -> None:
        """Initialize the demodulator.

        Args:
            bits_per_symbol: Number of bits encoded in each symbol
            is_soft: Determines if the demodulator uses soft decision method, i.e., returns LLRs.
        """
        self.bits_per_symbol = bits_per_symbol
