"""Core module for Kaira.

This module defines the core components of the Kaira library, including base classes for channels,
constraints, metrics, models, and pipelines. These base classes provide a foundation for building
custom communication system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List

import torch
from torch import nn

__all__ = [
    "BaseChannel",
    "BaseConstraint",
    "BaseMetric",
    "BaseModel",
    "BasePipeline",
    "BaseModulator",
    "BaseDemodulator",
]


# A base class for channel simulators.
class BaseChannel(nn.Module, ABC):
    """Base Channel Module.

    This is an abstract base class for defining communication channels. Subclasses should implement
    the forward method to simulate the effect of the channel on the input signal.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the channel.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after passing through the channel.
        """
        pass


# A base class for constraints and power normalizers.
class BaseConstraint(nn.Module, ABC):
    """Base Constraint Module.

    This is an abstract base class for defining constraints on the transmitted signal. Subclasses
    should implement the forward method to apply the constraint.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the constraint.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after applying the constraint.
        """
        pass


# A base class for metrics.
class BaseMetric(nn.Module, ABC):
    """Base Metric Module.

    This is an abstract base class for defining metrics to evaluate the performance of a
    communication system. Subclasses should implement the forward method to calculate the metric.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the metric.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The calculated metric.
        """
        pass


# A base class for models.
class BaseModel(nn.Module, ABC):
    """Base Model Module.

    This is an abstract base class for defining communication system models. Subclasses should
    implement the bandwidth_ratio and forward methods.
    """

    @property
    @abstractmethod
    def bandwidth_ratio(self) -> float:
        """Calculate the bandwidth ratio of the model.

        Returns:
            float: The bandwidth ratio.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after processing by the model.
        """
        pass


# A base class for pipelines.
class BasePipeline(nn.Module, ABC):
    """Base Pipeline Module.

    This is an abstract base class for defining communication system pipelines.
    It provides the foundation for building sequential, parallel, and other pipeline types.

    Attributes:
        steps: List of processing steps or components in the pipeline.
    """

    def __init__(self):
        """Initialize the pipeline with an empty steps list."""
        super().__init__()
        self.steps = []

    @abstractmethod
    def add_step(self, step: Callable):
        """Add a processing step to the pipeline.

        Args:
            step: A callable that processes input data

        Returns:
            The pipeline instance for method chaining
        """
        pass

    @abstractmethod
    def remove_step(self, index: int):
        """Remove a processing step from the pipeline.

        Args:
            index: The index of the step to remove

        Returns:
            The pipeline instance for method chaining
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pipeline.

        By default, this calls the run method, but subclasses may override
        this behavior for specific torch.nn.Module integration.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after processing by the pipeline.
        """
        pass


class BaseModulator(ABC):
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
    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """Modulate bit sequences into complex symbols.

        Args:
            bits: Input tensor of bits with shape (..., N) where N is the
                 number of bits to modulate (multiple of bits_per_symbol)

        Returns:
            Complex-valued tensor of modulated symbols with shape
            (..., N/bits_per_symbol)

        Raises:
            ValueError: If the input shape is not compatible with bits_per_symbol
        """
        pass

    @abstractmethod
    def _create_constellation(self) -> torch.Tensor:
        """Create the modulation constellation points.

        Returns:
            Complex-valued tensor of constellation points
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

    def __init__(self, bits_per_symbol: int) -> None:
        """Initialize the demodulator.

        Args:
            bits_per_symbol: Number of bits encoded in each symbol
        """
        self.bits_per_symbol = bits_per_symbol
        self.constellation = None

    @abstractmethod
    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """Demodulate complex symbols into bit sequences (hard decision).

        Args:
            symbols: Input tensor of complex symbols with shape (..., N)

        Returns:
            Tensor of demodulated bits with shape (..., N*bits_per_symbol)
        """
        pass

    @abstractmethod
    def soft_demodulate(self, symbols: torch.Tensor, noise_var: float) -> torch.Tensor:
        """Demodulate complex symbols into soft bit values (LLRs).

        Args:
            symbols: Input tensor of complex symbols with shape (..., N)
            noise_var: Noise variance for LLR calculation

        Returns:
            Tensor of log-likelihood ratios (LLRs) for each bit with
            shape (..., N*bits_per_symbol)
        """
        pass
