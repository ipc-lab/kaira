"""Base Pipeline Module for Kaira.

This module provides abstract base classes for building pipeline components in the Kaira framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, TypeVar, Union

import torch
from torch import nn

# Type variables for better type hinting
T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class BaseStep(nn.Module, ABC):
    """Base class for a single processing step in a pipeline.

    A step is a component that transforms input data in a pipeline.
    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Process the input data.

        Args:
            x: The input to process

        Returns:
            The processed output
        """
        pass


class BaseModel(nn.Module, ABC):
    """Base class for encoder/decoder models in communication systems.

    This class defines the interface for models that can encode or decode data in communication
    system pipelines.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process the input tensor.

        Args:
            x: Input tensor to process
            **kwargs: Additional arguments specific to the model implementation

        Returns:
            Processed tensor
        """
        pass


class BaseChannel(nn.Module, ABC):
    """Base class for communication channel models.

    Channel models simulate the effects of transmission through a physical medium, such as adding
    noise, fading, or other distortions.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Process a signal through the channel.

        Args:
            x: Input signal to transmit through the channel
            **kwargs: Additional channel parameters or options

        Returns:
            Received signal after channel effects, optionally with channel state information
        """
        pass


class BaseConstraint(nn.Module, ABC):
    """Base class for signal constraints in communication systems.

    Constraints enforce certain properties on signals, such as power limitations, bandwidth
    constraints, or quantization.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply constraints to the input signal.

        Args:
            x: Input signal to constrain

        Returns:
            Constrained version of the input signal
        """
        pass


class BasePipeline(nn.Module, ABC):
    """Base Pipeline Module.

    Abstract base class for defining communication system pipelines.
    It provides the foundation for building sequential, parallel, and other pipeline types.

    Attributes:
        steps: List of processing steps or components in the pipeline.
    """

    def __init__(self):
        """Initialize the pipeline with an empty steps list."""
        super().__init__()
        self.steps = []

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Execute the pipeline on the input data.

        Args:
            input_data: The input to process through the pipeline

        Returns:
            The output after processing
        """
        pass


class ConfigurablePipeline(BasePipeline):
    """Pipeline that supports dynamically adding and removing steps.

    This class extends the basic pipeline functionality with methods to add, remove, and manage
    pipeline steps during runtime.
    """

    def add_step(self, step: Any) -> "ConfigurablePipeline":
        """Add a processing step to the pipeline.

        Args:
            step: The processing step to add

        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self

    def remove_step(self, index: int) -> "ConfigurablePipeline":
        """Remove a processing step from the pipeline.

        Args:
            index: The index of the step to remove

        Returns:
            Self for method chaining

        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        self.steps.pop(index)
        return self
