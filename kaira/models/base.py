"""Base Model Module for Kaira.

This module provides abstract base classes for building model components in the Kaira framework.
"""

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

class BaseModel(nn.Module, ABC):
    """Base Model Module.

    Abstract base class for defining communication system models.
    It provides the foundation for building various model types.

    Attributes:
        steps: List of processing steps or components in the model.
    """

    def __init__(self):
        """Initialize the model with an empty steps list."""
        super().__init__()
        self.steps = []

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """Execute the model on the input data.

        Args:
            input_data: The input to process through the model

        Returns:
            The output after processing
        """
        pass


class ConfigurableModel(BaseModel):
    """Model that supports dynamically adding and removing steps.

    This class extends the basic model functionality with methods to add, remove, and manage
    model steps during runtime.
    """

    def add_step(self, step: Any) -> "ConfigurableModel":
        """Add a processing step to the model.

        Args:
            step: The processing step to add

        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self

    def remove_step(self, index: int) -> "ConfigurableModel":
        """Remove a processing step from the model.

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
