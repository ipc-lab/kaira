"""Base Pipeline Module for Kaira.

This module provides abstract base classes for building pipeline components in the Kaira framework.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

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
