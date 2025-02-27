from abc import ABC, abstractmethod

import torch
from torch import nn


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
