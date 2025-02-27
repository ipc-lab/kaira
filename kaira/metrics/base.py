from abc import ABC, abstractmethod

import torch
from torch import nn

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
