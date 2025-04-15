"""Base class for defining evaluation metrics in Kaira."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import nn


# A base class for metrics.
class BaseMetric(nn.Module, ABC):
    """Base Metric Module.

    This is an abstract base class for defining metrics to evaluate the performance of a
    communication system. Subclasses should implement the forward method to calculate the metric.
    """

    def __init__(self, name: Optional[str] = None, *args: Any, **kwargs: Any):
        """Initialize the metric.

        Args:
            name (Optional[str]): Name of the metric
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the metric.

        Args:
            x (torch.Tensor): The first input tensor (typically predictions)
            y (torch.Tensor): The second input tensor (typically targets)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The calculated metric value
        """
        pass

    def compute_with_stats(self, x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute metric with mean and standard deviation.

        Args:
            x (torch.Tensor): The first input tensor (typically predictions)
            y (torch.Tensor): The second input tensor (typically targets)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the metric
        """
        values = self.forward(x, y, *args, **kwargs)
        return values.mean(), values.std()

    def __str__(self) -> str:
        """Return string representation of the metric."""
        return f"{self.name} Metric"
