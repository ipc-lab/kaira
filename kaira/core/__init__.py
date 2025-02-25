"""Core module for Kaira.

This module defines the core components of the Kaira library, including base classes for channels,
constraints, metrics, models, and pipelines. These base classes provide a foundation for building
custom communication system components.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

__all__ = [
    "BaseChannel",
    "BaseConstraint",
    "BaseMetric",
    "BaseModel",
    "BasePipeline",
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

    This is an abstract base class for defining communication system pipelines. Subclasses should
    implement the forward method to define the pipeline's operation.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pipeline.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after processing by the pipeline.
        """
        pass
