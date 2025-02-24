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
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function takes a tensor as input and returns a tensor as output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.
        """
        pass


# A base class for constraints and power normalizers.
class BaseConstraint(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function takes a tensor as input and returns a tensor as output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.
        """
        pass


# A base class for metrics.
class BaseMetric(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The function "forward" takes a tensor as input and returns a tensor as output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.
        """
        pass


# A base class for models.
class BaseModel(nn.Module, ABC):
    @property
    @abstractmethod
    def bandwidth_ratio(self) -> float:
        """Returns the `bandwidth_ratio` as a float value."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward function takes a tensor as input and returns a tensor as output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.
        """
        pass


# A base class for pipelines.
class BasePipeline(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The function "forward" takes a tensor as input and returns a tensor as output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.
        """
        pass
