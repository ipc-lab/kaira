from abc import ABC, abstractmethod

import torch
from torch import nn


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
