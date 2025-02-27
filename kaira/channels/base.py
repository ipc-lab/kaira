from abc import ABC, abstractmethod

import torch
from torch import nn

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
