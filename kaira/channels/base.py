"""Base Channel Module for Communication System Modeling.

This module provides the foundation for modeling communication channels in signal processing and
communications systems simulation.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseChannel(nn.Module, ABC):
    """Base abstract class for communication channel models.

    In communications theory, a channel refers to the medium through which information
    is transmitted from a sender to a receiver. This class provides a foundation for
    implementing various channel models that simulate real-world effects like noise,
    fading, distortion, and interference.

    All channel implementations should inherit from this base class and implement
    the forward method, which applies the channel effects to the input signal.

    Channel models are implemented as PyTorch modules, allowing them to be:
    - Used in computational graphs
    - Combined with neural networks
    - Run on GPUs when available
    - Included in larger end-to-end communications system models

    Example:
        >>> class MyCustomChannel(BaseChannel):
        ...     def __init__(self, param):
        ...         super().__init__()
        ...         self.param = param
        ...
        ...     def forward(self, x):
        ...         # Apply channel effects to input signal x
        ...         return x + self.param * torch.randn_like(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input signal according to channel characteristics.

        This method defines how the channel transforms an input signal,
        which may include adding noise, applying fading, introducing
        hardware impairments, or other effects specific to the channel model.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after passing through the channel.
        """
        pass
