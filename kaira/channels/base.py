"""Base Channel Module for Communication System Modeling.

This module provides the foundation for modeling communication channels in signal processing and
communications systems simulation.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, TypeVar

import torch
from torch import nn

T = TypeVar("T", bound="BaseChannel")


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

    def get_config(self) -> Dict[str, Any]:
        """Get a dictionary of the channel's configuration.

        This method returns a dictionary containing the channel's parameters,
        which can be used to recreate the channel instance.

        Returns:
            Dict[str, Any]: Dictionary of parameter names and values
        """
        config = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                config[key] = value
        return config


class LambdaChannel(BaseChannel):
    """Customizable channel that applies user-defined functions to signals.

    This channel provides a flexible way to implement custom channel behavior by
    wrapping any arbitrary function. It can be used to model specific distortions,
    transformations, or to combine multiple channel effects into a single model.

    Mathematical Model:
        y = f(x)
        where f is any user-defined function

    Args:
        fn (Callable): The function to apply to the input signal.
            Must accept a torch.Tensor and return a torch.Tensor of compatible shape.

    Example:
        >>> # Create a custom channel that doubles the amplitude
        >>> amplifier = LambdaChannel(lambda x: 2 * x)
        >>> x = torch.ones(10)
        >>> y = amplifier(x)  # y will contain all 2's

        >>> # Create a channel that adds specific frequency distortion
        >>> def distort(x):
        ...     return x + 0.1 * torch.sin(2 * math.pi * 0.05 * torch.arange(len(x)))
        >>> channel = LambdaChannel(distort)
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input signal using the user-defined function.

        Args:
            x (torch.Tensor): Input signal tensor

        Returns:
            torch.Tensor: Transformed output signal
        """
        return self.fn(x)
