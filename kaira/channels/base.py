"""Base Channel Module for Communication System Modeling.

This module provides the foundation for modeling communication channels in signal processing and
communications systems simulation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TypeVar

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

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the base channel.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Transform input signal according to channel characteristics.

        This method defines how the channel transforms an input signal,
        which may include adding noise, applying fading, introducing
        hardware impairments, or other effects specific to the channel model.

        Args:
            x (torch.Tensor): The input signal.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

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
