"""Lambda Channel Implementation for Custom Functions.

This module contains the LambdaChannel class which allows users to create custom channel models by
applying arbitrary functions to input signals.
"""

from typing import Any, Callable

import torch

from .base import BaseChannel
from .registry import ChannelRegistry


@ChannelRegistry.register_channel()
class LambdaChannel(BaseChannel):
    """Customizable channel that applies user-defined functions to signals.

    This channel provides a flexible way to implement custom channel behavior by
    wrapping any arbitrary function. It can be used to model specific distortions,
    transformations, or to combine multiple channel effects into a single model.

    Mathematical Model:
        y = f(x)
        where f is any user-defined function

    Args:
        fn (callable): The function to apply to the input signal.
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

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        """Initialize the Lambda channel.

        Args:
            fn (callable): The function to apply to the input signal.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the custom function to the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments passed to the custom function.
            **kwargs: Additional keyword arguments passed to the custom function.

        Returns:
            torch.Tensor: The output tensor after applying the custom function.
        """
        return self.fn(x, *args, **kwargs)
