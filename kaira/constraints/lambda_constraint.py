"""Lambda constraint implementation.

This module defines the LambdaConstraint which wraps arbitrary functions as signal constraints,
allowing for custom constraint implementation without creating new subclasses.
"""

from typing import Callable

import torch

from kaira.constraints.base import BaseConstraint


class LambdaConstraint(BaseConstraint):
    """Constraint that applies a user-defined function to the signal.

    This constraint allows users to pass any function that operates on tensors
    to be used as a constraint, providing flexibility without requiring new class
    implementations for simple constraints.

    Attributes:
        function (Callable): The function to apply to the input tensor
    """

    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs):
        """Initialize with a user-defined constraint function.

        Args:
            function (Callable[[torch.Tensor], torch.Tensor]): A function that takes
                a torch.Tensor as input and returns a torch.Tensor as output.
                The function should maintain tensor dimensions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.function = function

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the user-defined function to the input tensor.

        Args:
            x (torch.Tensor): The input signal tensor
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The result of applying the function to x
        """
        return self.function(x, *args, **kwargs)
