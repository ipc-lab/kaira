"""Identity constraint implementation.

This module defines the IdentityConstraint which is a passthrough constraint that does not modify
the input signal. It's useful as a no-op constraint or as a baseline for comparison.
"""

import torch

from kaira.constraints.base import BaseConstraint


class IdentityConstraint(BaseConstraint):
    """Identity constraint that returns the input signal unchanged.

    This is a simple passthrough constraint that does not modify the input signal. It can be used
    when a constraint is expected in an interface but no actual constraint should be applied.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the identity constraint.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass that returns the input tensor unchanged.

        Args:
            x (torch.Tensor): The input signal tensor
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The same input tensor x (unchanged)
        """
        return x
