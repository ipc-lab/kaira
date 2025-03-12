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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns the input tensor unchanged.

        Args:
            x (torch.Tensor): The input signal tensor

        Returns:
            torch.Tensor: The same input tensor x (unchanged)
        """
        return x
