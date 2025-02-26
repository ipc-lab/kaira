"""Perfect Channel Implementation."""

import torch

from kaira.core import BaseChannel


class PerfectChannel(BaseChannel):
    """Perfect (identity) channel that passes signals unchanged.

    This channel represents an ideal communication medium with no distortion,
    noise, or interference. It simply returns the input signal as is.

    Mathematical Model:
        y = x

    Example:
        >>> channel = PerfectChannel()
        >>> x = torch.randn(10, 1)
        >>> y = channel(x)  # y is identical to x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass of the perfect channel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor without any modification.
        """
        return x
