"""Perfect Channel Implementation.

The perfect (identity) channel represents the theoretical ideal case in information theory
where information is transmitted without any error or loss :cite:`shannon1948mathematical`.
"""

from typing import Any

import torch

from .base import BaseChannel
from .registry import ChannelRegistry


@ChannelRegistry.register_channel()
class PerfectChannel(BaseChannel):
    """Identity channel that passes signals through unchanged.

    This channel represents an ideal communication medium with no distortion,
    noise, or interference. It simply returns the input signal as is.
    Perfect channels establish theoretical upper bounds on communication performance
    :cite:`cover2006elements` and serve as baselines in channel analysis
    :cite:`shannon1948mathematical`.

    Mathematical Model:
        y = x

    Example:
        >>> channel = PerfectChannel()
        >>> x = torch.randn(10, 1)
        >>> y = channel(x)  # y is identical to x
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the Perfect Channel.

        Args:
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Transmit signal without modification (identity operation).

        Implements an ideal noiseless, distortionless channel that perfectly
        preserves the input signal.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The input tensor without any modification.
        """
        return x


IdentityChannel = PerfectChannel
IdealChannel = PerfectChannel
