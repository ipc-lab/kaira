from typing import Any

from kaira.models.base import BaseModel
import torch
from torch import nn

from ..registry import ModelRegistry


@ModelRegistry.register_model()
class AFModule(BaseModel):
    """
    AFModule: Attention-Feature Module :cite:`xu2021wireless`.

    This module implements a an attention mechanism that recalibrates feature maps
    by explicitly modeling interdependencies between channel state information and
    the input features. This module allows the same model to be used during training
    and testing across channels with different signal-to-noise ratio without significant
    performance degradation.
    """

    def __init__(self, N, csi_length):
        """Initialize the AFModule.

        Args:
            N (int): The number of input and output features.
            csi_length (int): The length of the channel state information.
        """
        super().__init__()

        self.c_in = N

        self.layers = nn.Sequential(
            nn.Linear(in_features=N + csi_length, out_features=N),
            nn.LeakyReLU(),
            nn.Linear(in_features=N, out_features=N),
            nn.Sigmoid(),
        )

    def forward(self, x, *args: Any, **kwargs: Any):
        """Forward pass through the AFModule.

        Args:
            x (torch.Tensor or Tuple[torch.Tensor, torch.Tensor]): The input tensor or tuple of (tensor, side_info).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output tensor after passing through the linear layer,
            normalization layer, and activation function.
        """
        x, side_info = x
        context = torch.mean(x, dim=(2, 3))

        context_input = torch.cat([context, side_info], dim=1)
        mask = self.layers(context_input).view(-1, self.c_in, 1, 1)

        out = mask * x
        return out
