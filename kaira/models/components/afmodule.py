from typing import Any

import torch
from torch import nn

from kaira.models.base import BaseModel

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
        # Handle both tuple input and separate arguments
        if isinstance(x, tuple) and len(x) == 2:
            input_tensor, side_info = x
        else:
            # Assume x is the input tensor and side_info is the first arg
            input_tensor = x
            if args and len(args) > 0:
                side_info = args[0]
            else:
                raise ValueError("AFModule requires both input tensor and side information")
        
        # Handle different input dimensions
        input_dims = len(input_tensor.shape)
        batch_size = input_tensor.shape[0]
        
        # For 4D input (batch, channels, height, width)
        if input_dims == 4:
            context = torch.mean(input_tensor, dim=(2, 3))
        # For 3D input (batch, sequence, features)
        elif input_dims == 3:
            context = torch.mean(input_tensor, dim=1)
        # For 2D input (batch, features)
        else:
            context = input_tensor
        
        # Ensure side_info has the right shape for concatenation (batch, features)
        if len(side_info.shape) > 2:
            side_info = side_info.reshape(batch_size, -1)
            
        context_input = torch.cat([context, side_info], dim=1)
        mask = self.layers(context_input)
        
        # Apply the mask according to input dimensions
        if input_dims == 4:
            mask = mask.view(-1, self.c_in, 1, 1)
        elif input_dims == 3:
            mask = mask.view(-1, 1, self.c_in)
            
        out = mask * x
        return out
