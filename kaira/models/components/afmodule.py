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

    def __init__(self, N, csi_length, *args: Any, **kwargs: Any):
        """Initialize the AFModule.

        Args:
            N (int): The number of input and output features.
            csi_length (int): The length of the channel state information.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

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
            *args: Additional positional arguments. The first positional argument is expected
                   to be the side_info if x is not a tuple.
            **kwargs: Additional keyword arguments (unused).

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

        # Get the actual number of channels from the input tensor
        if input_dims == 4:
            actual_channels = input_tensor.shape[1]
            context = torch.mean(input_tensor, dim=(2, 3))
        elif input_dims == 3:
            actual_channels = input_tensor.shape[2]
            context = torch.mean(input_tensor, dim=1)
        else:
            actual_channels = input_tensor.shape[1] if len(input_tensor.shape) > 1 else 1
            context = input_tensor

        # Convert side_info to 2D tensor if needed
        if len(side_info.shape) == 1:
            side_info = side_info.view(batch_size, 1)
        elif len(side_info.shape) > 2:
            side_info = side_info.flatten(start_dim=1)

        # Make sure the context and side_info dimensions match what the linear layer expects
        # The first linear layer expects N + csi_length input features
        expected_context_dim = self.layers[0].in_features - side_info.shape[1]

        if context.shape[1] != expected_context_dim:
            if context.shape[1] > expected_context_dim:
                # Trim extra dimensions if needed
                context = context[:, :expected_context_dim]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, expected_context_dim - context.shape[1], device=context.device)
                context = torch.cat([context, padding], dim=1)

        context_input = torch.cat([context, side_info], dim=1)

        mask = self.layers(context_input)

        # Apply the mask according to input dimensions and actual channels
        if input_dims == 4:
            # Reshape mask to match the actual number of channels in the input tensor
            mask = mask[:, :actual_channels]
            mask = mask.view(-1, actual_channels, 1, 1)
        elif input_dims == 3:
            mask = mask[:, :actual_channels]
            mask = mask.view(-1, 1, actual_channels)
        else:
            mask = mask[:, :actual_channels]

        # Apply mask to the input tensor
        out = mask * input_tensor
        return out
