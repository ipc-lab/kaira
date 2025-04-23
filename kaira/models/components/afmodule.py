"""Attention Feature (AF) Module implementation."""

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

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the AFModule.

        Args:
            x (torch.Tensor): The input tensor.
            csi (torch.Tensor): Channel State Information tensor.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism.
        """
        input_tensor = x

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

        # Convert csi to 2D tensor if needed
        if len(csi.shape) == 1:
            csi = csi.view(batch_size, 1)
        elif len(csi.shape) > 2:
            csi = csi.flatten(start_dim=1)

        # Make sure the context and csi dimensions match what the linear layer expects
        # The first linear layer expects N + csi_length input features
        expected_context_dim = self.layers[0].in_features - csi.shape[1]

        if context.shape[1] != expected_context_dim:
            if context.shape[1] > expected_context_dim:
                # Trim extra dimensions if needed
                context = context[:, :expected_context_dim]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, expected_context_dim - context.shape[1], device=context.device)
                context = torch.cat([context, padding], dim=1)

        context_input = torch.cat([context, csi], dim=1)

        mask = self.layers(context_input)

        # Apply the mask according to input dimensions and actual channels
        if input_dims == 4:
            # Reshape mask to match the number of channels in the original AFModule config
            mask = mask.view(-1, mask.shape[1], 1, 1)

            # If input has more channels than the mask, extend the mask
            if actual_channels > mask.shape[1]:
                additional_channels = actual_channels - mask.shape[1]
                extension = torch.ones(batch_size, additional_channels, 1, 1, device=mask.device)
                mask = torch.cat([mask, extension], dim=1)
            else:
                # Trim the mask if needed
                mask = mask[:, :actual_channels, :, :]

        elif input_dims == 3:
            mask = mask.view(-1, 1, mask.shape[1])

            # If input has more channels than the mask, extend the mask
            if actual_channels > mask.shape[2]:
                additional_channels = actual_channels - mask.shape[2]
                extension = torch.ones(batch_size, 1, additional_channels, device=mask.device)
                mask = torch.cat([mask, extension], dim=2)
            else:
                # Trim the mask if needed
                mask = mask[:, :, :actual_channels]

        else:
            # If input has more features than the mask, extend the mask
            if actual_channels > mask.shape[1]:
                additional_channels = actual_channels - mask.shape[1]
                extension = torch.ones(batch_size, additional_channels, device=mask.device)
                mask = torch.cat([mask, extension], dim=1)
            else:
                # Trim the mask if needed
                mask = mask[:, :actual_channels]

        # Apply mask to the input tensor
        out = mask * input_tensor
        return out
