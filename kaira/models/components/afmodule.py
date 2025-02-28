import torch
from torch import nn

from ..registry import ModelRegistry


@ModelRegistry.register_model()
class AFModule(nn.Module):
    """
    AFModule: Activation-Normalization-Linear Module.

    This module combines an activation function, a normalization layer, and a linear layer
    into a single module. It is a common building block in many neural network architectures.
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

    def forward(self, x):
        """Forward pass through the AFModule.

        Args:
            x (torch.Tensor): The input tensor.

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
