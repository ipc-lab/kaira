import torch
import torch.nn as nn
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)

from .base import BaseModel


class DeepJSCCQEncoder(BaseModel):
    """DeepJSCCQ Encoder Module.

    This module encodes an image into a latent representation using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCQEncoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks in the g_a module.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AttentionBlock(M),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The encoded latent representation.
        """

        for layer in self.g_a:
            x = layer(x)

        return x


class DeepJSCCQDecoder(nn.Module):
    """DeepJSCCQ Decoder Module.

    This module decodes a latent representation into an image using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCQDecoder.

        Args:
            N (int): The number of input channels.
            M (int): The number of output channels.
        """
        super().__init__()

        self.g_s = nn.ModuleList(
            [
                AttentionBlock(M),
                ResidualBlock(in_ch=M, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=3, upsample=2),
            ]
        )

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The encoded latent representation.

        Returns:
            torch.Tensor: The decoded image.
        """

        for layer in self.g_s:
            x = layer(x)

        return x
