import torch
import torch.nn as nn
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)

from kaira.core import BaseModel


class DeepJSCCQEncoder(BaseModel):
    def __init__(self, N: int, M: int) -> None:
        """The function initializes a neural network model with a series of residual blocks and
        attention blocks.

        Parameters
        ----------
        N : int
            The parameter N represents the number of output channels for the ResidualBlocks in the g_a
        module. It is an integer value.
        M : int
            The parameter M represents the number of output channels in the last convolutional layer of the
        network.
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
        """The forward function applies a series of layers to the input tensor and returns the
        final output.

        Parameters
        ----------
        x : torch.Tensor
            The parameter `x` is a tensor of type `torch.Tensor`.

        Returns
        -------
            the tensor `x` after passing it through all the layers in the encoder
        """

        for layer in self.g_a:
            x = layer(x)

        return x


class DeepJSCCQDecoder(nn.Module):
    def __init__(self, N: int, M: int) -> None:
        """The function initializes a neural network model with a series of attention blocks and
        residual blocks for image processing.

        Parameters
        ----------
        N : int
            The parameter N represents the number of input channels, while M represents the number of
        output channels.
        M : int
            The parameter M represents the number of input channels for the neural network model.
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
        """The forward function applies a series of layers to the input x and returns the final
        output.

        Parameters
        ----------
        x
            The parameter "x" represents the input data that will be passed through the layers of the
        neural network.

        Returns
        -------
            The output of the last layer in the self.g_s list.
        """

        for layer in self.g_s:
            x = layer(x)

        return x
