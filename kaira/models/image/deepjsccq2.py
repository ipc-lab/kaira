import torch.nn as nn
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)

from kaira.models.components.afmodule import AFModule


class DeepJSCCQ2Encoder(nn.Module):
    def __init__(self, N: int, M: int, csi_length: int = 1) -> None:
        """The function initializes a neural network model with a specific architecture consisting
        of residual blocks, attention blocks, and activation function modules.

        Parameters
        ----------
        N : int
            The parameter N represents the number of input channels or feature maps in the neural network
        model. It is used to define the number of input channels for the ResidualBlock and AFModule
        layers.
        M : int
            The parameter M represents the number of output channels in the final layer of the neural
        network.
        csi_length : int, optional
            The `csi_length` parameter represents the number of dimensions in the CSI (Channel State
        Information) data.
        """
        super().__init__()

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=C, out_ch=N, stride=2),
                AFModule(N=N, csi_length=csi_length),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlock(in_ch=N, out_ch=N),
                AFModule(N=N, csi_length=csi_length),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N=N, csi_length=csi_length),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlock(in_ch=N, out_ch=M),
                AFModule(N=M, csi_length=csi_length),
                AttentionBlock(M),
            ]
        )

    def forward(self, x):
        """The forward function takes an input x and passes it through a series of layers in
        self.g_a, applying an AFModule layer if encountered, and returns the final output.

        Parameters
        ----------
        x
            The parameter `x` represents the input to the forward method. It can be either a single value
        or a tuple. If it is a tuple, the first element represents the input data `x`, and the second
        element represents the channel state information (CSI) (e.g. the signal-to-noise ratio `snr`).

        Returns
        -------
            the output of the forward pass through the layers of the model.
        """

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x


class DeepJSCCQ2Decoder(nn.Module):
    def __init__(self, N: int, M: int, csi_length: int = 1) -> None:
        """The function initializes a neural network model with a specific architecture consisting
        of attention blocks, residual blocks, and AF modules.

        Parameters
        ----------
        N : int
            The parameter N represents the number of channels in the input and output feature maps of the
        neural network. It determines the dimensionality of the feature space.
        M : int
            The parameter M represents the number of input channels for the AttentionBlock and
        ResidualBlock modules.
        csi_length : int, optional
            The `csi_length` parameter represents the number of dimensions in the CSI (Channel State
        Information) data. It is used in the `AFModule` class to determine the number of channels in the
        convolutional layers.
        """
        super().__init__()

        self.g_s = nn.ModuleList(
            [
                AttentionBlock(M),
                ResidualBlock(in_ch=M, out_ch=N),
                ResidualBlock(in_ch=N, out_ch=N),
                AFModule(N=N, csi_length=csi_length),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(N=N, csi_length=csi_length),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlock(in_ch=N, out_ch=N),
                AFModule(N=N, csi_length=csi_length),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=3, upsample=2),
                AFModule(N=3, csi_length=csi_length),
            ]
        )

    def forward(self, x):
        """The function takes an input x and passes it through a series of layers, applying an
        activation function if the layer is an AFModule.

        Parameters
        ----------
        x
            The parameter `x` represents the input to the forward method. It can be either a single tensor
        or a tuple containing a tensor and a signal-to-noise ratio (snr).

        Returns
        -------
            the value of `x` after passing it through the layers in `self.g_s`.
        """

        if isinstance(x, tuple):
            x, snr = x
        else:
            snr = None

        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer((x, snr))
            else:
                x = layer(x)

        return x
