import torch.nn as nn
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)

from kaira.models.components.afmodule import AFModule

from ..base import BaseModel
from ..registry import ModelRegistry


@ModelRegistry.register_model()
class DeepJSCCQ2Encoder(BaseModel):
    """DeepJSCCQ2 Encoder Module.

    This module encodes an image into a latent representation using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, csi_length: int = 1) -> None:
        """Initialize the DeepJSCCQ2Encoder.

        Args:
            N (int): The number of input channels or feature maps in the neural network model.
            M (int): The number of output channels in the final layer of the neural network.
            csi_length (int, optional): The number of dimensions in the CSI (Channel State Information) data.
        """
        super().__init__()

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
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

    @property
    def bandwidth_ratio(self) -> float:
        """Calculate the bandwidth ratio of the model.

        Returns:
            float: The bandwidth ratio.
        """
        return 1 / 4  # Downsampling 2x twice

    def forward(self, x):
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input image.
            snr (torch.Tensor): The signal-to-noise ratio.

        Returns:
            torch.Tensor: The encoded latent representation.
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


@ModelRegistry.register_model()
class DeepJSCCQ2Decoder(BaseModel):
    """DeepJSCCQ2 Decoder Module.

    This module decodes a latent representation into an image using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, csi_length: int = 1) -> None:
        """Initialize the DeepJSCCQ2Decoder.

        Args:
            N (int): The number of channels in the input and output feature maps of the neural network.
            M (int): The number of input channels for the AttentionBlock and ResidualBlock modules.
            csi_length (int, optional): The number of dimensions in the CSI (Channel State Information) data.
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

    @property
    def bandwidth_ratio(self) -> float:
        """Calculate the bandwidth ratio of the model.

        Returns:
            float: The bandwidth ratio.
        """
        return 4.0  # Upsampling 2x twice

    def forward(self, x):
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The encoded latent representation.
            snr (torch.Tensor): The signal-to-noise ratio.

        Returns:
            torch.Tensor: The decoded image.
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
