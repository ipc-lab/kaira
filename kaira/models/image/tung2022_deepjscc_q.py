"""Implementation of the DeepJSCC-Q model from :cite:`tung2022deepjsccq`."""

from typing import Any

import torch
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
class Tung2022DeepJSCCQEncoder(BaseModel):
    """DeepJSCCQ Encoder Module :cite:`tung2022deepjsccq`.

    This module encodes an image into a latent representation using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, in_ch: int = 3, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCCQEncoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks in the g_a module.
            M (int): The number of output channels in the last convolutional layer of the network.
            in_ch (int, optional): The number of input channels. Defaults to 3.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=in_ch, out_ch=N, stride=2),
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

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input image.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The encoded latent representation.
        """

        for layer in self.g_a:
            x = layer(x)

        return x


@ModelRegistry.register_model()
class Tung2022DeepJSCCQDecoder(BaseModel):
    """DeepJSCCQ Decoder Module :cite:`tung2022deepjsccq`.

    This module decodes a latent representation into an image using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, out_ch: int = 3, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCCQDecoder.

        Args:
            N (int): The number of input channels.
            M (int): The number of output channels.
            out_ch (int, optional): The number of output channels. Defaults to 3.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

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
                ResidualBlockUpsample(in_ch=N, out_ch=out_ch, upsample=2),
            ]
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The encoded latent representation.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The decoded image.
        """

        for layer in self.g_s:
            x = layer(x)

        return x


@ModelRegistry.register_model()
class Tung2022DeepJSCCQ2Encoder(BaseModel):
    """DeepJSCCQ2 Encoder Module :cite:`tung2022deepjsccq2`.

    This module is from the conference paper, not the journal version. Note that this module is
    different than DeepJSCCQ, which contains 4 strided layers and does not contain the AFModule.

    This module encodes an image into a latent representation using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, in_ch: int = 3, csi_length: int = 1, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCCQ2Encoder.

        Args:
            N (int): The number of input channels or feature maps in the neural network model.
            M (int): The number of output channels in the final layer of the neural network.
            in_ch (int, optional): The number of input channels. Defaults to 3.
            csi_length (int, optional): The number of dimensions in the CSI (Channel State Information) data.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=in_ch, out_ch=N, stride=2),
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

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input image tensor.
            csi (torch.Tensor): Channel State Information tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The encoded latent representation.
        """

        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer(x, csi=csi)
            else:
                x = layer(x)

        return x


@ModelRegistry.register_model()
class Tung2022DeepJSCCQ2Decoder(BaseModel):
    """DeepJSCCQ2 Decoder Module :cite:`tung2022deepjsccq2`.

    This module is from the conference paper, not the journal version. Note that this module is
    different than DeepJSCCQ, which contains 4 strided layers and does not contain the AFModule.

    This module decodes a latent representation into an image using a series of convolutional
    layers and AFModules.
    """

    def __init__(self, N: int, M: int, out_ch: int = 3, csi_length: int = 1, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCCQ2Decoder.

        Args:
            N (int): The number of channels in the input and output feature maps of the neural network.
            M (int): The number of input channels for the AttentionBlock and ResidualBlock modules.
            out_ch (int, optional): The number of output channels. Defaults to 3.
            csi_length (int, optional): The number of dimensions in the CSI (Channel State Information) data.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

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
                ResidualBlockUpsample(in_ch=N, out_ch=out_ch, upsample=2),
                AFModule(N=out_ch, csi_length=csi_length),
            ]
        )

    @property
    def bandwidth_ratio(self) -> float:
        """Calculate the bandwidth ratio of the model.

        Returns:
            float: The bandwidth ratio.
        """
        return 4.0  # Upsampling 2x twice

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): The encoded latent representation tensor.
            csi (torch.Tensor): Channel State Information tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The decoded image.
        """

        for layer in self.g_s:
            if isinstance(layer, AFModule):
                x = layer(x, csi=csi)
            else:
                x = layer(x)

        return x
