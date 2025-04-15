"""Implementation of the DeepJSCC model from Bourtsoulatze et al.

(2019).
"""

from typing import Any, Optional

import torch
from torch import nn

from kaira.models.registry import ModelRegistry

from ..base import BaseModel


class _ConvWithPReLU(nn.Module):
    """Convolutional layer followed by PReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, *args: Any, **kwargs: Any):
        """Initialize the _ConvWithPReLU module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through convolutional layer and PReLU activation.

        Args:
            x: Input tensor
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Activated output tensor
        """
        return self.prelu(self.conv(x))


class _TransConvWithPReLU(nn.Module):
    """Transposed convolutional layer followed by activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0, output_padding: int = 0, activate: Optional[nn.Module] = None, *args: Any, **kwargs: Any):
        """Initialize the _TransConvWithPReLU module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the transposed convolutional kernel.
            stride: Stride of the transposed convolution.
            padding: Padding added to both sides of the input.
            output_padding: Additional size added to one side of the output shape.
            activate: Activation function to use. Defaults to PReLU.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

        # Use PReLU by default if no activation is specified
        self.activate = activate if activate is not None else nn.PReLU()

        if isinstance(self.activate, nn.PReLU):
            nn.init.kaiming_normal_(self.transconv.weight, mode="fan_out", nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through transposed convolutional layer and activation.

        Args:
            x: Input tensor
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Activated output tensor
        """
        return self.activate(self.transconv(x))


@ModelRegistry.register_model()
class Bourtsoulatze2019DeepJSCCEncoder(BaseModel):
    """DeepJSCC encoder model from :cite:`bourtsoulatze2019deep`.

    This model encodes the input image into a latent representation for transmission.

    Args:
        num_transmitted_filters: Number of filters in the final encoding layer
    """

    def __init__(self, num_transmitted_filters: int, *args: Any, **kwargs: Any):
        """Initialize the Bourtsoulatze2019DeepJSCCEncoder.

        Args:
            num_transmitted_filters: Number of filters in the final encoding layer.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
            _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            _ConvWithPReLU(in_channels=32, out_channels=num_transmitted_filters, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x: Input image tensor of shape (B, 3, H, W)
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Encoded representation of shape (B, num_transmitted_filters, H//4, W//4)
        """
        return self.model(x)


@ModelRegistry.register_model()
class Bourtsoulatze2019DeepJSCCDecoder(BaseModel):
    """DeepJSCC decoder model from :cite:`bourtsoulatze2019deep`.

    This model decodes the transmitted representation back into an image.

    Args:
        num_transmitted_filters: Number of filters in the transmitted representation
    """

    def __init__(self, num_transmitted_filters: int, *args: Any, **kwargs: Any):
        """Initialize the Bourtsoulatze2019DeepJSCCDecoder.

        Args:
            num_transmitted_filters: Number of filters in the transmitted representation.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            _TransConvWithPReLU(in_channels=num_transmitted_filters, out_channels=32, kernel_size=5, stride=1, padding=2),
            _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            _TransConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
            _TransConvWithPReLU(in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1, activate=nn.Sigmoid()),
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x: Encoded representation of shape (B, num_transmitted_filters, H//4, W//4)
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Decoded image tensor of shape (B, 3, H, W)
        """
        return self.model(x)
