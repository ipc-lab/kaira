"""CNN-based encoder and decoder components for deep communications."""

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaira.models.base import BaseModel

from ..registry import ModelRegistry


@ModelRegistry.register_model()
class ConvEncoder(BaseModel):
    """Convolutional Neural Network (CNN) Encoder for image transmission systems.

    This module implements a CNN-based encoder that maps input images to encoded signals suitable
    for transmission over a communication channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        hidden_dims: Optional[List[int]] = None,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the ConvEncoder.

        Args:
            in_channels (int): Number of input channels in the image.
            out_features (int): Dimensionality of the output encoded signals.
            hidden_dims (List[int], optional): List of feature dimensions for hidden layers.
                If None, default dimensions [16, 32, 64] will be used.
            kernel_size (int, optional): Kernel size for convolutions. Default is 3.
            stride (int, optional): Stride for convolutions. Default is 2.
            padding (int, optional): Padding for convolutions. Default is 1.
            activation (nn.Module, optional): Activation function to use.
                If None, ReLU is used.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if hidden_dims is None:
            hidden_dims = [16, 32, 64]

        if activation is None:
            activation = nn.ReLU()

        # Build CNN encoder layers
        layers = []

        # First convolutional layer
        layers.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(activation)

        # Additional convolutional layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(activation)

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of flattened features after convolutions
        # This is an approximate calculation assuming square input images and valid padding
        self._feature_size: Optional[int] = None

        # Add a final linear layer to map to the desired output dimension
        calculated_feature_size = self._get_flattened_size(in_channels, hidden_dims)
        if calculated_feature_size is None:
            # This case should ideally not happen if _get_flattened_size works correctly
            raise RuntimeError("Could not determine flattened feature size.")
        self.fc = nn.Linear(calculated_feature_size, out_features)

    def _get_flattened_size(self, in_channels: int, hidden_dims: List[int]) -> Optional[int]:
        """Calculate the flattened size after convolutions.

        Since the actual spatial dimensions depend on the input size, we'll use a
        forward pass with a dummy input to determine the size.

        Args:
            in_channels (int): Number of input channels.
            hidden_dims (List[int]): Hidden dimensions list.

        Returns:
            int: Size of flattened feature vector.
        """
        if self._feature_size is not None:
            return self._feature_size

        # Use a small dummy input to calculate output size
        dummy_input = torch.zeros(1, in_channels, 32, 32)  # Assume minimum size of 32x32
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)

        self._feature_size = dummy_output.numel()
        return self._feature_size

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the ConvEncoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, height, width).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = x.size(0)

        # Apply convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = x.view(batch_size, -1)

        # Apply the final linear layer
        x = self.fc(x)

        return x


@ModelRegistry.register_model()
class ConvDecoder(BaseModel):
    """Convolutional Neural Network (CNN) Decoder for image transmission systems.

    This module implements a CNN-based decoder that maps received signals back to their
    corresponding images.
    """

    def __init__(
        self,
        in_features: int,
        out_channels: int,
        output_size: Tuple[int, int],
        hidden_dims: Optional[List[int]] = None,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the ConvDecoder.

        Args:
            in_features (int): Dimensionality of the input received signals.
            out_channels (int): Number of output channels in the reconstructed image.
            output_size (Tuple[int, int]): Height and width of the output image.
            hidden_dims (List[int], optional): List of feature dimensions for hidden layers.
                If None, default dimensions [64, 32, 16] will be used.
            kernel_size (int, optional): Kernel size for transposed convolutions. Default is 3.
            stride (int, optional): Stride for transposed convolutions. Default is 2.
            padding (int, optional): Padding for transposed convolutions. Default is 1.
            output_padding (int, optional): Output padding for transposed convolutions. Default is 1.
            activation (nn.Module, optional): Activation function to use between layers.
                If None, ReLU is used.
            output_activation (nn.Module, optional): Activation function to use at the output.
                If None, Sigmoid is used to output values in [0, 1] range.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]  # Decoder usually goes from smaller to larger

        if activation is None:
            activation = nn.ReLU()

        if output_activation is None:
            output_activation = nn.Sigmoid()  # For image output in [0, 1] range

        # Calculate initial spatial dimension
        self.output_height, self.output_width = output_size
        self.initial_height = self.output_height // (2 ** len(hidden_dims))
        self.initial_width = self.output_width // (2 ** len(hidden_dims))

        # Ensure minimum size
        self.initial_height = max(1, self.initial_height)
        self.initial_width = max(1, self.initial_width)

        # Calculate initial feature map size for the first layer
        self.initial_features = hidden_dims[0]

        # Initial linear layer to transform from code vector to initial feature maps
        self.fc = nn.Linear(in_features, self.initial_features * self.initial_height * self.initial_width)

        # Build transpose convolutional layers
        layers = []

        # Add transpose convolutional layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            layers.append(activation)

        # Final transpose convolutional layer to produce the output image
        layers.append(nn.ConvTranspose2d(hidden_dims[-1], out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))

        # Final activation for output in [0, 1] range
        if output_activation is not None:
            layers.append(output_activation)

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the ConvDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Output image tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size = x.size(0)

        # Apply the initial linear layer
        x = self.fc(x)

        # Reshape to initial feature maps
        x = x.view(batch_size, self.initial_features, self.initial_height, self.initial_width)

        # Apply transpose convolutional layers
        x = self.conv_layers(x)

        # Ensure output size is correct (in case of dimension mismatch due to rounding)
        # This can happen due to integer division in calculating initial dimensions
        x = F.interpolate(x, size=(self.output_height, self.output_width), mode="bilinear", align_corners=False)

        return x
