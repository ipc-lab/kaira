"""MLP-based encoder and decoder components for communication systems."""

from typing import Any, List, Optional

import torch
import torch.nn as nn

from kaira.models.base import BaseModel

from ..registry import ModelRegistry


@ModelRegistry.register_model()
class MLPEncoder(BaseModel):
    """Multi-Layer Perceptron (MLP) Encoder for communication systems.

    This module implements a simple MLP-based encoder that maps input messages to encoded signals
    suitable for transmission over a communication channel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the MLPEncoder.

        Args:
            in_features (int): The dimensionality of the input messages.
            out_features (int): The dimensionality of the output encoded signals.
            hidden_dims (List[int], optional): Dimensions of hidden layers.
                If None, a single hidden layer with (in_features + out_features) // 2 units is used.
            activation (nn.Module, optional): Activation function to use between layers.
                If None, ReLU is used.
            output_activation (nn.Module, optional): Activation function to use at the output.
                If None, no activation is applied to the output.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if hidden_dims is None:
            hidden_dims = [(in_features + out_features) // 2]

        if activation is None:
            activation = nn.ReLU()

        # Build MLP layers
        layers = []

        # Input layer
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, out_features))
        if output_activation is not None:
            layers.append(output_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the MLPEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.model(x)


@ModelRegistry.register_model()
class MLPDecoder(BaseModel):
    """Multi-Layer Perceptron (MLP) Decoder for communication systems.

    This module implements a simple MLP-based decoder that maps received signals back to their
    corresponding messages.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the MLPDecoder.

        Args:
            in_features (int): The dimensionality of the input received signals.
            out_features (int): The dimensionality of the output decoded messages.
            hidden_dims (List[int], optional): Dimensions of hidden layers.
                If None, a single hidden layer with (in_features + out_features) // 2 units is used.
            activation (nn.Module, optional): Activation function to use between layers.
                If None, ReLU is used.
            output_activation (nn.Module, optional): Activation function to use at the output.
                If None, no activation is applied to the output.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if hidden_dims is None:
            hidden_dims = [(in_features + out_features) // 2]

        if activation is None:
            activation = nn.ReLU()

        # Build MLP layers
        layers = []

        # Input layer
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, out_features))
        if output_activation is not None:
            layers.append(output_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the MLPDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.model(x)
