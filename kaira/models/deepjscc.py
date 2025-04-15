"""DeepJSCC module for Kaira.

This module implements Deep Joint Source-Channel Coding (DeepJSCC), a deep learning approach that
jointly optimizes source compression and channel coding for efficient transmission over noisy
channels. The implementation is based on the seminal work by Bourtsoulatze et al. and subsequent
extensions.
"""

from typing import Any

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint

from .base import BaseModel
from .generic.sequential import SequentialModel
from .registry import ModelRegistry


@ModelRegistry.register_model("deepjscc")
class DeepJSCCModel(SequentialModel):
    """Deep Joint Source-Channel Coding model.

    This model implements end-to-end joint source-channel coding using deep neural
    networks. It consists of:
    - An encoder that compresses and encodes the source signal
    - A power constraint that normalizes the encoded signal
    - A channel that simulates the transmission medium
    - A decoder that reconstructs the original signal

    Key features:
    - End-to-end differentiable architecture
    - Automatic adaptation to channel conditions
    - No separate source/channel coding
    - Graceful degradation with channel quality
    - Support for various source types (images, audio, etc.)

    Example:
        >>> # Create a DeepJSCC model for image transmission
        >>> model = DeepJSCCModel(
        ...     encoder=image_encoder,
        ...     constraint=power_constraint,
        ...     channel=awgn_channel,
        ...     decoder=image_decoder
        ... )
        >>> # Transmit image through noisy channel
        >>> received = model(image, snr=10.0)
    Attributes:
        encoder (BaseModel): Neural network that compresses the input
        constraint (BaseConstraint): Module that applies power constraints to the encoded signal
        channel (BaseChannel): Simulates the communication channel effects
        decoder (BaseModel): Neural network that reconstructs the input from the received signal
    """

    def __init__(
        self,
        encoder: BaseModel,
        constraint: BaseConstraint,
        channel: BaseChannel,
        decoder: BaseModel,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the DeepJSCC model.

        Args:
            encoder (BaseModel): Neural network model for encoding/compressing the input
            constraint (BaseConstraint): Module for applying power constraints to the encoded signal
            channel (BaseChannel): Module simulating the communication channel
            decoder (BaseModel): Neural network model for decoding/reconstructing the input
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Pass steps as a positional argument
        super().__init__([encoder, constraint, channel, decoder], *args, **kwargs)
        self.encoder = encoder
        self.constraint = constraint
        self.channel = channel
        self.decoder = decoder
