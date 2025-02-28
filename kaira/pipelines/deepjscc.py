"""DeepJSCC module for Kaira.

This module contains the DeepJSCCPipeline, which is a pipeline for image compression and
transmission using Deep Joint Source-Channel Coding (DeepJSCC).
"""


from .base import BaseChannel, BaseConstraint, BaseModel
from .sequential import SequentialPipeline


class DeepJSCCPipeline(SequentialPipeline):
    """A specialized pipeline for Deep Joint Source-Channel Coding (DeepJSCC).

    DeepJSCC is a neural network-based approach that jointly optimizes for both
    source compression and channel coding. This pipeline connects an encoder,
    power constraint, channel simulator, and decoder in a sequential manner to
    form a complete image transmission system.

    The typical workflow is:
    1. Input images are encoded into a lower-dimensional representation
    2. The encoded representation is power-constrained
    3. The constrained representation passes through a simulated channel
    4. The decoder reconstructs the original image from the channel output

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
    ):
        """Initialize the DeepJSCC pipeline.

        Args:
            encoder (BaseModel): Neural network model for encoding/compressing the input
            constraint (BaseConstraint): Module for applying power constraints to the encoded signal
            channel (BaseChannel): Module simulating the communication channel
            decoder (BaseModel): Neural network model for decoding/reconstructing the input
        """
        super().__init__(steps=[encoder, constraint, channel, decoder])
        self.encoder = encoder
        self.constraint = constraint
        self.channel = channel
        self.decoder = decoder
