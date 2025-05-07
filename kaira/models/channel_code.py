"""Channel coding module for Kaira.

This module contains the ChannelCodeModel, which is a model for channel transmission using a
conventional encoding/decoding pipeline.
"""

from typing import Any, Callable, List  # Modified import

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint
from kaira.modulations import BaseDemodulator, BaseModulator

from .base import BaseModel
from .generic import SequentialModel
from .registry import ModelRegistry


@ModelRegistry.register_model("channel_code")
class ChannelCodeModel(SequentialModel):
    """A specialized model for Channel Code.

    Channel Code is an information transmission approach that performs encoding and decoding using given channel code.
    This model connects an encoder, power constraint, channel simulator, and decoder in an information transmission system.

    The typical workflow is:
    1. Input data is encoded with additional redundancy for further information recovery
    2. The encoded representation is power-constrained
    3. The constrained representation is modulated and passed over a noisy channel
    4. The decoder reconstructs the original data from the demodulated channel output

    Attributes:
        encoder (BaseModel): Channel code encoder that algorithmically encodes the input
        constraint (BaseConstraint): Module that applies power constraints to the encoded signal
        modulator (BaseModulator): Module that modulates the encoded signal
        channel (BaseChannel): Simulates the communication channel effects
        demodulator (BaseDemodulator): Module that demodulates the received signal
        decoder (BaseModel): Channel code decoder that algorithmically reconstructs the input from the received signal
    """

    def __init__(
        self,
        encoder: BaseModel,
        constraint: BaseConstraint,
        modulator: BaseModulator,
        channel: BaseChannel,
        demodulator: BaseDemodulator,
        decoder: BaseModel,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Channel Code model.

        Args:
            encoder (BaseModel): Channel code encoder for encoding the input
            constraint (BaseConstraint): Module for applying power constraints to the encoded signal
            modulator (BaseModulator): Module for modulating the encoded signal
            channel (BaseChannel): Module simulating the communication channel
            demodulator (BaseDemodulator): Module for demodulating the received signal
            decoder (BaseModel): Channel code decoder for decoding the demodulated channel output
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Pass steps as a positional argument
        # The order of steps in the list defines the execution order.
        steps_list: List[Callable[..., Any]] = [
            encoder,
            modulator,
            constraint,
            channel,
            demodulator,
            decoder,
        ]
        super().__init__(steps_list, *args, **kwargs)
        self.encoder = encoder
        self.modulator = modulator
        self.constraint = constraint
        self.channel = channel
        self.demodulator = demodulator
        self.decoder = decoder
