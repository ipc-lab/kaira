"""Channel coding module for Kaira.

This module contains the ChannelCodeModel, which is a model for channel transmission using a
conventional encoding/decoding pipeline.
"""

from typing import Any, Dict

import torch

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
    ):
        """Initialize the Channel Code model.

        Args:
            encoder (BaseModel): Channel code encoder for encoding the input
            constraint (BaseConstraint): Module for applying power constraints to the encoded signal
            modulator (BaseModulator): Module for modulating the encoded signal
            channel (BaseChannel): Module simulating the communication channel
            demodulator (BaseDemodulator): Module for demodulating the received signal
            decoder (BaseModel): Channel code decoder for decoding the demodulated channel output
        """
        super().__init__(steps=[encoder, modulator, constraint, channel, demodulator, decoder])
        self.encoder = encoder
        self.modulator = modulator
        self.constraint = constraint
        self.channel = channel
        self.demodulator = demodulator
        self.decoder = decoder

    def forward(self, input_data: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process input through the feedback channel system.

        Performs an iterative transmission process where:
        1. Transmitter encodes and modulates the input data
        2. The modulated signal is transmitted over the channel
        2. Receiver demodulates, decodes and generates the estimate of the input data

        Args:
            input_data (torch.Tensor): The input data to transmit
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: A dictionary containing:
                - final_output: The final decoded output
                - history: The history of encoded, received and decoded results
        """
        input_data.shape[0]

        # Storage for results
        history = []

        # Transmission process

        # Encode the input
        encoded = self.encoder(input_data, *args, **kwargs)

        encoded = (encoded > 0).float()

        # Modulation of the encoded data
        modulated = self.modulator(encoded)

        # Apply power constraint
        constrained = self.constraint(modulated)

        # Transmit through the channel
        received = self.channel(constrained)

        # Demodulate and decode the received signal
        decoded, soft_estimate = self.decoder(self.demodulator(received), *args, **kwargs)

        # Store results
        history.append(
            {
                "encoded": encoded,
                "received": received,
                "decoded": decoded,
                "soft_estimate": soft_estimate,
            }
        )

        return {
            "final_output": decoded,
            "history": history,
        }
