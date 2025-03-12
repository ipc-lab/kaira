"""TransCoder module for Kaira.

This module contains the TransCoderModel, which is a model for channel
transmission using TransCoder pipeline.
"""

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint
from kaira.modulations import BaseModulator, BaseDemodulator
from ..base import BaseModel
from ..registry import ModelRegistry
from ..generic import SequentialModel
import torch
from typing import Any, Dict

@ModelRegistry.register_model("transcoder")
class Kurmukova2025TransCoderModel(SequentialModel):
    """A specialized model for TransCoder.

    TransCoder is a neural network-based approach that enhances the channel coding performance and 
    this model pipeline combines both transformer-based coding and channel coding. This scheme connects 
    neural and channel encoders, power constraint, channel simulator, and neural and channel decoders 
    in a sequential manner to enhance an information transmission system.

    The typical workflow is:
    1. Input data is encoded into a codeword with channel encoder
    2. If TransCoder neural encoder is enabled, then the codeword is encoded into 
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
        encoder_tc: BaseModel,
        encoder_ec: BaseModel,
        constraint: BaseConstraint,
        modulator: BaseModulator,
        channel: BaseChannel,
        demodulator: BaseDemodulator,
        decoder_tc: BaseModel,
        decoder_ec: BaseModel,
        n_iterations: int = 1,
    ):
        """Initialize the TransCoder model.

        Args:
            encoder_ec (BaseModel): Channel code encoder for encoding the input
            encoder_tc (BaseModel): TransCoder neural encoder for the modulation(encoding) of the encoded input
            modulator (BaseModulator): Module for modulating the encoded signal (in case encoder_tc is not used)
            constraint (BaseConstraint): Module for applying power constraints to the modulated signal
            channel (BaseChannel): Module simulating the communication channel
            demodulator (BaseDemodulator): Module for demodulating the received signal (in case decoder_tc is not used)
            decoder_tc (BaseModel): TransCoder neural decoding model for decoding the channel output
            decoder_ec (BaseModel): Channel code decoder for recovering the original input
            n_iterations (int): Number of consecutive decoding iterations: TransCoder and channel decoder (default: 1)
        """
        super().__init__(steps=[encoder_tc, encoder_ec, modulator, constraint,
                                channel, demodulator, decoder_tc, decoder_ec,
                                n_iterations])
        self.encoder_tc = encoder_tc
        self.encoder_ec = encoder_ec
        self.modulator = modulator
        self.constraint = constraint
        self.channel = channel
        self.demodulator = demodulator
        self.decoder_tc = decoder_tc
        self.decoder_ec = encoder_ec
        self.n_iterations = n_iterations

    def forward(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Process input through the feedback channel system.

        Performs an iterative transmission process where:
        1. The input data is encoded with a channel encoder
        2. If the TransCoder neural encoder is enabled, the encoded data is further encoded into a sequence of power-constrained symbols, 
        else the encoded data is modulated and power-constrained before transmission
        3. The constrained data is transmitted over the channel
        4. The received data is passed to TransCoder neural decoder (in case the TransCoder decoder is used) or to demodulator (otherwise)
        5. The refined/demodulated data is passed to the channel decoder
        6. The steps 5-6 are repeated for the specified number of iterations

        Args:
            input_data (torch.Tensor): The input data to transmit

        Returns:
            Dict[str, Any]: A dictionary containing:
                - final_output: The final decoded output
                - iterations: List of per-iteration results
                - history: History of transmitted signals
        """
        batch_size = input_data.shape[0]
        device = input_data.device

        # Storage for results
        iterations = []
        history = []

        # Channel encoding
        encoded_ec = self.encoder_ec(input_data)
        if self.encoder_tc is not None:
            # TransCoder encoding
            constrained = self.encoder_tc(encoded_ec)
        else:
            # Modulation
            modulated = self.modulator(encoded_ec)
            # Power constraint
            constrained = self.constraint(modulated)
        
        # Transmit through channel
        received = self.channel(constrained)

        history.append(
                {
                    "encoded": encoded,
                    "constrained": constrained,
                    "received": received,
                }
            )
        
        # Iterative decoding process
        for i in range(self.n_iterations):
            if self.decoder_tc is not None:
                # TransCoder decoding
                if i == 0:
                    demodulated = self.decoder_tc(received)
                else:
                    demodulated = self.decoder_tc([received, soft_estimate])
            else:
                # Demodulation
                demodulated = self.demodulator(received)

            # Channel decoding
            decoded, soft_estimate = self.decoder_ec(demodulated)

            # Store results for this iteration
            iterations.append(
                {
                    "demodulated": demodulated,
                    "decoded": decoded,
                    "soft_estimate": soft_estimate,
                }
            )

        return {
            "final_output": decoded,
            "iterations": iterations,
            "history": history,
        }