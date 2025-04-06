"""Wyner-Ziv module for Kaira.

This module contains the WynerZivModel, which implements a distributed source coding
system with side information at the decoder, based on the Wyner-Ziv coding theorem.

The Wyner-Ziv coding theorem (A. Wyner and J. Ziv, 1976) is a fundamental result in
information theory that establishes the rate-distortion function for lossy source coding
with side information available only at the decoder. This provides theoretical foundations
for various practical applications such as distributed video/image coding, sensor networks,
and distributed computing where communication resources are limited but correlated
information exists at different nodes.

The implementation follows the key principles of Wyner-Ziv coding:
1. Source encoding without access to side information
2. Binning/quantization of encoded source
3. Syndrome generation for efficient transmission
4. Reconstruction at decoder using both received syndromes and side information
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint
from kaira.data.correlation import WynerZivCorrelationModel

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register_model("wyner_ziv")
class WynerZivModel(BaseModel):
    """A model for Wyner-Ziv coding with decoder side information.

    Wyner-Ziv coding is a form of lossy source coding with side information at the decoder.
    This model implements the complete process including source encoding, quantization,
    syndrome generation, channel transmission, and decoding with side information.

    The model follows these key steps:

    1. The encoder compresses the source without knowledge of side information
    2. The quantizer maps the encoded values to discrete symbols/indices
    3. The syndrome generator creates a compressed representation (syndromes)
       that will be used for reconstruction when combined with side information
    4. The syndromes are transmitted through a potentially noisy channel
    5. The decoder combines received syndromes with side information to reconstruct
       the original source with minimal distortion

    This implementation can be used for various distributed coding scenarios like
    distributed image/video compression, sensor networks, etc.

    Attributes:
        encoder (BaseModel): Transforms the source data into a suitable representation
        quantizer (nn.Module): Discretizes the continuous encoded representation
        syndrome_generator (nn.Module): Creates syndrome bits for efficient transmission
        channel (BaseChannel): Models the communication channel characteristics
        correlation_model (WynerZivCorrelationModel): Models statistical relationship
            between source and side information (used when side info is not provided)
        decoder (BaseModel): Reconstructs source using received syndromes and side info
        constraint (BaseConstraint): Optional constraint on transmitted data (e.g., power)
    """

    def __init__(
        self,
        encoder: BaseModel,
        channel: BaseChannel,
        decoder: BaseModel,
        correlation_model: Optional[WynerZivCorrelationModel] = None,
        quantizer: Optional[nn.Module] = None,
        syndrome_generator: Optional[nn.Module] = None,
        constraint: Optional[BaseConstraint] = None,
    ):
        """Initialize the Wyner-Ziv model.

        Args:
            encoder: Model that encodes the source data into a latent representation
                without knowledge of the side information
            channel: Channel model that simulates transmission effects such as
                noise, fading, or packet loss on the syndromes
            decoder: Model that reconstructs the source using received syndromes
                and the side information available at the decoder
            correlation_model: Model that generates or simulates the correlation
                between the source and side information. Optional for subclasses that
                always expect side_info to be provided.
            quantizer: Module that discretizes the encoded representation into
                a finite set of indices or symbols. Optional for subclasses that
                don't require explicit quantization.
            syndrome_generator: Module that generates syndromes (parity bits or
                compressed representation) for error correction or compression.
                Optional for subclasses that don't use explicit syndromes.
            constraint: Optional constraint (e.g., power, rate) applied to the
                transmitted syndromes
        """
        super().__init__()
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder
        self.correlation_model = correlation_model
        self.quantizer = quantizer
        self.syndrome_generator = syndrome_generator
        self.constraint = constraint

    def validate_side_info(self, source: torch.Tensor, side_info: Optional[torch.Tensor]) -> torch.Tensor:
        """Validate and/or generate side information if needed.

        Args:
            source: The source data
            side_info: Optional side information

        Returns:
            Valid side information, either provided or generated

        Raises:
            ValueError: If side_info is None and no correlation_model is available
        """
        if side_info is None:
            if self.correlation_model is None:
                raise ValueError("Side information must be provided when correlation_model is not available")
            # Generate side information from correlation model
            generated_side_info = self.correlation_model(source)
            return generated_side_info
        return side_info

    def forward(self, source: torch.Tensor, side_info: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Process source through the Wyner-Ziv coding system.

        Implements the full Wyner-Ziv coding model:
        1. Encodes source into a latent representation
        2. Quantizes the latent representation (if quantizer is present)
        3. Generates syndromes (if syndrome_generator is present)
        4. Applies optional constraints on syndromes
        5. Transmits syndromes through the channel
        6. Either uses provided side information or generates it through correlation model
        7. Reconstructs source using received syndromes and side information

        Args:
            source: The source data to encode and transmit efficiently
            side_info: Optional pre-generated side information available at decoder.
                If None, side information is generated using the correlation_model,
                simulating a real-world scenario where side info is independently
                available at the decoder
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing intermediate and final outputs of the model:
                - encoded: Latent representation from source encoder
                - quantized: Discretized representation after quantization (if applicable)
                - syndromes: Compressed representation for transmission (if applicable)
                - constrained: Syndromes/encoded data after applying optional constraints
                - received: Data after channel transmission (possibly corrupted)
                - side_info: Side information available at decoder
                - decoded: Final reconstruction of the source using received data and side info
        """
        # Source encoding
        encoded = self.encoder(source, *args, **kwargs)
        result = {"encoded": encoded}

        # Quantization (if available)
        if self.quantizer is not None:
            result["quantized"] = self.quantizer(encoded)
        else:
            result["quantized"] = encoded

        # Generate syndromes for error correction (if available)
        if self.syndrome_generator is not None:
            result["syndromes"] = self.syndrome_generator(result["quantized"])
        else:
            result["syndromes"] = result["quantized"]

        # Apply optional power constraint on syndromes
        if self.constraint is not None:
            result["constrained"] = self.constraint(result["syndromes"])
        else:
            result["constrained"] = result["syndromes"]

        # Transmit syndromes through channel
        result["received"] = self.channel(result["constrained"])

        # Validate/generate side information if needed
        result["side_info"] = self.validate_side_info(source, side_info)

        # Decode using received syndromes and side information
        result["decoded"] = self.decoder(result["received"], result["side_info"], *args, **kwargs)

        return result
