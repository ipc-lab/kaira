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

from .base import BaseModel
from .registry import ModelRegistry


class WynerZivCorrelationModel(nn.Module):
    """Model for simulating correlation between source and side information.

    In Wyner-Ziv coding, there is correlation between the source X and the side information
    Y available at the decoder. This module simulates different correlation models between
    the source X and the side information Y. The correlation structure is critical as it
    determines the theoretical rate bounds and practical coding efficiency.

    The correlation model effectively creates a virtual channel between X and Y, which
    can be modeled as various types of conditional probability distributions p(Y|X).

    Attributes:
        correlation_type (str): Type of correlation model ('gaussian', 'binary', 'custom')
            - 'gaussian': Additive white Gaussian noise model (Y = X + N, where N ~ N(0, σ²))
            - 'binary': Binary symmetric channel model with crossover probability p
            - 'custom': User-defined correlation model through a transform function
        correlation_params (Dict): Parameters specific to the correlation model
    """

    def __init__(self, correlation_type: str = "gaussian", correlation_params: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Initialize the correlation model.

        Args:
            correlation_type: Type of correlation model:
                - 'gaussian': Additive Gaussian noise (requires 'sigma' parameter)
                - 'binary': Binary symmetric channel (requires 'crossover_prob' parameter)
                - 'custom': User-defined model (requires 'transform_fn' parameter)
            correlation_params: Parameters for the correlation model:
                - For 'gaussian': {'sigma': float} - Standard deviation of the noise
                - For 'binary': {'crossover_prob': float} - Probability of bit flipping
                - For 'custom': {'transform_fn': callable} - Custom transformation function
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.correlation_type = correlation_type
        self.correlation_params = correlation_params or {}

    def forward(self, source: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Generate correlated side information from the source.

        Creates side information Y that is correlated with the source X according to
        the specified correlation model. This simulates the scenario where the decoder
        has access to side information that is statistically related to the source.

        Args:
            source: Source signal X (can be continuous or discrete valued)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Correlated side information Y with statistical dependence on X according
            to the specified correlation model

        Raises:
            ValueError: If the correlation type is unknown or if the custom correlation
                model is missing the required transform function
        """
        if self.correlation_type == "gaussian":
            # Y = X + Z, where Z ~ N(0, sigma²)
            sigma = self.correlation_params.get("sigma", 1.0)
            noise = torch.randn_like(source) * sigma
            return source + noise

        elif self.correlation_type == "binary":
            # Binary symmetric channel with crossover probability p
            p = self.correlation_params.get("crossover_prob", 0.1)
            flip_mask = torch.bernoulli(torch.full_like(source, p, dtype=torch.float))
            return (source.float() * (1 - flip_mask) + (1 - source.float()) * flip_mask).to(source.dtype)

        elif self.correlation_type == "custom":
            # Custom correlation model
            if "transform_fn" in self.correlation_params:
                return self.correlation_params["transform_fn"](source)
            else:
                raise ValueError("Custom correlation model requires 'transform_fn' parameter")

        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")


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
        *args: Any,
        **kwargs: Any,
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
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.channel = channel
        self.decoder = decoder
        self.correlation_model = correlation_model
        self.quantizer = quantizer
        self.syndrome_generator = syndrome_generator
        self.constraint = constraint

    def forward(self, source: torch.Tensor, side_info: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
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
                If None, side information is generated using the correlation_model.
            *args: Additional positional arguments passed to encoder, quantizer, syndrome_generator, channel, and decoder.
            **kwargs: Additional keyword arguments passed to encoder, quantizer, syndrome_generator, channel, and decoder.

        Returns:
            torch.Tensor: The final reconstructed source tensor after decoding.

        Raises:
            ValueError: If side_info is None and no correlation_model is available.
        """
        # Source encoding
        res = self.encoder(source, *args, **kwargs)

        # Quantization (if available)
        if self.quantizer is not None:
            res = self.quantizer(res, *args, **kwargs)

        # Generate syndromes for error correction (if available)
        if self.syndrome_generator is not None:
            res = self.syndrome_generator(res, *args, **kwargs)

        # Apply optional power constraint on syndromes
        if self.constraint is not None:
            res = self.constraint(res)

        # Transmit syndromes through channel
        res = self.channel(res, *args, **kwargs)

        # Validate/generate side information if needed (moved from validate_side_info)
        if side_info is None:
            if self.correlation_model is None:
                raise ValueError("Side information must be provided when correlation_model is not available")
            # Generate side information from correlation model
            side_info = self.correlation_model(source)
        # If side_info was provided, use it directly.

        # Decode using received syndromes and side information
        res = self.decoder(res, side_info, *args, **kwargs)

        return res
