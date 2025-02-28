"""Fading Channel module for Kaira.

This module contains the FadingChannelPipeline, which models communication systems operating over
various types of fading channels (Rayleigh, Rician, Nakagami, etc.)
"""

from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseModel, BasePipeline


class FadingType(Enum):
    """Enumeration of supported fading channel models."""

    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    NAKAGAMI = "nakagami"
    LOG_NORMAL = "log_normal"
    CUSTOM = "custom"


class FadingChannelPipeline(BasePipeline):
    """A pipeline for communication over fading channels.

    This pipeline models transmission over wireless fading channels, which are
    characterized by random variations in signal amplitude and phase. It supports
    several common fading models including Rayleigh, Rician, and Nakagami.

    Attributes:
        encoder (BaseModel): Encodes the source signal
        modulator (nn.Module): Modulates encoded data
        fading_channel (nn.Module): Models the fading channel effects
        equalizer (nn.Module): Performs channel estimation and equalization
        demodulator (nn.Module): Demodulates the received signal
        decoder (BaseModel): Decodes the demodulated signal
        fading_type (FadingType): Type of fading channel
        channel_params (Dict): Parameters for the fading model
    """

    def __init__(
        self,
        encoder: BaseModel,
        modulator: nn.Module,
        fading_channel: nn.Module,
        equalizer: nn.Module,
        demodulator: nn.Module,
        decoder: BaseModel,
        fading_type: FadingType = FadingType.RAYLEIGH,
        channel_params: Optional[Dict] = None,
    ):
        """Initialize the fading channel pipeline.

        Args:
            encoder: Model that encodes source data
            modulator: Module that modulates encoded data
            fading_channel: Module that simulates fading channel effects
            equalizer: Module that performs channel estimation and equalization
            demodulator: Module that demodulates the received signal
            decoder: Model that decodes the received data
            fading_type: Type of fading channel model (default: Rayleigh)
            channel_params: Parameters for the fading channel
        """
        super().__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.fading_channel = fading_channel
        self.equalizer = equalizer
        self.demodulator = demodulator
        self.decoder = decoder
        self.fading_type = fading_type
        self.channel_params = channel_params or {}

        # Configure fading channel with provided parameters
        if hasattr(self.fading_channel, "configure"):
            self.fading_channel.configure(fading_type=fading_type.value, **self.channel_params)

    def set_channel_params(self, fading_type: Optional[FadingType] = None, **kwargs) -> "FadingChannelPipeline":
        """Update the fading channel parameters.

        Args:
            fading_type: New fading type (if None, keeps current type)
            **kwargs: Parameters for the fading model

        Returns:
            Self for method chaining
        """
        if fading_type is not None:
            self.fading_type = fading_type

        self.channel_params.update(kwargs)

        # Reconfigure fading channel with updated parameters
        if hasattr(self.fading_channel, "configure"):
            self.fading_channel.configure(fading_type=self.fading_type.value, **self.channel_params)

        return self

    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input through the fading channel communication system.

        Args:
            input_data: The source data to transmit

        Returns:
            Dictionary containing intermediate and final outputs of the pipeline:
                - encoded: Output from the encoder
                - modulated: Modulated signal
                - channel_output: Signal after passing through the fading channel
                - channel_estimate: Estimated channel state information
                - equalized: Signal after equalization
                - demodulated: Demodulated symbols
                - decoded: Final decoded output
        """
        # Encode input data
        encoded = self.encoder(input_data)

        # Modulate the encoded data
        modulated = self.modulator(encoded)

        # Pass through fading channel
        channel_output, channel_state = self.fading_channel(modulated, return_state=True)

        # Apply equalization using channel state information
        equalized = self.equalizer(channel_output, channel_state=channel_state)

        # Demodulate the equalized signal
        demodulated = self.demodulator(equalized)

        # Decode the demodulated signal
        decoded = self.decoder(demodulated)

        return {
            "encoded": encoded,
            "modulated": modulated,
            "channel_output": channel_output,
            "channel_estimate": channel_state,
            "equalized": equalized,
            "demodulated": demodulated,
            "decoded": decoded,
        }
