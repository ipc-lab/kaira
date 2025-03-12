"""Multiple Access Channel models for Kaira.

This module contains the abstract base class for Multiple Access Channel (MAC) models,
which enable multiple devices to transmit data over a shared wireless channel.
"""

import torch
from torch import nn
from typing import List, Dict, Optional, Tuple, Union, Type, Any, Callable

from kaira.models.base import BaseModel
from kaira.channels.base import BaseChannel
from kaira.constraints.base import BaseConstraint


class MultipleAccessChannelModel(BaseModel):
    """Abstract base class for Multiple Access Channel (MAC) models.
    
    A Multiple Access Channel (MAC) model represents a communication system where
    multiple transmitters send data to a single receiver over a shared channel.
    This abstraction provides common functionality for various MAC protocols.
    
    Attributes:
        channel: The channel model for transmission
        power_constraint: Power constraint applied to transmitted signals
        encoders: List of encoder networks for each device
        decoders: List of decoder networks for each device
        num_devices: Number of transmitting devices in the system
        shared_encoder: Whether to use the same encoder for all devices
        shared_decoder: Whether to use the same decoder for all devices
    """

    def __init__(
        self, 
        channel: BaseChannel, 
        power_constraint: BaseConstraint, 
        encoder: Optional[Type[BaseModel]] = None,
        decoder: Optional[Type[BaseModel]] = None,
        num_devices: int = 2, 
        shared_encoder: bool = False,
        shared_decoder: bool = False,
        **kwargs
    ):
        """Initialize the MultipleAccessChannelModel.

        Args:
            channel: Channel model for transmission
            power_constraint: Power constraint to apply to transmitted signals
            encoder: Encoder network class or constructor
            decoder: Decoder network class or constructor
            num_devices: Number of transmitting devices
            shared_encoder: Whether to use a shared encoder across devices
            shared_decoder: Whether to use a shared decoder across devices
        """
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.num_devices = num_devices
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # The derived classes should initialize encoders and decoders

    def _initialize_encoders(
        self, 
        encoder_cls: Type[BaseModel], 
        encoder_count: int, 
        encoder_kwargs: Dict[str, Any]
    ) -> None:
        """Initialize encoder modules.
        
        Args:
            encoder_cls: The encoder class to instantiate
            encoder_count: Number of encoders to initialize
            encoder_kwargs: Keyword arguments to pass to encoder constructor
        """
        for _ in range(encoder_count):
            try:
                enc = encoder_cls(**encoder_kwargs)
                self.encoders.append(enc)
            except Exception as e:
                raise ValueError(f"Failed to initialize encoder: {str(e)}")

    def _initialize_decoders(
        self, 
        decoder_cls: Type[BaseModel], 
        decoder_count: int, 
        decoder_kwargs: Dict[str, Any]
    ) -> None:
        """Initialize decoder modules.
        
        Args:
            decoder_cls: The decoder class to instantiate
            decoder_count: Number of decoders to initialize
            decoder_kwargs: Keyword arguments to pass to decoder constructor
        """
        for _ in range(decoder_count):
            try:
                dec = decoder_cls(**decoder_kwargs)
                self.decoders.append(dec)
            except Exception as e:
                raise ValueError(f"Failed to initialize decoder: {str(e)}")

    def forward(
        self, 
        x: torch.Tensor, 
        csi: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the Multiple Access Channel model.

        Args:
            x: Input data with shape [batch_size, num_devices, channels, height, width]
            csi: Channel state information with shape [batch_size, csi_length]

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        raise NotImplementedError("Subclasses must implement the forward method.")