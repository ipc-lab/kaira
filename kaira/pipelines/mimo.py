"""MIMO module for Kaira.

This module contains the MIMOPipeline, which implements a Multiple-Input Multiple-Output
communication system with spatial multiplexing and diversity techniques.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn

from .base import BasePipeline, BaseChannel, BaseModel, BaseConstraint


class MIMOPipeline(BasePipeline):
    """A pipeline for Multiple-Input Multiple-Output (MIMO) communication systems.
    
    MIMO uses multiple antennas at both the transmitter and receiver to improve
    communication performance through spatial multiplexing and/or diversity gain.
    This pipeline models the encoding, precoding, channel effects, and detection
    stages of a MIMO system.
    
    Attributes:
        encoder (BaseModel): Encodes the source signal
        precoder (nn.Module): Applies precoding/beamforming to encoded signal
        channel (BaseChannel): Models the MIMO channel effects
        equalizer (nn.Module): Performs channel equalization/detection
        decoder (BaseModel): Decodes the detected signal
        num_tx_antennas (int): Number of transmit antennas
        num_rx_antennas (int): Number of receive antennas
    """
    
    def __init__(
        self,
        encoder: BaseModel,
        precoder: nn.Module,
        channel: BaseChannel,
        equalizer: nn.Module,
        decoder: BaseModel,
        constraint: Optional[BaseConstraint] = None,
        num_tx_antennas: int = 2,
        num_rx_antennas: int = 2,
    ):
        """Initialize the MIMO pipeline.
        
        Args:
            encoder: Model that encodes source data
            precoder: Module that applies precoding/beamforming
            channel: MIMO channel model
            equalizer: Module that performs equalization/detection
            decoder: Model that decodes the detected signal
            constraint: Optional power or other constraint module
            num_tx_antennas: Number of transmit antennas (default: 2)
            num_rx_antennas: Number of receive antennas (default: 2)
        """
        super().__init__()
        self.encoder = encoder
        self.precoder = precoder
        self.channel = channel
        self.equalizer = equalizer
        self.decoder = decoder
        self.constraint = constraint
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
    
    def add_step(self, step: nn.Module):
        """Not applicable to MIMO pipeline."""
        raise NotImplementedError(
            "Cannot add steps directly to MIMOPipeline. "
            "Use the appropriate components in the constructor."
        )
    
    def remove_step(self, index: int):
        """Not applicable to MIMO pipeline."""
        raise NotImplementedError(
            "Cannot remove steps from MIMOPipeline. "
            "Create a new instance with the desired components."
        )
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input through the MIMO communication system.
        
        Args:
            input_data: The source data to transmit
            
        Returns:
            Dictionary containing intermediate and final outputs of the pipeline:
                - encoded: Output from the encoder
                - precoded: Output after precoding
                - constrained: Output after power constraint (if applicable)
                - received: Signal after passing through the channel
                - equalized: Signal after equalization
                - decoded: Final decoded output
        """
        # Encode input data
        encoded = self.encoder(input_data)
        
        # Apply precoding for the MIMO channel
        precoded = self.precoder(encoded, num_antennas=self.num_tx_antennas)
        
        # Apply optional power constraint
        if self.constraint is not None:
            constrained = self.constraint(precoded)
        else:
            constrained = precoded
            
        # Pass through MIMO channel
        received = self.channel(constrained)
        
        # Apply equalization/detection
        equalized = self.equalizer(received, num_antennas=self.num_rx_antennas)
        
        # Decode the equalized signal
        decoded = self.decoder(equalized)
        
        return {
            'encoded': encoded,
            'precoded': precoded,
            'constrained': constrained if self.constraint is not None else precoded,
            'received': received,
            'equalized': equalized,
            'decoded': decoded
        }
