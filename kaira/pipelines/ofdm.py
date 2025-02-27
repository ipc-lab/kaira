"""OFDM module for Kaira.

This module contains the OFDMPipeline, which implements a complete Orthogonal
Frequency Division Multiplexing system with multi-carrier modulation.
"""

from typing import Dict, Optional, Tuple, List, Any
import torch
import torch.nn as nn

from .base import BasePipeline, BaseChannel, BaseModel


class OFDMPipeline(BasePipeline):
    """A pipeline for Orthogonal Frequency Division Multiplexing (OFDM) systems.
    
    OFDM is a frequency-division multiplexing method used in many wireless and wireline
    communication systems. This pipeline models the complete OFDM transmission and
    reception process, including pilot insertion, IFFT/FFT operations, cyclic prefix 
    addition/removal, and channel estimation.
    
    Attributes:
        encoder (BaseModel): Encodes the source data
        mapper (nn.Module): Maps encoded bits to QAM/PSK symbols
        pilot_inserter (nn.Module): Inserts pilot symbols for channel estimation
        ifft (nn.Module): Performs IFFT for OFDM modulation
        cp_adder (nn.Module): Adds cyclic prefix to OFDM symbols
        channel (BaseChannel): Models the wireless channel effects
        cp_remover (nn.Module): Removes cyclic prefix from received symbols
        fft (nn.Module): Performs FFT for OFDM demodulation
        channel_estimator (nn.Module): Estimates channel using pilot symbols
        equalizer (nn.Module): Performs frequency-domain equalization
        demapper (nn.Module): Maps QAM/PSK symbols back to bits
        decoder (BaseModel): Decodes the received bits
    """
    
    def __init__(
        self,
        encoder: BaseModel,
        mapper: nn.Module,
        pilot_inserter: nn.Module,
        ifft: nn.Module,
        cp_adder: nn.Module,
        channel: BaseChannel,
        cp_remover: nn.Module,
        fft: nn.Module,
        channel_estimator: nn.Module,
        equalizer: nn.Module,
        demapper: nn.Module,
        decoder: BaseModel,
        num_subcarriers: int = 64,
        cp_length: int = 16,
    ):
        """Initialize the OFDM pipeline.
        
        Args:
            encoder: Model that encodes source data
            mapper: Module that maps bits to QAM/PSK symbols
            pilot_inserter: Module that inserts pilot symbols
            ifft: Module that performs IFFT for OFDM modulation
            cp_adder: Module that adds cyclic prefix
            channel: Module that models the wireless channel
            cp_remover: Module that removes cyclic prefix
            fft: Module that performs FFT for OFDM demodulation
            channel_estimator: Module that estimates channel from pilots
            equalizer: Module that performs frequency-domain equalization
            demapper: Module that maps symbols back to bits
            decoder: Model that decodes the received bits
            num_subcarriers: Number of OFDM subcarriers (default: 64)
            cp_length: Cyclic prefix length (default: 16)
        """
        super().__init__()
        self.encoder = encoder
        self.mapper = mapper
        self.pilot_inserter = pilot_inserter
        self.ifft = ifft
        self.cp_adder = cp_adder
        self.channel = channel
        self.cp_remover = cp_remover
        self.fft = fft
        self.channel_estimator = channel_estimator
        self.equalizer = equalizer
        self.demapper = demapper
        self.decoder = decoder
        
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        
    def add_step(self, step: nn.Module):
        """Not applicable to OFDM pipeline."""
        raise NotImplementedError(
            "Cannot add steps directly to OFDMPipeline. "
            "Use the appropriate components in the constructor."
        )
    
    def remove_step(self, index: int):
        """Not applicable to OFDM pipeline."""
        raise NotImplementedError(
            "Cannot remove steps from OFDMPipeline. "
            "Create a new instance with the desired components."
        )
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input through the OFDM communication system.
        
        Args:
            input_data: The source data to transmit
            
        Returns:
            Dictionary containing intermediate and final outputs:
                - encoded: Output from the encoder
                - mapped: QAM/PSK symbols
                - pilot_inserted: Symbols with pilots inserted
                - time_domain: Time-domain OFDM symbols after IFFT
                - cp_added: OFDM symbols with cyclic prefix
                - received: Signal after passing through the channel
                - cp_removed: Received signal with cyclic prefix removed
                - freq_domain: Frequency-domain symbols after FFT
                - channel_estimate: Estimated channel frequency response
                - equalized: Equalized frequency-domain symbols
                - demapped: Demapped bits
                - decoded: Final decoded output
        """
        # Transmitter side
        encoded = self.encoder(input_data)
        mapped = self.mapper(encoded)
        pilot_inserted = self.pilot_inserter(mapped, self.num_subcarriers)
        time_domain = self.ifft(pilot_inserted, self.num_subcarriers)
        cp_added = self.cp_adder(time_domain, self.cp_length)
        
        # Channel
        received = self.channel(cp_added)
        
        # Receiver side
        cp_removed = self.cp_remover(received, self.cp_length)
        freq_domain = self.fft(cp_removed, self.num_subcarriers)
        channel_estimate = self.channel_estimator(freq_domain, pilot_pattern=self.pilot_inserter.get_pattern())
        equalized = self.equalizer(freq_domain, channel_estimate)
        demapped = self.demapper(equalized)
        decoded = self.decoder(demapped)
        
        return {
            'encoded': encoded,
            'mapped': mapped,
            'pilot_inserted': pilot_inserted,
            'time_domain': time_domain,
            'cp_added': cp_added,
            'received': received,
            'cp_removed': cp_removed,
            'freq_domain': freq_domain,
            'channel_estimate': channel_estimate,
            'equalized': equalized,
            'demapped': demapped,
            'decoded': decoded
        }
