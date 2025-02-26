"""Channel Composition and Pipeline Support."""

import torch
from typing import List, Union, Dict, Any, Optional
from kaira.core import BaseChannel
import numpy as np

class ChannelPipeline(BaseChannel):
    """Pipeline for chaining multiple channel models sequentially.
    
    This class allows combining multiple channel models in sequence to model
    complex communication systems with multiple impairments.
    
    Args:
        channels (list): List of BaseChannel instances to apply in sequence
        
    Example:
        >>> awgn = AWGNChannel(snr_db=30)
        >>> phase_noise = PhaseNoiseChannel(phase_noise_variance=0.01)
        >>> pipeline = ChannelPipeline([awgn, phase_noise])
        >>> y = pipeline(x)  # Applies AWGN followed by phase noise
    """
    
    def __init__(self, channels: List[BaseChannel]):
        super().__init__()
        if not all(isinstance(channel, BaseChannel) for channel in channels):
            raise TypeError("All elements in channels must be BaseChannel instances")
        self.channels = torch.nn.ModuleList(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all channels sequentially to the input.
        
        Args:
            x (torch.Tensor): Input signal tensor
            
        Returns:
            torch.Tensor: Output after passing through all channels
        """
        output = x
        for channel in self.channels:
            output = channel(output)
        return output
    
    def __len__(self):
        """Return the number of channels in the pipeline."""
        return len(self.channels)
    
    def __getitem__(self, idx):
        """Access a specific channel by index."""
        return self.channels[idx]
    
    def append(self, channel: BaseChannel):
        """Add a channel to the end of the pipeline.
        
        Args:
            channel (BaseChannel): Channel to add
        """
        if not isinstance(channel, BaseChannel):
            raise TypeError("Channel must be a BaseChannel instance")
        self.channels.append(channel)
        return self


class ParallelChannels(BaseChannel):
    """Apply multiple channels in parallel and combine outputs.
    
    This class applies multiple channel models to the same input and 
    combines the outputs using a specified combination function.
    
    Args:
        channels (list): List of BaseChannel instances to apply in parallel
        combination_fn (callable): Function to combine outputs (default: sum)
        
    Example:
        >>> path1 = RayleighChannel(snr_db=20)
        >>> path2 = RicianChannel(k_factor=3, snr_db=25)
        >>> # Model two-path reception (diversity)
        >>> diversity = ParallelChannels([path1, path2])
        >>> y = diversity(x)  # Combines outputs from both channels
    """
    
    def __init__(self, channels: List[BaseChannel], combination_fn=torch.add):
        super().__init__()
        if not all(isinstance(channel, BaseChannel) for channel in channels):
            raise TypeError("All elements in channels must be BaseChannel instances")
        self.channels = torch.nn.ModuleList(channels)
        self.combination_fn = combination_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all channels in parallel to the input and combine outputs.
        
        Args:
            x (torch.Tensor): Input signal tensor
            
        Returns:
            torch.Tensor: Combined output from all channels
        """
        # Apply each channel to the input
        outputs = [channel(x) for channel in self.channels]
        
        # Combine outputs using the combination function
        # Start with the first output and combine with the rest
        result = outputs[0]
        for output in outputs[1:]:
            result = self.combination_fn(result, output)
            
        return result
