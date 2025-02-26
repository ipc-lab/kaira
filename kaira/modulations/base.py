"""Base classes for modulation schemes."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


class Modulator(nn.Module, ABC):
    """Base class for all modulators.
    
    Modulators convert bits or symbols into complex constellation points
    for transmission over a communication channel.
    """
    
    def __init__(self) -> None:
        """Initialize the modulator."""
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input bits or symbols to constellation points.
        
        Args:
            x: Input tensor of bits or symbols
            
        Returns:
            Complex-valued tensor of constellation points
        """
        pass
    
    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        """Number of bits represented by each modulation symbol."""
        pass


class Demodulator(nn.Module, ABC):
    """Base class for all demodulators.
    
    Demodulators convert received complex constellation points back to bits or
    reliability metrics (e.g., log-likelihood ratios).
    """
    
    def __init__(self) -> None:
        """Initialize the demodulator."""
        super().__init__()
    
    @abstractmethod
    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate received constellation points.
        
        Args:
            y: Received complex-valued tensor containing constellation points
            noise_var: Noise variance for soft demodulation, if applicable
            
        Returns:
            Tensor of demodulated bits or log-likelihood ratios
        """
        pass
    
    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        """Number of bits represented by each modulation symbol."""
        pass
