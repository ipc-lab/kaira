"""Identity modulation module.

This module provides identity modulators and demodulators that simply pass data through
without modification. These are useful as placeholders or for testing pipelines without
actual modulation.
"""

from typing import Optional

import torch

from .base import BaseModulator, BaseDemodulator


class IdentityModulator(BaseModulator):
    """Identity modulator that passes input data through unchanged.
    
    This modulator implements the BaseModulator interface but doesn't perform
    any actual modulation. It's useful as a no-op placeholder in pipelines
    or for testing.
    
    Attributes:
        bits_per_symbol (int): Always 1, as this is a passthrough
    """
    
    def __init__(self):
        """Initialize the identity modulator."""
        super().__init__(bits_per_symbol=1)
        self.constellation = self._create_constellation()
    
    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """Pass input bits through unchanged.
        
        Args:
            bits: Input tensor of bits with any shape
                 
        Returns:
            The same tensor, unchanged
        """
        return bits
    
    def _create_constellation(self) -> torch.Tensor:
        """Create a trivial constellation (just 0 and 1).
        
        Returns:
            Complex-valued tensor of constellation points [0, 1]
        """
        return torch.tensor([0.0, 1.0], dtype=torch.complex64)


class IdentityDemodulator(BaseDemodulator):
    """Identity demodulator that passes input data through unchanged.
    
    This demodulator implements the BaseDemodulator interface but doesn't perform
    any actual demodulation. It's useful as a no-op placeholder in pipelines
    or for testing.
    
    Attributes:
        bits_per_symbol (int): Always 1, as this is a passthrough
    """
    
    def __init__(self):
        """Initialize the identity demodulator."""
        super().__init__(bits_per_symbol=1)
        self.constellation = torch.tensor([0.0, 1.0], dtype=torch.complex64)
    
    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """Pass input symbols through unchanged.
        
        Args:
            symbols: Input tensor with any shape
                    
        Returns:
            The same tensor, unchanged
        """
        return symbols
    
    def soft_demodulate(self, symbols: torch.Tensor, noise_var: float) -> torch.Tensor:
        """Pass input symbols through unchanged, ignoring noise variance.
        
        For true soft demodulation, the implementation would calculate LLRs.
        This implementation simply returns the symbols unchanged.
        
        Args:
            symbols: Input tensor with any shape
            noise_var: Noise variance (ignored in this implementation)
                    
        Returns:
            The same tensor, unchanged
        """
        return symbols
