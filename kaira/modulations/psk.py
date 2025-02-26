"""Phase-Shift Keying (PSK) modulation schemes."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from .base import Modulator, Demodulator


class BPSKModulator(Modulator):
    """Binary Phase-Shift Keying (BPSK) modulator.
    
    Maps binary inputs (0, 1) to constellation points (-1, 1).
    """
    
    def __init__(self) -> None:
        """Initialize the BPSK modulator."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate binary inputs to BPSK symbols.
        
        Args:
            x: Input tensor of bits with shape (..., N)
            
        Returns:
            Tensor of BPSK symbols with shape (..., N)
        """
        # Convert binary 0/1 to -1/+1
        return 2 * x.float() - 1
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per BPSK symbol."""
        return 1


class BPSKDemodulator(Demodulator):
    """Binary Phase-Shift Keying (BPSK) demodulator."""
    
    def __init__(self) -> None:
        """Initialize the BPSK demodulator."""
        super().__init__()
    
    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate BPSK symbols.
        
        Args:
            y: Received tensor of BPSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            
        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        if noise_var is None:
            # Hard decision: y < 0 -> 0, y >= 0 -> 1
            return (y >= 0).float()
        else:
            # Soft decision: LLR calculation
            return 2 * y / noise_var
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per BPSK symbol."""
        return 1


class QPSKModulator(Modulator):
    """Quadrature Phase-Shift Keying (QPSK) modulator.
    
    Maps pairs of bits to complex constellation points in QPSK modulation.
    """
    
    def __init__(self, normalize: bool = True) -> None:
        """Initialize the QPSK modulator.
        
        Args:
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1/np.sqrt(2) if normalize else 1.0
        
        # QPSK mapping table: pairs of bits to complex symbols
        # Format: [b0b1] -> symbol
        # 00 -> (1+1j)
        # 01 -> (1-1j)
        # 10 -> (-1+1j)
        # 11 -> (-1-1j)
        re_part = torch.tensor([1, 1, -1, -1], dtype=torch.float) * self._normalization
        im_part = torch.tensor([1, -1, 1, -1], dtype=torch.float) * self._normalization
        self.register_buffer('symbols', torch.complex(re_part, im_part))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bit pairs to QPSK symbols.
        
        Args:
            x: Input tensor of bits with shape (..., 2*N)
            
        Returns:
            Complex tensor of QPSK symbols with shape (..., N)
        """
        # Ensure input length is even
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]
        assert bit_len % 2 == 0, "Input bit length must be even for QPSK"
        
        # Reshape to pairs of bits
        x_reshaped = x.reshape(*batch_shape, -1, 2)
        
        # Convert bit pairs to indices
        indices = x_reshaped[..., 0] * 2 + x_reshaped[..., 1]
        
        # Map indices to symbols
        return self.symbols[indices]
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QPSK symbol."""
        return 2


class QPSKDemodulator(Demodulator):
    """Quadrature Phase-Shift Keying (QPSK) demodulator."""
    
    def __init__(self, normalize: bool = True) -> None:
        """Initialize the QPSK demodulator.
        
        Args:
            normalize: If True, assume normalized constellation with unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1/np.sqrt(2) if normalize else 1.0
    
    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate QPSK symbols.
        
        Args:
            y: Received tensor of QPSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            
        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # Extract real and imaginary parts
        y_real = y.real
        y_imag = y.imag
        batch_shape = y.shape
        
        if noise_var is None:
            # Hard decision
            bits_real = (y_real >= 0).float()
            bits_imag = (y_imag >= 0).float()
            return torch.cat([bits_real.reshape(*batch_shape, 1), 
                              bits_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)
        else:
            # Soft decision: LLRs
            llr_real = 2 * y_real * self._normalization / noise_var
            llr_imag = 2 * y_imag * self._normalization / noise_var
            return torch.cat([llr_real.reshape(*batch_shape, 1), 
                              llr_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QPSK symbol."""
        return 2
