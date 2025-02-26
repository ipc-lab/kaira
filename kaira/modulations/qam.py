"""Quadrature Amplitude Modulation (QAM) schemes."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Literal
from .base import Modulator, Demodulator


class QAMModulator(Modulator):
    """Quadrature Amplitude Modulation (QAM) modulator.
    
    Supports 16-QAM, 64-QAM, and 256-QAM.
    """
    
    def __init__(self, order: Literal[16, 64, 256], normalize: bool = True) -> None:
        """Initialize the QAM modulator.
        
        Args:
            order: Modulation order (16, 64, or 256)
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()
        self.order = order
        self.normalize = normalize
        self._bits_per_symbol = int(np.log2(order))
        
        # Create QAM constellation
        self._create_constellation()
    
    def _create_constellation(self) -> None:
        """Create the QAM constellation mapping."""
        # Determine constellation size in one dimension
        side = int(np.sqrt(self.order))
        
        # Create Gray-coded constellation
        # First generate values in one dimension with Gray coding
        values = torch.arange(side).float()
        gray_code = values ^ (values >> 1)
        
        # Map to constellation points (-side+1, -side+3, ..., side-1)
        constellation_1d = 2 * gray_code - (side - 1)
        
        # Create 2D constellation grid
        real_part = constellation_1d.repeat_interleave(side)
        imag_part = constellation_1d.repeat(side)
        
        # Combine into complex symbols
        constellation = torch.complex(real_part, imag_part)
        
        # Normalize if required
        if self.normalize:
            # Calculate average energy
            energy = torch.mean(torch.abs(constellation) ** 2)
            constellation = constellation / torch.sqrt(energy)
        
        # Create mapping from bit patterns to symbols
        bit_indices = torch.arange(self.order).reshape(1, -1)
        bit_patterns = torch.zeros(self._bits_per_symbol, self.order)
        
        for i in range(self._bits_per_symbol):
            bit_patterns[i] = (bit_indices >> i) & 1
        
        bit_patterns = bit_patterns.T.reshape(-1)
        
        self.register_buffer('constellation', constellation)
        self.register_buffer('bit_patterns', bit_patterns)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bits to QAM symbols.
        
        Args:
            x: Input tensor of bits with shape (..., K*N), where K is bits_per_symbol
            
        Returns:
            Complex tensor of QAM symbols with shape (..., N)
        """
        # Ensure input length is divisible by bits_per_symbol
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]
        assert bit_len % self._bits_per_symbol == 0, f"Input bit length must be divisible by {self._bits_per_symbol}"
        
        # Reshape to groups of bits_per_symbol
        x_reshaped = x.reshape(*batch_shape, -1, self._bits_per_symbol)
        
        # Convert bit groups to indices
        indices = torch.zeros((*x_reshaped.shape[:-1],), dtype=torch.long, device=x.device)
        for i in range(self._bits_per_symbol):
            indices = indices | (x_reshaped[..., i].long() << i)
        
        # Map indices to symbols
        return self.constellation[indices]
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol


class QAMDemodulator(Demodulator):
    """Quadrature Amplitude Modulation (QAM) demodulator."""
    
    def __init__(self, order: Literal[16, 64, 256], normalize: bool = True) -> None:
        """Initialize the QAM demodulator.
        
        Args:
            order: Modulation order (16, 64, or 256)
            normalize: If True, assume normalized constellation with unit energy
        """
        super().__init__()
        self.order = order
        self.normalize = normalize
        self._bits_per_symbol = int(np.log2(order))
        
        # Create reference QAM modulator to access constellation
        self.modulator = QAMModulator(order, normalize)
    
    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate QAM symbols.
        
        Args:
            y: Received tensor of QAM symbols
            noise_var: Noise variance for soft demodulation (optional)
            
        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]
        constellation = self.modulator.constellation
        
        if noise_var is None:
            # Hard decision: find closest constellation point
            expanded_y = y.unsqueeze(-1)  # (..., N, 1)
            expanded_const = constellation.expand(*([1] * len(batch_shape)), symbol_shape, self.order)  # (..., N, order)
            
            # Calculate distances to constellation points
            distances = torch.abs(expanded_y - expanded_const)**2
            closest_indices = torch.argmin(distances, dim=-1)  # (..., N)
            
            # Map to bit patterns
            bits = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), device=y.device)
            for i in range(self._bits_per_symbol):
                bits[..., i] = (closest_indices >> i) & 1
            
            return bits.reshape(*batch_shape, -1)
        else:
            # Soft decision: compute LLRs for each bit
            y_expanded = y.unsqueeze(-1).expand(*batch_shape, symbol_shape, self.order)  # (..., N, order)
            const_expanded = constellation.repeat(*batch_shape, symbol_shape)  # (..., N, order)
            
            # Calculate distances
            distances = -torch.abs(y_expanded - const_expanded)**2 / noise_var  # (..., N, order)
            
            # Calculate LLRs for each bit position
            llrs = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), device=y.device)
            
            for bit_idx in range(self._bits_per_symbol):
                # Find symbols where the bit is 0
                bit_mask = ((torch.arange(self.order, device=y.device) >> bit_idx) & 1) == 0
                
                # Max-log approximation for LLR calculation
                max_prob_0 = torch.max(distances[..., bit_mask], dim=-1)[0]
                max_prob_1 = torch.max(distances[..., ~bit_mask], dim=-1)[0]
                
                llrs[..., bit_idx] = max_prob_0 - max_prob_1
            
            return llrs.reshape(*batch_shape, -1)
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol
