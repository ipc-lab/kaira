"""Quadrature Amplitude Modulation (QAM) schemes."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Literal, Tuple
from .base import Modulator, Demodulator
from .utils import plot_constellation, generate_gray_code_mapping
import matplotlib.pyplot as plt


class QAMModulator(Modulator):
    """Quadrature Amplitude Modulation (QAM) modulator.
    
    Supports 16-QAM, 64-QAM, 256-QAM, and custom orders that are perfect squares.
    """
    
    def __init__(self, order: Union[int, Literal[16, 64, 256]], normalize: bool = True) -> None:
        """Initialize the QAM modulator.
        
        Args:
            order: Modulation order (16, 64, 256, or any perfect square)
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()
        
        # Validate order is a perfect square
        side = int(np.sqrt(order))
        if side * side != order:
            raise ValueError(f"QAM order must be a perfect square, got {order}")
        
        self.order = order
        self.normalize = normalize
        self._bits_per_symbol = int(np.log2(order))
        self._side = side
        
        # Check that order is a power of 2
        if not (order > 0 and (order & (order - 1) == 0)):
            raise ValueError(f"QAM order must be a power of 2, got {order}")
        
        # Create QAM constellation
        self._create_constellation()
    
    def _create_constellation(self) -> None:
        """Create the QAM constellation mapping with Gray coding."""
        side = self._side
        
        # Generate Gray-coded values for each dimension
        gray_indices = torch.arange(side).apply_(lambda x: x ^ (x >> 1)).float()
        
        # Map to constellation points (-side+1, -side+3, ..., side-1)
        constellation_1d = 2 * gray_indices - (side - 1)
        
        # Create 2D constellation grid
        real_part = constellation_1d.repeat_interleave(side)
        imag_part = constellation_1d.repeat(side)
        
        # Combine into complex symbols
        constellation = torch.complex(real_part, imag_part)
        
        # Generate bit mapping (Gray-coded binary)
        bit_patterns = torch.zeros(self.order, self._bits_per_symbol)
        half_bits = self._bits_per_symbol // 2
        
        # For each symbol index
        for i in range(self.order):
            # Extract row and column in constellation
            row = i // side
            col = i % side
            
            # Get Gray codes for row and column
            row_gray = row ^ (row >> 1)
            col_gray = col ^ (col >> 1)
            
            # Map to bit pattern (MSB to LSB)
            for j in range(half_bits):
                bit_patterns[i, j] = (row_gray >> (half_bits - j - 1)) & 1
                bit_patterns[i, j + half_bits] = (col_gray >> (half_bits - j - 1)) & 1
        
        # Create symbol-to-index mapping for easy lookups
        symbol_indices = torch.zeros(self.order, dtype=torch.long)
        for i in range(self.order):
            # Convert bit pattern to binary index
            idx = 0
            for j in range(self._bits_per_symbol):
                idx |= (int(bit_patterns[i, j]) << (self._bits_per_symbol - j - 1))
            symbol_indices[i] = idx
        
        # Normalize if required
        if self.normalize:
            # Calculate average energy
            energy = torch.mean(torch.abs(constellation) ** 2)
            constellation = constellation / torch.sqrt(energy)
        
        self.register_buffer('constellation', constellation)
        self.register_buffer('bit_patterns', bit_patterns)
        self.register_buffer('symbol_indices', symbol_indices)
    
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
        if bit_len % self._bits_per_symbol != 0:
            raise ValueError(f"Input bit length must be divisible by {self._bits_per_symbol}")
        
        # Reshape to groups of bits_per_symbol
        x_reshaped = x.reshape(*batch_shape, -1, self._bits_per_symbol)
        
        # Convert bit groups to indices
        indices = torch.zeros((*x_reshaped.shape[:-1],), dtype=torch.long, device=x.device)
        for i in range(self._bits_per_symbol):
            indices = indices | (x_reshaped[..., i].long() << (self._bits_per_symbol - i - 1))
        
        # Map indices to symbols
        return self.constellation[indices]
    
    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the QAM constellation diagram.
        
        Args:
            **kwargs: Additional arguments passed to plot_constellation
            
        Returns:
            Matplotlib figure object
        """
        bits = []
        for i in range(self.order):
            bit_str = ""
            for j in range(self._bits_per_symbol):
                bit_str += str(int(self.bit_patterns[i, j].item()))
            bits.append(bit_str)
            
        return plot_constellation(
            self.constellation, 
            labels=bits,
            title=f"{self.order}-QAM Constellation",
            **kwargs
        )
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol


class QAMDemodulator(Demodulator):
    """Quadrature Amplitude Modulation (QAM) demodulator."""
    
    def __init__(self, order: Union[int, Literal[16, 64, 256]], normalize: bool = True) -> None:
        """Initialize the QAM demodulator.
        
        Args:
            order: Modulation order (16, 64, 256, or any perfect square)
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
            y: Received tensor of QAM symbols with shape (..., N)
            noise_var: Noise variance for soft demodulation (optional)
            
        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
            with shape (..., N*bits_per_symbol)
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
            
            # Map to bit patterns using the modulator's bit patterns
            bits = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), dtype=torch.float, device=y.device)
            for i in range(self.order):
                bits = torch.where(
                    closest_indices.unsqueeze(-1) == i,
                    self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, self._bits_per_symbol),
                    bits
                )
            
            return bits.reshape(*batch_shape, -1)
        else:
            # Support both scalar and tensor noise variance
            if not torch.is_tensor(noise_var):
                noise_var = torch.tensor(noise_var, device=y.device)
                
            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)
            
            # More efficient LLR calculation
            llrs = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), device=y.device)
            
            # For each bit position
            for bit_idx in range(self._bits_per_symbol):
                # Create masks for symbols where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask
                
                # Get constellation points for each bit value
                const_bit_0 = constellation[bit_0_mask]
                const_bit_1 = constellation[bit_1_mask]
                
                # Calculate minimum distance to constellation points for each bit value
                min_dist_0 = self._min_distance_to_points(y, const_bit_0, noise_var)
                min_dist_1 = self._min_distance_to_points(y, const_bit_1, noise_var)
                
                # Calculate LLR: log(P(bit=0)/P(bit=1))
                llrs[..., bit_idx] = min_dist_1 - min_dist_0
            
            return llrs.reshape(*batch_shape, -1)
    
    def _min_distance_to_points(self, y: torch.Tensor, points: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        """Calculate minimum (negative) distance to a set of constellation points.
        
        Uses max-log approximation for computational efficiency.
        
        Args:
            y: Received symbols with shape (..., N)
            points: Constellation points to compare against with shape (M,)
            noise_var: Noise variance with shape (..., N)
            
        Returns:
            Minimum negative distance for each symbol in y
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]
        num_points = points.shape[0]
        
        # Reshape inputs for broadcasting
        y_expanded = y.unsqueeze(-1).expand(*batch_shape, symbol_shape, num_points)
        points_expanded = points.reshape(1, 1, -1).expand(*batch_shape, symbol_shape, num_points)
        noise_var_expanded = noise_var.unsqueeze(-1).expand(*batch_shape, symbol_shape, num_points)
        
        # Calculate distances
        distances = -torch.abs(y_expanded - points_expanded)**2 / noise_var_expanded
        
        # Return maximum (least negative) value
        return torch.max(distances, dim=-1)[0]
    
    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol
