"""Quadrature Amplitude Modulation (QAM) schemes."""

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch

from .base import BaseDemodulator, BaseModulator
from .registry import ModulationRegistry
from .utils import binary_to_gray, plot_constellation


@ModulationRegistry.register_modulator()
class QAMModulator(BaseModulator):
    """Quadrature Amplitude Modulation (QAM) modulator.

    Maps groups of bits to constellation points with different amplitudes and phases.
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer

    def __init__(self, order: Literal[4, 16, 64, 256], gray_coding: bool = True, normalize: bool = True) -> None:
        """Initialize the QAM modulator.

        Args:
            order: Modulation order (must be a perfect square and power of 4)
            gray_coding: Whether to use Gray coding for mapping
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()

        # Validate order is a perfect square and a power of 4
        sqrt_order = int(np.sqrt(order))
        if sqrt_order**2 != order or order not in (4, 16, 64, 256):
            raise ValueError(f"QAM order must be a perfect square and power of 4, got {order}")

        self.order = order
        self.gray_coding = gray_coding
        self.normalize = normalize
        self._bits_per_symbol: int = int(np.log2(order))
        self._k: int = sqrt_order  # Number of points on each dimension

        # Create QAM constellation
        self._create_constellation()

    def _create_constellation(self) -> None:
        """Create the QAM constellation mapping."""
        # Generate base grid for QAM
        k = self._k
        base_levels = torch.arange(-(k - 1), k, 2, dtype=torch.float)

        # Create rectangular grid
        real_parts = torch.repeat_interleave(base_levels, k)
        imag_parts = base_levels.repeat(k)

        # Create complex constellation
        constellation = torch.complex(real_parts, imag_parts)

        if self.normalize:
            # Normalize to unit average energy
            energy = torch.mean(torch.abs(constellation) ** 2)
            constellation = constellation / torch.sqrt(energy)

        # Create bit pattern mapping
        bit_patterns = torch.zeros(self.order, self._bits_per_symbol)

        # Apply Gray coding if requested
        if self.gray_coding:
            # Apply Gray coding separately to real and imaginary indices
            for i in range(k):
                i_gray = binary_to_gray(i)
                for j in range(k):
                    j_gray = binary_to_gray(j)
                    idx = i * k + j

                    # Merge binary patterns
                    bits_i = format(i_gray, f"0{self._bits_per_symbol//2}b")
                    bits_j = format(j_gray, f"0{self._bits_per_symbol//2}b")

                    for b, bit in enumerate(bits_i + bits_j):
                        bit_patterns[idx, b] = int(bit)
        else:
            # Standard binary coding
            for i in range(self.order):
                bin_str = format(i, f"0{self._bits_per_symbol}b")
                for j, bit in enumerate(bin_str):
                    bit_patterns[i, j] = int(bit)

        # Register buffers directly with the computed values
        self.register_buffer("constellation", constellation)
        self.register_buffer("bit_patterns", bit_patterns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bit groups to QAM symbols.

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

        # Map indices to constellation points
        return self.constellation[indices]

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the QAM constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        labels = []
        for i in range(self.order):
            bit_pattern = self.bit_patterns[i]
            bit_str = "".join(str(int(bit)) for bit in bit_pattern)
            labels.append(bit_str)

        return plot_constellation(self.constellation, labels=labels, title=f"{self.order}-QAM Constellation", **kwargs)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol


@ModulationRegistry.register_demodulator()
class QAMDemodulator(BaseDemodulator):
    """Quadrature Amplitude Modulation (QAM) demodulator."""

    def __init__(self, order: Literal[4, 16, 64, 256], gray_coding: bool = True, normalize: bool = True) -> None:
        """Initialize the QAM demodulator.

        Args:
            order: Modulation order (must be a perfect square and power of 4)
            gray_coding: Whether Gray coding was used for mapping
            normalize: If True, assumes normalized constellation
        """
        super().__init__()
        self.order = order
        self.gray_coding = gray_coding
        self.normalize = normalize
        self._bits_per_symbol: int = int(np.log2(order))

        # Create reference modulator to access constellation
        self.modulator = QAMModulator(order, gray_coding, normalize)

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate QAM symbols.

        Args:
            y: Received tensor of QAM symbols
            noise_var: Noise variance for soft demodulation (optional)

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        constellation = self.modulator.constellation
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]

        if noise_var is None:
            # Hard decision: find closest constellation point
            expanded_y = y.unsqueeze(-1)  # (..., N, 1)
            expanded_const = constellation.expand(*([1] * len(batch_shape)), symbol_shape, self.order)  # (..., N, order)

            # Calculate Euclidean distances in complex plane
            distances = torch.abs(expanded_y - expanded_const)
            closest_indices = torch.argmin(distances, dim=-1)  # (..., N)

            # Map to bit patterns using the modulator's bit patterns
            bits = torch.zeros(
                (*batch_shape, symbol_shape, self._bits_per_symbol),
                dtype=torch.float,
                device=y.device,
            )
            for i in range(self.order):
                bits = torch.where(
                    closest_indices.unsqueeze(-1) == i,
                    self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, self._bits_per_symbol),
                    bits,
                )

            return bits.reshape(*batch_shape, -1)
        else:
            # Soft decision: LLR calculation
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)

            # Calculate LLRs for each bit position
            llrs = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), device=y.device)

            # For each bit position
            for bit_idx in range(self._bits_per_symbol):
                # Create masks for symbols where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get constellation points for each bit value
                const_bit_0 = constellation[bit_0_mask]
                const_bit_1 = constellation[bit_1_mask]

                # Calculate minimum distance for each bit value
                min_dist_0 = self._min_distance_to_points(y, const_bit_0, noise_var)
                min_dist_1 = self._min_distance_to_points(y, const_bit_1, noise_var)

                # Calculate LLR: log(P(bit=0)/P(bit=1))
                llrs[..., bit_idx] = min_dist_1 - min_dist_0

            return llrs.reshape(*batch_shape, -1)

    def _min_distance_to_points(self, y: torch.Tensor, points: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        """Calculate minimum (negative) distance to constellation points.

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
        distances = -torch.abs(y_expanded - points_expanded) ** 2 / noise_var_expanded

        # Return maximum (least negative) value
        return torch.max(distances, dim=-1)[0]

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QAM symbol."""
        return self._bits_per_symbol
