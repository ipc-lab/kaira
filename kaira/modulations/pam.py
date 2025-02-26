"""Pulse Amplitude Modulation (PAM) schemes."""

from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from kaira.core import BaseDemodulator, BaseModulator

from .utils import binary_to_gray, gray_to_binary, plot_constellation


class PAMModulator(BaseModulator):
    """Pulse Amplitude Modulation (PAM) modulator.

    Maps groups of bits to amplitude levels for transmission.
    """

    def __init__(
        self, order: Literal[2, 4, 8, 16, 32, 64], gray_coding: bool = True, normalize: bool = True
    ) -> None:
        """Initialize the PAM modulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether to use Gray coding for mapping
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()

        # Validate order is a power of 2
        if not (order > 0 and (order & (order - 1) == 0)):
            raise ValueError(f"PAM order must be a power of 2, got {order}")

        self.order = order
        self.gray_coding = gray_coding
        self.normalize = normalize
        self._bits_per_symbol = int(np.log2(order))

        # Create PAM constellation
        self._create_constellation()

    def _create_constellation(self) -> None:
        """Create the PAM constellation mapping."""
        # Generate amplitude levels
        base_levels = torch.arange(-(self.order - 1), self.order, 2, dtype=torch.float)

        # Reorder levels if using Gray coding
        if self.gray_coding:
            indices = torch.arange(self.order)
            gray_indices = torch.tensor([binary_to_gray(i) for i in range(self.order)])
            _, sorted_indices = torch.sort(gray_indices)
            levels = base_levels[sorted_indices]
        else:
            levels = base_levels

        # Normalize if requested
        if self.normalize:
            energy = torch.mean(levels**2)
            levels = levels / torch.sqrt(energy)

        # Store as complex for consistency with other modulators
        self.register_buffer("levels", levels)
        self.register_buffer("constellation", torch.complex(levels, torch.zeros_like(levels)))

        # Create bit pattern mapping
        bit_patterns = torch.zeros(self.order, self._bits_per_symbol)

        for i in range(self.order):
            idx = i
            if self.gray_coding:
                idx = binary_to_gray(i)
            bin_str = format(idx, f"0{self._bits_per_symbol}b")
            for j, bit in enumerate(bin_str):
                bit_patterns[i, j] = int(bit)

        self.register_buffer("bit_patterns", bit_patterns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bit groups to PAM symbols.

        Args:
            x: Input tensor of bits with shape (..., K*N), where K is bits_per_symbol

        Returns:
            Complex tensor of PAM symbols with shape (..., N)
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

        # Map indices to symbols (real-valued)
        return torch.complex(self.levels[indices], torch.zeros_like(self.levels[indices]))

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the PAM constellation diagram.

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

        return plot_constellation(
            self.constellation, labels=labels, title=f"{self.order}-PAM Constellation", **kwargs
        )

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per PAM symbol."""
        return self._bits_per_symbol


class PAMDemodulator(BaseDemodulator):
    """Pulse Amplitude Modulation (PAM) demodulator."""

    def __init__(
        self, order: Literal[2, 4, 8, 16, 32, 64], gray_coding: bool = True, normalize: bool = True
    ) -> None:
        """Initialize the PAM demodulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether Gray coding was used for mapping
            normalize: If True, assumes normalized constellation
        """
        super().__init__()
        self.order = order
        self.gray_coding = gray_coding
        self.normalize = normalize
        self._bits_per_symbol = int(np.log2(order))

        # Create reference modulator to access constellation
        self.modulator = PAMModulator(order, gray_coding, normalize)

    def forward(
        self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Demodulate PAM symbols.

        Args:
            y: Received tensor of PAM symbols (complex, but only real part is used)
            noise_var: Noise variance for soft demodulation (optional)

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # PAM only uses real part
        y_real = y.real
        batch_shape = y_real.shape[:-1]
        symbol_shape = y_real.shape[-1]
        levels = self.modulator.levels

        if noise_var is None:
            # Hard decision: find closest level
            expanded_y = y_real.unsqueeze(-1)  # (..., N, 1)
            expanded_levels = levels.expand(
                *([1] * len(batch_shape)), symbol_shape, self.order
            )  # (..., N, order)

            # Calculate distances to levels
            distances = torch.abs(expanded_y - expanded_levels)
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
                    self.modulator.bit_patterns[i].expand(
                        *batch_shape, symbol_shape, self._bits_per_symbol
                    ),
                    bits,
                )

            return bits.reshape(*batch_shape, -1)
        else:
            # Soft decision: LLR calculation
            if not torch.is_tensor(noise_var):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)

            # Calculate LLRs for each bit position
            llrs = torch.zeros(
                (*batch_shape, symbol_shape, self._bits_per_symbol), device=y.device
            )

            # For each bit position
            for bit_idx in range(self._bits_per_symbol):
                # Create masks for symbols where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get levels for each bit value
                levels_bit_0 = levels[bit_0_mask]
                levels_bit_1 = levels[bit_1_mask]

                # Calculate minimum distance for each bit value (max-log approximation)
                min_dist_0 = self._min_distance_to_levels(y_real, levels_bit_0, noise_var)
                min_dist_1 = self._min_distance_to_levels(y_real, levels_bit_1, noise_var)

                # Calculate LLR: log(P(bit=0)/P(bit=1))
                llrs[..., bit_idx] = min_dist_1 - min_dist_0

            return llrs.reshape(*batch_shape, -1)

    def _min_distance_to_levels(
        self, y: torch.Tensor, levels: torch.Tensor, noise_var: torch.Tensor
    ) -> torch.Tensor:
        """Calculate minimum (negative) distance to a set of amplitude levels.

        Uses max-log approximation for computational efficiency.

        Args:
            y: Received symbols with shape (..., N)
            levels: Levels to compare against with shape (M,)
            noise_var: Noise variance with shape (..., N)

        Returns:
            Minimum negative distance for each symbol in y
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]
        num_levels = levels.shape[0]

        # Reshape inputs for broadcasting
        y_expanded = y.unsqueeze(-1).expand(*batch_shape, symbol_shape, num_levels)
        levels_expanded = levels.reshape(1, 1, -1).expand(*batch_shape, symbol_shape, num_levels)
        noise_var_expanded = noise_var.unsqueeze(-1).expand(*batch_shape, symbol_shape, num_levels)

        # Calculate distances
        distances = -torch.abs(y_expanded - levels_expanded) ** 2 / noise_var_expanded

        # Return maximum (least negative) value
        return torch.max(distances, dim=-1)[0]

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per PAM symbol."""
        return self._bits_per_symbol
