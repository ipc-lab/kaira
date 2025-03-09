"""Π/4-QPSK modulation scheme."""

from typing import Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch

from .base import BaseDemodulator, BaseModulator
from .utils import plot_constellation
from .registry import ModulationRegistry


@ModulationRegistry.register_modulator("pi4qpsk")
class Pi4QPSKModulator(BaseModulator):
    """Π/4-QPSK (π/4 shifted QPSK) modulator.

    A variant of QPSK where the constellation is rotated by π/4 radians on alternating symbols,
    providing improved envelope properties.
    """

    qpsk: torch.Tensor  # Type annotation for the buffer
    qpsk_rotated: torch.Tensor  # Type annotation for the buffer
    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer
    _use_rotated: torch.Tensor  # Type annotation for the buffer

    def __init__(self) -> None:
        """Initialize the π/4-QPSK modulator."""
        super().__init__()
        self._bits_per_symbol: int = 2

        # Create two QPSK constellations, one rotated by π/4
        self._create_constellations()

        # Keep track of which constellation to use
        self.register_buffer("_use_rotated", torch.tensor(False))

    def _create_constellations(self) -> None:
        """Create standard and rotated QPSK constellations."""
        # Standard QPSK
        angles = torch.tensor([1, 3, 5, 7]) * np.pi / 4
        re_part = torch.cos(angles)
        im_part = torch.sin(angles)
        qpsk = torch.complex(re_part, im_part)

        # π/4 rotated QPSK
        angles_rotated = torch.tensor([0, 2, 4, 6]) * np.pi / 4
        re_part_rotated = torch.cos(angles_rotated)
        im_part_rotated = torch.sin(angles_rotated)
        qpsk_rotated = torch.complex(re_part_rotated, im_part_rotated)

        # Store both constellations
        self.register_buffer("qpsk", qpsk)
        self.register_buffer("qpsk_rotated", qpsk_rotated)

        # Combined constellation for visualization
        self.register_buffer("constellation", torch.cat([qpsk, qpsk_rotated]))

        # Bit patterns for each symbol (same for both constellations)
        bit_patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        self.register_buffer("bit_patterns", bit_patterns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bit pairs to π/4-QPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., 2*N)

        Returns:
            Complex tensor of π/4-QPSK symbols with shape (..., N)
        """
        # Ensure input length is even
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]
        if bit_len % 2 != 0:
            raise ValueError("Input bit length must be even for π/4-QPSK modulation")

        # Reshape to pairs of bits
        x_reshaped = x.reshape(*batch_shape, -1, 2)
        symbol_len = x_reshaped.shape[-2]

        # Convert bit pairs to indices
        indices = x_reshaped[..., 0].long() * 2 + x_reshaped[..., 1].long()

        # Outputs array
        y = torch.zeros(*batch_shape, symbol_len, dtype=torch.complex64, device=x.device)

        # Alternate between standard and rotated constellation for each symbol
        use_rotated = self._use_rotated.clone()

        for i in range(symbol_len):
            if use_rotated:
                y[..., i] = self.qpsk_rotated[indices[..., i]]
            else:
                y[..., i] = self.qpsk[indices[..., i]]
            use_rotated = ~use_rotated

        # Store final state for next call if in training
        if self.training:
            self._use_rotated = use_rotated.detach()

        return y

    def reset_state(self) -> None:
        """Reset internal state (constellation alternation)."""
        self._use_rotated.fill_(False)

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the π/4-QPSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        labels = []
        for pattern in self.bit_patterns:
            bit_str = f"{int(pattern[0])}{int(pattern[1])}"
            # Add each bit pattern twice (once for each constellation)
            labels.extend([bit_str + "⊙", bit_str + "⊗"])

        return plot_constellation(self.constellation, labels=labels, title="π/4-QPSK Constellation", **kwargs)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per π/4-QPSK symbol."""
        return self._bits_per_symbol


@ModulationRegistry.register_demodulator("pi4qpsk")
class Pi4QPSKDemodulator(BaseDemodulator):
    """Π/4-QPSK demodulator."""

    _use_rotated: torch.Tensor  # Type annotation for the buffer

    def __init__(self) -> None:
        """Initialize the π/4-QPSK demodulator."""
        super().__init__()
        self._bits_per_symbol: int = 2

        # Create reference modulator to access constellations
        self.modulator = Pi4QPSKModulator()

        # Keep track of which constellation to use for demodulation
        self.register_buffer("_use_rotated", torch.tensor(False))

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate π/4-QPSK symbols.

        Args:
            y: Received tensor of π/4-QPSK symbols
            noise_var: Noise variance for soft demodulation (optional)

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]

        # Prepare output
        if noise_var is None:
            # Hard decisions: two bits per symbol
            bits = torch.zeros(*batch_shape, symbol_shape, 2, dtype=torch.float, device=y.device)
        else:
            # Soft decisions: two LLRs per symbol
            bits = torch.zeros(*batch_shape, symbol_shape, 2, dtype=torch.float, device=y.device)
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)
            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)

        # Get constellations from modulator
        qpsk = self.modulator.qpsk
        qpsk_rotated = self.modulator.qpsk_rotated

        # Demodulate each symbol using the appropriate constellation
        use_rotated = self._use_rotated.clone()

        for i in range(symbol_shape):
            # Select current constellation
            constellation = qpsk_rotated if use_rotated else qpsk

            # Process current symbol
            if noise_var is None:
                # Hard decision
                distances = torch.abs(y[..., i : i + 1] - constellation.unsqueeze(0))
                closest_idx = torch.argmin(distances, dim=-1)
                bits[..., i, :] = self.modulator.bit_patterns[closest_idx]
            else:
                # Soft decision (LLR calculation)
                current_noise_var = noise_var[..., i]

                # Calculate LLRs for each bit position
                for bit_idx in range(2):
                    # Create masks for symbols where bit is 0 or 1
                    bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                    bit_1_mask = ~bit_0_mask

                    # Get constellation points for each bit value
                    const_bit_0 = constellation[bit_0_mask]
                    const_bit_1 = constellation[bit_1_mask]

                    # Calculate distances for each bit value
                    expanded_y = y[..., i : i + 1]  # Keep dimensions for broadcasting

                    # Distance to constellation points where bit is 0
                    distances_0 = -torch.abs(expanded_y - const_bit_0.unsqueeze(0)) ** 2
                    min_dist_0 = torch.max(distances_0, dim=-1)[0] / current_noise_var

                    # Distance to constellation points where bit is 1
                    distances_1 = -torch.abs(expanded_y - const_bit_1.unsqueeze(0)) ** 2
                    min_dist_1 = torch.max(distances_1, dim=-1)[0] / current_noise_var

                    # LLR: log(P(bit=0)/P(bit=1))
                    bits[..., i, bit_idx] = min_dist_0 - min_dist_1

            # Toggle constellation for next symbol
            use_rotated = ~use_rotated

        # Store state for next call if in training
        if self.training:
            self._use_rotated = use_rotated.detach()

        return bits.reshape(*batch_shape, -1)

    def reset_state(self) -> None:
        """Reset internal state (constellation alternation)."""
        self._use_rotated.fill_(False)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per π/4-QPSK symbol."""
        return self._bits_per_symbol
