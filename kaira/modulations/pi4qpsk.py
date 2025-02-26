"""Π/4-QPSK (Pi/4-QPSK) modulation scheme."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from kaira.core import BaseDemodulator, BaseModulator

from .utils import plot_constellation


class Pi4QPSKModulator(BaseModulator):
    """Π/4-QPSK modulator.

    A variant of QPSK where the constellation is rotated by π/4 radians on alternating symbols,
    providing better spectral properties.
    """

    def __init__(self, normalize: bool = True) -> None:
        """Initialize the π/4-QPSK modulator.

        Args:
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # Create the two QPSK constellations (standard and π/4 rotated)
        # Standard QPSK
        angles_standard = torch.tensor([1, 3, 5, 7]) * np.pi / 4
        re_std = torch.cos(angles_standard) * self._normalization
        im_std = torch.sin(angles_standard) * self._normalization
        self.register_buffer("constellation_std", torch.complex(re_std, im_std))

        # π/4 rotated QPSK
        angles_rotated = torch.tensor([2, 4, 6, 0]) * np.pi / 4
        re_rot = torch.cos(angles_rotated) * self._normalization
        im_rot = torch.sin(angles_rotated) * self._normalization
        self.register_buffer("constellation_rot", torch.complex(re_rot, im_rot))

        # Bit patterns for each symbol in Gray coding
        bit_patterns = torch.tensor(
            [
                [0, 0],  # π/4
                [0, 1],  # 3π/4
                [1, 1],  # 5π/4
                [1, 0],  # 7π/4
            ],
            dtype=torch.float,
        )
        self.register_buffer("bit_patterns", bit_patterns)

        # Track which constellation to use next (0=standard, 1=rotated)
        self.register_buffer("_use_rotated", torch.tensor(0, dtype=torch.long))

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

        # Convert bit pairs to indices (using Gray coding)
        indices = x_reshaped[..., 0].long() * 2 + x_reshaped[..., 1].long()

        # Alternating use of standard and rotated constellation
        output = torch.zeros(*batch_shape, symbol_len, dtype=torch.complex64, device=x.device)

        # Get current constellation selector state
        use_rotated = self._use_rotated.item()

        for i in range(symbol_len):
            if (i + use_rotated) % 2 == 0:
                # Use standard constellation
                output[..., i] = self.constellation_std[indices[..., i]]
            else:
                # Use rotated constellation
                output[..., i] = self.constellation_rot[indices[..., i]]

        # Update constellation selector for next call (if we have an odd number of symbols)
        if self.training and symbol_len % 2 == 1:
            self._use_rotated.fill_((use_rotated + symbol_len) % 2)

        return output

    def reset_state(self) -> None:
        """Reset the internal state tracking constellation alternation."""
        self._use_rotated.fill_(0)

    def plot_constellation(
        self, both: bool = True, **kwargs
    ) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
        """Plot the π/4-QPSK constellation diagram(s).

        Args:
            both: If True, returns both standard and rotated constellations
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            If both is True, returns tuple of (standard, rotated) figures;
            otherwise returns combined figure
        """
        labels = []
        for i in range(4):
            bit_pattern = self.bit_patterns[i]
            labels.append(f"{int(bit_pattern[0])}{int(bit_pattern[1])}")

        if both:
            fig1 = plot_constellation(
                self.constellation_std,
                labels=labels,
                title="π/4-QPSK Standard Constellation",
                **kwargs,
            )
            fig2 = plot_constellation(
                self.constellation_rot,
                labels=labels,
                title="π/4-QPSK Rotated Constellation",
                **kwargs,
            )
            return fig1, fig2
        else:
            # Show both constellations in one plot
            combined = torch.cat([self.constellation_std, self.constellation_rot])
            # Create alternating labels
            combined_labels = []
            for i in range(4):
                combined_labels.append(f"{labels[i]} (std)")
            for i in range(4):
                combined_labels.append(f"{labels[i]} (rot)")

            return plot_constellation(
                combined, labels=combined_labels, title="π/4-QPSK Combined Constellation", **kwargs
            )

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per π/4-QPSK symbol."""
        return 2


class Pi4QPSKDemodulator(BaseDemodulator):
    """Π/4-QPSK demodulator."""

    def __init__(self, normalize: bool = True) -> None:
        """Initialize the π/4-QPSK demodulator.

        Args:
            normalize: If True, assume normalized constellation with unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # Create reference modulator to access constellations
        self.modulator = Pi4QPSKModulator(normalize)

        # Combined constellation for demodulation
        self.register_buffer(
            "combined_constellation",
            torch.cat([self.modulator.constellation_std, self.modulator.constellation_rot]),
        )

    def forward(
        self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Demodulate π/4-QPSK symbols.

        Args:
            y: Received tensor of π/4-QPSK symbols
            noise_var: Noise variance for soft demodulation (optional)

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # For π/4-QPSK, we can demodulate using the closest point from either constellation
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]

        # Extract real and imaginary parts for simpler calculations
        y_real = y.real
        y_imag = y.imag

        if noise_var is None:
            # Hard decision: find closest point in combined constellation
            expanded_y = y.unsqueeze(-1)  # (..., N, 1)
            expanded_const = self.combined_constellation.expand(
                *([1] * len(batch_shape)), symbol_shape, 8
            )  # (..., N, 8)

            # Calculate distances to constellation points
            distances = torch.abs(expanded_y - expanded_const) ** 2
            closest_indices = torch.argmin(distances, dim=-1)  # (..., N)

            # Map closest indices to bit patterns
            # For π/4-QPSK, the bit pattern is the same for both constellations
            # So we just need to get the index modulo 4
            bit_indices = closest_indices % 4

            bits = torch.zeros((*batch_shape, symbol_shape, 2), dtype=torch.float, device=y.device)
            for i in range(4):
                mask = (bit_indices == i).unsqueeze(-1)
                bit_pattern = self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, 2)
                bits = torch.where(mask, bit_pattern, bits)

            return bits.reshape(*batch_shape, -1)
        else:
            # Soft decision: LLRs
            if not torch.is_tensor(noise_var):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)

            # For π/4-QPSK, LLRs can be approximated using the adjusted quadrant decision
            # Since the bit mapping is gray-coded, the real part determines bit 0 and
            # the imaginary part determines bit 1, regardless of rotation

            # For bit 0: negative real -> 1, positive real -> 0
            llr_bit0 = -2 * y_real * self._normalization / noise_var

            # For bit 1: negative imaginary -> 0, positive imaginary -> 1
            llr_bit1 = 2 * y_imag * self._normalization / noise_var

            return torch.cat(
                [
                    llr_bit0.reshape(*batch_shape, symbol_shape, 1),
                    llr_bit1.reshape(*batch_shape, symbol_shape, 1),
                ],
                dim=-1,
            ).reshape(*batch_shape, -1)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per π/4-QPSK symbol."""
        return 2
