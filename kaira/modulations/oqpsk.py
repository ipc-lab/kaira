"""Offset Quadrature Phase-Shift Keying (OQPSK) modulation scheme."""

from typing import Optional, Union

import matplotlib.pyplot as plt  # type: ignore  # type: ignore
import numpy as np
import torch

from .base import BaseDemodulator, BaseModulator
from .registry import ModulationRegistry
from .utils import plot_constellation


@ModulationRegistry.register_modulator("oqpsk")
class OQPSKModulator(BaseModulator):
    """Offset Quadrature Phase-Shift Keying (OQPSK) modulator.

    Similar to QPSK but with a half-symbol delay in the quadrature component, which results in only
    single-bit transitions and improved spectral properties.
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer
    _delayed_quad: torch.Tensor  # Type annotation for the buffer

    def __init__(self, normalize: bool = True, *args, **kwargs) -> None:
        """Initialize the OQPSK modulator.

        Args:
            normalize: If True, normalize constellation to unit energy
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # OQPSK constellation (same as QPSK but with offset timing)
        re_part = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float) * self._normalization
        im_part = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float) * self._normalization
        self.register_buffer("constellation", torch.complex(re_part, im_part))

        # Bit patterns for each symbol
        bit_patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        self.register_buffer("bit_patterns", bit_patterns)

        # Store delayed quadrature value for stateful modulation
        self.register_buffer("_delayed_quad", torch.tensor(0.0))

        self._bits_per_symbol = 2  # OQPSK has 2 bits per symbol

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bit pairs to OQPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., 2*N)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Complex tensor of OQPSK symbols with shape (..., N)

        Note:
            For simplicity, this implementation models the offset by
            applying the delay at the symbol level rather than at the
            pulse shaping level, which would be done in a real system.
        """
        # Ensure input length is even
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]
        if bit_len % 2 != 0:
            raise ValueError("Input bit length must be even for OQPSK modulation")

        # Reshape to pairs of bits
        x_reshaped = x.reshape(*batch_shape, -1, 2)
        x_reshaped.shape[-2]

        # Separate in-phase and quadrature bits
        in_phase_bits = x_reshaped[..., 0]  # (..., N)
        quad_bits = x_reshaped[..., 1]  # (..., N)

        # Map bits to amplitudes (0->1.0, 1->-1.0 after normalization)
        in_phase = (1.0 - 2.0 * in_phase_bits) * self._normalization  # (..., N)
        quad = (1.0 - 2.0 * quad_bits) * self._normalization  # (..., N)

        # Apply half-symbol delay to quadrature component by shifting
        # For first symbol, use the stored delayed value
        prev_quad = self._delayed_quad.expand(batch_shape)  # Pass batch_shape as a single tuple argument

        # Construct output: first symbol uses previous quad bit, last quad bit is stored
        delayed_quad = torch.cat([prev_quad.unsqueeze(-1), quad[..., :-1]], dim=-1)

        # Store last quad bit for next call
        if self.training:
            self._delayed_quad = quad[..., -1].detach().mean()

        # Combine to form complex symbols
        return torch.complex(in_phase, delayed_quad)

    def reset_state(self) -> None:
        """Reset internal state (delayed quadrature value)."""
        self._delayed_quad.fill_(0.0)

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the OQPSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        labels = []
        for i in range(4):
            bit_pattern = self.bit_patterns[i]
            labels.append(f"{int(bit_pattern[0])}{int(bit_pattern[1])}")

        fig, _ = plot_constellation(self.constellation, labels=labels, title="OQPSK Constellation", **kwargs)
        return fig


@ModulationRegistry.register_demodulator("oqpsk")
class OQPSKDemodulator(BaseDemodulator):
    """Offset Quadrature Phase-Shift Keying (OQPSK) demodulator."""

    def __init__(self, normalize: bool = True, *args, **kwargs) -> None:
        """Initialize the OQPSK demodulator.

        Args:
            normalize: If True, assume normalized constellation with unit energy
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        self._bits_per_symbol = 2  # OQPSK has 2 bits per symbol

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate OQPSK symbols.

        Args:
            y: Received tensor of OQPSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # Extract real and imaginary parts
        y_real = y.real
        y_imag = y.imag
        batch_shape = y.shape

        if noise_var is None:
            # Hard decision: independent decisions for I and Q
            bits_real = (y_real >= 0).float()  # 1 if positive, 0 if negative
            bits_imag = (y_imag >= 0).float()  # 1 if positive, 0 if negative

            return torch.cat([bits_real.reshape(*batch_shape, 1), bits_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)
        else:
            # Soft decision: LLRs
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape)

            # OQPSK demodulation is same as QPSK for LLR calculation
            # since I and Q are orthogonal
            llr_real = 2 * y_real * self._normalization / noise_var
            llr_imag = 2 * y_imag * self._normalization / noise_var

            return torch.cat([llr_real.reshape(*batch_shape, 1), llr_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)
