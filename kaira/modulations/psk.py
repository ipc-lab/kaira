"""Phase-Shift Keying (PSK) modulation schemes."""

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch

from .base import BaseDemodulator, BaseModulator
from .registry import ModulationRegistry
from .utils import plot_constellation


@ModulationRegistry.register_modulator()
class BPSKModulator(BaseModulator):
    """Binary Phase-Shift Keying (BPSK) modulator.

    Maps binary inputs (0, 1) to constellation points (-1, 1).
    """

    constellation: torch.Tensor  # Type annotation for the buffer

    def __init__(self) -> None:
        """Initialize the BPSK modulator."""
        super().__init__()

        # Define constellation points
        re_part = torch.tensor([1.0, -1.0])
        im_part = torch.tensor([0.0, 0.0])
        self.register_buffer("constellation", torch.complex(re_part, im_part))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate binary inputs to BPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., N)

        Returns:
            Complex tensor of BPSK symbols with shape (..., N)
        """
        # Convert binary 0/1 to -1/+1
        return torch.complex(2 * x.float() - 1, torch.zeros_like(x.float()))

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the BPSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        return plot_constellation(self.constellation, labels=["0", "1"], title="BPSK Constellation", **kwargs)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per BPSK symbol."""
        return 1


@ModulationRegistry.register_demodulator()
class BPSKDemodulator(BaseDemodulator):
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
        # Extract real part for decision (BPSK uses only real axis)
        y_real = y.real

        if noise_var is None:
            # Hard decision: y < 0 -> 0, y >= 0 -> 1
            return (y_real >= 0).float()
        else:
            # Support both scalar and tensor noise variance
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Soft decision: LLR calculation
            return 2 * y_real / noise_var

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per BPSK symbol."""
        return 1


@ModulationRegistry.register_modulator()
class QPSKModulator(BaseModulator):
    """Quadrature Phase-Shift Keying (QPSK) modulator.

    Maps pairs of bits to complex constellation points in QPSK modulation.
    """

    constellation: torch.Tensor  # Type annotation for the buffer

    def __init__(self, normalize: bool = True) -> None:
        """Initialize the QPSK modulator.

        Args:
            normalize: If True, normalize constellation to unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # QPSK mapping table: pairs of bits to complex symbols with Gray coding
        # Format: [b0b1] -> symbol
        # 00 -> (1+1j)
        # 01 -> (1-1j)
        # 10 -> (-1+1j)
        # 11 -> (-1-1j)
        re_part = torch.tensor([1, 1, -1, -1], dtype=torch.float) * self._normalization
        im_part = torch.tensor([1, -1, 1, -1], dtype=torch.float) * self._normalization
        self.register_buffer("constellation", torch.complex(re_part, im_part))

        # Bit patterns for each symbol
        bit_patterns = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
        self.register_buffer("bit_patterns", bit_patterns)

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
        if bit_len % 2 != 0:
            raise ValueError("Input bit length must be even for QPSK modulation")

        # Reshape to pairs of bits
        x_reshaped = x.reshape(*batch_shape, -1, 2)

        # Convert bit pairs to indices (using Gray coding)
        indices = x_reshaped[..., 0].to(torch.long) * 2 + x_reshaped[..., 1].to(torch.long)

        # Handle empty tensor case to avoid None indexing
        if indices.numel() == 0:
            return torch.empty((*batch_shape, 0), dtype=torch.complex64, device=x.device)

        # Map indices to symbols
        return self.constellation[indices]

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the QPSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        labels = []
        for i in range(4):
            bit_pattern = self.bit_patterns[i]
            labels.append(f"{int(bit_pattern[0])}{int(bit_pattern[1])}")

        return plot_constellation(self.constellation, labels=labels, title="QPSK Constellation", **kwargs)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QPSK symbol."""
        return 2


@ModulationRegistry.register_demodulator()
class QPSKDemodulator(BaseDemodulator):
    """Quadrature Phase-Shift Keying (QPSK) demodulator."""

    def __init__(self, normalize: bool = True) -> None:
        """Initialize the QPSK demodulator.

        Args:
            normalize: If True, assume normalized constellation with unit energy
        """
        super().__init__()
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # Create modulator to access constellation
        self.modulator = QPSKModulator(normalize)

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
            return torch.cat([bits_real.reshape(*batch_shape, 1), bits_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)
        else:
            # Support both scalar and tensor noise variance
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape)

            # Soft decision: LLRs
            llr_real = 2 * y_real * self._normalization / noise_var
            llr_imag = 2 * y_imag * self._normalization / noise_var
            return torch.cat([llr_real.reshape(*batch_shape, 1), llr_imag.reshape(*batch_shape, 1)], dim=-1).reshape(*batch_shape[:-1], -1)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per QPSK symbol."""
        return 2


@ModulationRegistry.register_modulator()
class PSKModulator(BaseModulator):
    """General M-ary Phase-Shift Keying (PSK) modulator.

    Maps groups of bits to complex constellation points around the unit circle.
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer

    def __init__(self, order: Literal[4, 8, 16, 32, 64], gray_coding: bool = True) -> None:
        """Initialize the PSK modulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether to use Gray coding for constellation mapping
        """
        super().__init__()

        # Validate order is a power of 2
        if not (order > 0 and (order & (order - 1) == 0)):
            raise ValueError(f"PSK order must be a power of 2, got {order}")

        self.order = order
        self.gray_coding = gray_coding
        self._bits_per_symbol: int = int(np.log2(order))

        # Create PSK constellation
        self._create_constellation()

    def _create_constellation(self) -> None:
        """Create the PSK constellation mapping."""
        # Generate points evenly spaced around the unit circle
        angles = torch.arange(0, self.order) * (2 * np.pi / self.order)
        re_part = torch.cos(angles)
        im_part = torch.sin(angles)
        constellation = torch.complex(re_part, im_part)

        # Create bit pattern mapping
        bit_patterns = torch.zeros(self.order, self._bits_per_symbol)

        if self.gray_coding:
            # Apply Gray coding
            for i in range(self.order):
                gray_idx = i ^ (i >> 1)  # Binary to Gray conversion
                bin_str = format(gray_idx, f"0{self._bits_per_symbol}b")
                for j, bit in enumerate(bin_str):
                    bit_patterns[i, j] = int(bit)
        else:
            # Standard binary coding
            for i in range(self.order):
                bin_str = format(i, f"0{self._bits_per_symbol}b")
                for j, bit in enumerate(bin_str):
                    bit_patterns[i, j] = int(bit)

        self.register_buffer("constellation", constellation)
        self.register_buffer("bit_patterns", bit_patterns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bit groups to PSK symbols.

        Args:
            x: Input tensor of bits with shape (..., M)

        Returns:
            Complex tensor of PSK symbols with shape (..., N)
        """
        # Ensure input length is a multiple of bits_per_symbol
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]
        if bit_len % self._bits_per_symbol != 0:
            raise ValueError(f"Input bit length must be a multiple of {self._bits_per_symbol} for PSK modulation")

        # Reshape to groups of bits
        x_reshaped = x.reshape(*batch_shape, -1, self._bits_per_symbol)

        # Convert bit groups to indices - ensure we're not creating None values
        indices = torch.zeros(x_reshaped.shape[:-1], dtype=torch.long, device=x.device)
        for i in range(self._bits_per_symbol):
            # Using explicit .to(torch.long) instead of .long() to ensure proper type conversion
            power = 2 ** (self._bits_per_symbol - 1 - i)
            bit_value = x_reshaped[..., i].to(torch.long)
            indices = indices + (bit_value * power)

        # Map indices to symbols - handle edge cases to avoid None indexing
        if indices.numel() == 0:
            return torch.empty((*batch_shape, 0), dtype=torch.complex64, device=x.device)

        return self.constellation[indices]

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the PSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        labels = []
        for i in range(self.order):
            bit_pattern = self.bit_patterns[i]
            label = "".join(str(int(bit)) for bit in bit_pattern)
            labels.append(label)

        return plot_constellation(self.constellation, labels=labels, title=f"{self.order}-PSK Constellation", **kwargs)

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per PSK symbol."""
        return self._bits_per_symbol


@ModulationRegistry.register_demodulator()
class PSKDemodulator(BaseDemodulator):
    """General M-ary Phase-Shift Keying (PSK) demodulator.

    Demodulates complex constellation points back to bits.
    """

    def __init__(self, order: Literal[4, 8, 16, 32, 64], gray_coding: bool = True) -> None:
        """Initialize the PSK demodulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether Gray coding was used for constellation mapping
        """
        super().__init__()
        self.order = order
        self.gray_coding = gray_coding
        self._bits_per_symbol: int = int(np.log2(order))

        # Create modulator to access constellation
        self.modulator = PSKModulator(order, gray_coding)

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate PSK symbols.

        Args:
            y: Received tensor of PSK symbols
            noise_var: Noise variance for soft demodulation (optional)

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]
        constellation = self.modulator.constellation

        if noise_var is None:
            # Hard decision: find closest constellation point
            # Compute phase angle of received symbols
            y_angle = torch.angle(y)

            # Compute phase angles of constellation points
            const_angles = torch.angle(constellation)

            # Find closest angle (considering circular distance)
            expanded_y_angle = y_angle.unsqueeze(-1)  # (..., N, 1)
            expanded_const_angle = const_angles.expand(*([1] * len(batch_shape)), symbol_shape, self.order)  # (..., N, order)

            # Calculate circular distance
            angle_diff = torch.abs((expanded_y_angle - expanded_const_angle + np.pi) % (2 * np.pi) - np.pi)
            closest_indices = torch.argmin(angle_diff, dim=-1)  # (..., N)

            # Map to bit patterns using the modulator's bit patterns
            bits = torch.zeros(
                (*batch_shape, symbol_shape, self._bits_per_symbol),
                dtype=torch.float,
                device=y.device,
            )
            for i in range(self.order):
                mask = (closest_indices == i).unsqueeze(-1)
                bit_pattern = self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, self._bits_per_symbol)
                bits = torch.where(mask, bit_pattern, bits)

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

            for bit_idx in range(self._bits_per_symbol):
                # Create masks for symbols where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get constellation points for each bit value
                const_bit_0 = constellation[bit_0_mask]
                const_bit_1 = constellation[bit_1_mask]

                # Calculate minimum distance for each bit value using max-log approximation
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
        distances = -torch.abs(y_expanded - points_expanded) ** 2 / noise_var_expanded

        # Return maximum (least negative) value - fix the type error by explicitly dealing with tuple return
        max_values, _ = torch.max(distances, dim=-1)
        return max_values

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per PSK symbol."""
        return self._bits_per_symbol
