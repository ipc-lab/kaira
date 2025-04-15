"""Differential Phase-Shift Keying (DPSK) modulation schemes."""

from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt  # type: ignore  # type: ignore
import numpy as np
import torch

from .base import BaseDemodulator, BaseModulator
from .registry import ModulationRegistry
from .utils import plot_constellation


@ModulationRegistry.register_modulator()
class DPSKModulator(BaseModulator):
    """Differential Phase-Shift Keying (DPSK) modulator.

    Encodes information in the phase differences between consecutive symbols rather than absolute
    phases, making it robust to phase ambiguities.
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer
    _phase_memory: torch.Tensor  # Type annotation for the buffer

    def __init__(self, order: Optional[Literal[2, 4, 8, 16]] = None, gray_coding: bool = True, bits_per_symbol: Optional[int] = None, gray_coded: Optional[bool] = None, *args, **kwargs) -> None:
        """Initialize the DPSK modulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether to use Gray coding for phase mapping
            bits_per_symbol: Alternative way to specify order (2^bits_per_symbol)
            gray_coded: Alternative name for gray_coding
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Pass *args and **kwargs to the base class initializer
        super().__init__(*args, **kwargs)

        # Support both initialization styles (order or bits_per_symbol)
        if bits_per_symbol is not None:
            self._bits_per_symbol = bits_per_symbol
            self.order = 2**bits_per_symbol
        elif order is not None:
            # Validate order is a power of 2
            if not (order > 0 and (order & (order - 1) == 0)):
                raise ValueError(f"DPSK order must be a power of 2, got {order}")
            self.order = order
            self._bits_per_symbol: int = int(np.log2(order))
        else:
            raise ValueError("Either order or bits_per_symbol must be provided")

        # Support both naming conventions
        self.gray_coding = gray_coded if gray_coded is not None else gray_coding

        # Create constellation
        self._create_constellation()

        # Initialize phase memory for differential encoding
        self.register_buffer("_phase_memory", torch.tensor(1.0 + 0.0j))

    def _create_constellation(self) -> None:
        """Create the DPSK constellation mapping."""
        # Generate differential phase shifts
        angles = torch.arange(0, self.order) * (2 * np.pi / self.order)

        # For non-gray-coded, rotate constellation to make it different
        if not self.gray_coding:
            # Add a small rotation to make non-gray constellation visibly different
            angles = angles + np.pi / self.order

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

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bit groups to DPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., K*N), where K is bits_per_symbol,
               or direct symbol indices with shape (..., N) where each value is < order
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Complex tensor of DPSK symbols with shape (..., N)
        """
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]

        # Determine if input contains bit patterns or indices
        is_binary_input = torch.all((x == 0) | (x == 1))

        # If it's a binary input, check length divisibility ONLY if not handling a single element tensor
        # Single element tensors should be treated as indices even if their values are 0 or 1
        if is_binary_input and bit_len > 1 and bit_len % self._bits_per_symbol != 0:
            raise ValueError(f"Input bit length must be divisible by {self._bits_per_symbol}")

        # For single element tensors containing 0s or 1s, treat them as indices not bits
        if is_binary_input and bit_len == 1:
            is_binary_input = False

        if is_binary_input:
            # Calculate number of symbols
            symbol_len = bit_len // self._bits_per_symbol

            # Reshape to groups of bits_per_symbol for processing
            x_reshaped = x.reshape(*batch_shape, symbol_len, self._bits_per_symbol)

            # Convert bit groups to indices
            indices = torch.zeros((*batch_shape, symbol_len), dtype=torch.long, device=x.device)
            for i in range(self._bits_per_symbol):
                indices = indices | (x_reshaped[..., i].long() << (self._bits_per_symbol - i - 1))
        else:
            # Process as direct indices
            indices = x.long()

            # Validate indices are within range
            if torch.any(indices >= self.order):
                raise ValueError(f"Symbol indices must be less than order ({self.order})")

            symbol_len = x.shape[-1]

        # Map indices to differential phase shifts
        phase_shifts = self.constellation[indices]

        # Apply differential encoding
        ref_phase = self._phase_memory.clone().detach()

        # Expand reference phase to match batch dimensions if needed
        if batch_shape:
            # Expand to match batch dimensions
            for _ in range(len(batch_shape)):
                ref_phase = ref_phase.unsqueeze(0)
            ref_phase = ref_phase.expand(*batch_shape, 1)
        else:
            ref_phase = ref_phase.unsqueeze(0)

        # Create output tensor with the right shape
        output = torch.zeros(*batch_shape, symbol_len, dtype=torch.complex64, device=x.device)

        # Apply differential modulation to all symbols
        if symbol_len > 0:
            # First symbol is modulated using the phase memory
            output[..., 0] = ref_phase.squeeze(-1) * phase_shifts[..., 0]

            # Apply differential encoding to subsequent symbols
            for i in range(1, symbol_len):
                output[..., i] = output[..., i - 1] * phase_shifts[..., i]

            # Update phase memory with the last output symbol
            if self.training:
                if batch_shape:
                    self._phase_memory = output[..., -1].detach().mean().view(1)
                else:
                    self._phase_memory = output[..., -1].detach().view(1)

        return output

    def reset_state(self) -> None:
        """Reset the internal phase memory to the default state."""
        self._phase_memory = torch.tensor(1.0 + 0.0j)

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the DPSK constellation diagram.

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

        fig, _ = plot_constellation(self.constellation, labels=labels, title=f"{self.order}-DPSK Constellation", **kwargs)
        return fig


@ModulationRegistry.register_demodulator()
class DPSKDemodulator(BaseDemodulator):
    """Differential Phase-Shift Keying (DPSK) demodulator."""

    def __init__(self, order: Optional[Literal[2, 4, 8, 16]] = None, gray_coding: bool = True, bits_per_symbol: Optional[int] = None, gray_coded: Optional[bool] = None, *args, **kwargs) -> None:
        """Initialize the DPSK demodulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether Gray coding was used for phase mapping
            bits_per_symbol: Alternative way to specify order (2^bits_per_symbol)
            gray_coded: Alternative name for gray_coding
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Pass *args and **kwargs to the base class initializer
        super().__init__(*args, **kwargs)

        # Support both initialization styles (order or bits_per_symbol)
        if bits_per_symbol is not None:
            self._bits_per_symbol = bits_per_symbol
            self.order = 2**bits_per_symbol
        elif order is not None:
            self.order = order
            self._bits_per_symbol: int = int(np.log2(order))
        else:
            raise ValueError("Either order or bits_per_symbol must be provided")

        # Support both naming conventions
        self.gray_coding = gray_coded if gray_coded is not None else gray_coding

        # Create reference modulator to access constellation
        self.modulator = DPSKModulator(self.order, self.gray_coding, *args, **kwargs)

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate DPSK symbols.

        Args:
            y: Received tensor of DPSK symbols with shape (..., N)
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
            with shape (..., (N-1)*bits_per_symbol) because first symbol is reference
        """
        batch_shape = y.shape[:-1]
        symbol_len = y.shape[-1]
        constellation = self.modulator.constellation

        # Need at least two symbols for differential demodulation
        if symbol_len < 2:
            raise ValueError("Need at least two symbols for differential demodulation")

        # Calculate phase differences between consecutive symbols
        y_prev = y[..., :-1]
        y_current = y[..., 1:]

        # z contains the differential phases (normalized by previous symbol)
        z = y_current * torch.conj(y_prev)
        z = z / (torch.abs(z) + 1e-9)  # Normalize to unit magnitude

        if noise_var is None:
            # Hard decision: find closest constellation point
            z_angle = torch.angle(z)
            const_angles = torch.angle(constellation)

            # Find closest angle (considering circular distance)
            expanded_z_angle = z_angle.unsqueeze(-1)  # (..., N-1, 1)
            expanded_const_angle = const_angles.expand(*([1] * len(batch_shape)), symbol_len - 1, self.order)  # (..., N-1, order)

            # Calculate circular distance
            angle_diff = torch.abs((expanded_z_angle - expanded_const_angle + np.pi) % (2 * np.pi) - np.pi)
            closest_indices = torch.argmin(angle_diff, dim=-1)  # (..., N-1)

            # Map to bit patterns using the modulator's bit patterns
            bits = torch.zeros(
                (*batch_shape, symbol_len - 1, self._bits_per_symbol),
                dtype=torch.float,
                device=y.device,
            )

            for i in range(self.order):
                mask = (closest_indices == i).unsqueeze(-1)
                bit_pattern = self.modulator.bit_patterns[i].expand(*batch_shape, symbol_len - 1, self._bits_per_symbol)
                bits = torch.where(mask, bit_pattern, bits)

            return bits.reshape(*batch_shape, -1)
        else:
            # Soft decision
            # For differential demodulation with noise, the effective noise variance is doubled
            # because noise affects both current and previous symbols

            # Convert noise_var to appropriate tensor form and apply 2x factor for differential detection
            if not isinstance(noise_var, torch.Tensor):
                effective_noise_var = torch.tensor(2.0 * noise_var, device=y.device)
            else:
                effective_noise_var = 2.0 * noise_var.to(device=y.device)

            # Handle scalar noise variance
            if effective_noise_var.dim() == 0:  # scalar
                effective_noise_var = effective_noise_var.expand(*batch_shape, symbol_len - 1)

            # Calculate LLRs for each bit position
            llrs = torch.zeros((*batch_shape, symbol_len - 1, self._bits_per_symbol), device=y.device)

            for bit_idx in range(self._bits_per_symbol):
                # Create masks for symbols where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get constellation points for each bit value
                const_bit_0 = constellation[bit_0_mask]
                const_bit_1 = constellation[bit_1_mask]

                # Calculate minimum distance for each bit value
                min_dist_0 = self._min_distance_to_points(z, const_bit_0, effective_noise_var)
                min_dist_1 = self._min_distance_to_points(z, const_bit_1, effective_noise_var)

                # Calculate LLR: log(P(bit=0)/P(bit=1))
                llrs[..., bit_idx] = min_dist_1 - min_dist_0

            return llrs.reshape(*batch_shape, -1)

    def _min_distance_to_points(self, y: torch.Tensor, points: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
        """Calculate minimum (negative) distance to constellation points for DPSK.

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

        # Fix: Ensure points_expanded has the right dimensions for all tensor shapes
        # Reshape inputs for broadcasting
        y_expanded = y.unsqueeze(-1)
        if batch_shape:
            # For multi-dimensional tensors, use proper expand
            y_expanded = y_expanded.expand(*batch_shape, symbol_shape, num_points)

            # Create points_expanded to match dimensions
            points_reshaped = points.reshape(*([1] * len(batch_shape)), 1, -1)
            points_expanded = points_reshaped.expand(*batch_shape, symbol_shape, num_points)

            # Expand noise variance similarly
            noise_var_expanded = noise_var.unsqueeze(-1).expand(*batch_shape, symbol_shape, num_points)
        else:
            # For 1D tensors, simpler expansion
            y_expanded = y_expanded.expand(symbol_shape, num_points)
            points_expanded = points.reshape(1, -1).expand(symbol_shape, num_points)
            noise_var_expanded = noise_var.unsqueeze(-1).expand(symbol_shape, num_points)

        # Calculate distances (using phase difference for DPSK)
        distances = -torch.abs(y_expanded - points_expanded) ** 2 / noise_var_expanded

        # Return maximum (least negative) value
        return torch.max(distances, dim=-1)[0]


@ModulationRegistry.register_modulator("dbpsk")
class DBPSKModulator(DPSKModulator):
    """Differential Binary Phase-Shift Keying (DBPSK) modulator."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize DBPSK Modulator."""
        # Filter out conflicting keys to avoid duplicate argument errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("order", "gray_coding")}
        # Pass remaining args and filtered kwargs
        super().__init__(2, False, *args, **filtered_kwargs)


@ModulationRegistry.register_demodulator("dbpsk")
class DBPSKDemodulator(DPSKDemodulator):
    """Differential Binary Phase-Shift Keying (DBPSK) demodulator."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize DBPSK Demodulator."""
        # Filter out conflicting keys to avoid duplicate argument errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("order", "gray_coding")}
        # Pass remaining args and filtered kwargs
        super().__init__(2, False, *args, **filtered_kwargs)


@ModulationRegistry.register_modulator("dqpsk")
class DQPSKModulator(DPSKModulator):
    """Differential Quadrature Phase-Shift Keying (DQPSK) modulator."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize DQPSK Modulator."""
        # Filter out conflicting keys to avoid duplicate argument errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("order", "gray_coding")}
        # Pass remaining args and filtered kwargs
        super().__init__(4, True, *args, **filtered_kwargs)


@ModulationRegistry.register_demodulator("dqpsk")
class DQPSKDemodulator(DPSKDemodulator):
    """Differential Quadrature Phase-Shift Keying (DQPSK) demodulator."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize DQPSK Demodulator."""
        # Filter out conflicting keys to avoid duplicate argument errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ("order", "gray_coding")}
        # Pass remaining args and filtered kwargs
        super().__init__(4, True, *args, **filtered_kwargs)
