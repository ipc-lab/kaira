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

    Maps binary inputs (0, 1) to constellation points (1, -1).
    Following standard convention where:
    - Bit 0 maps to +1
    - Bit 1 maps to -1
    """

    constellation: torch.Tensor  # Type annotation for the buffer

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BPSK modulator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        # Define constellation points
        re_part = torch.tensor([1.0, -1.0])
        im_part = torch.tensor([0.0, 0.0])
        self.register_buffer("constellation", torch.complex(re_part, im_part))
        self._bits_per_symbol = 1  # BPSK has 1 bit per symbol

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate binary inputs to BPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., N)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Complex tensor of BPSK symbols with shape (..., N)
        """
        # Convert binary 0/1 to 1/-1
        return torch.complex(1.0 - 2.0 * x.float(), torch.zeros_like(x.float()))

    def plot_constellation(self, **kwargs) -> plt.Figure:
        """Plot the BPSK constellation diagram.

        Args:
            **kwargs: Additional arguments passed to plot_constellation

        Returns:
            Matplotlib figure object
        """
        return plot_constellation(self.constellation, labels=["0", "1"], title="BPSK Constellation", **kwargs)


@ModulationRegistry.register_demodulator()
class BPSKDemodulator(BaseDemodulator):
    """Binary Phase-Shift Keying (BPSK) demodulator.

    Following standard convention where:
    - Positive values map to bit 0
    - Negative values map to bit 1
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BPSK demodulator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._bits_per_symbol = 1  # BPSK has 1 bit per symbol

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate BPSK symbols.

        Args:
            y: Received tensor of BPSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # Extract real part for decision (BPSK uses only real axis)
        y_real = y.real

        if noise_var is None:
            # Hard decision: y < 0 -> 1, y >= 0 -> 0
            return (y_real < 0).float()
        else:
            # Support both scalar and tensor noise variance
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Soft decision: LLR calculation
            # Negative LLR means bit 1 is more likely, positive means bit 0 is more likely
            # LLR = log(P(y|b=0)/P(y|b=1)) = log(exp(-(y-1)²/2σ²)/exp(-(y+1)²/2σ²)) = 2y/σ²
            return -2.0 * y_real / noise_var


@ModulationRegistry.register_modulator()
class QPSKModulator(BaseModulator):
    """Quadrature Phase-Shift Keying (QPSK) modulator.

    Maps pairs of bits to complex constellation points in QPSK modulation.
    Following standard Gray-coded QPSK convention where:
    - 00 maps to (1+j)/√2  (first quadrant)
    - 01 maps to (1-j)/√2  (fourth quadrant)
    - 10 maps to (-1+j)/√2 (second quadrant)
    - 11 maps to (-1-j)/√2 (third quadrant)
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer

    def __init__(self, normalize: bool = True, *args, **kwargs) -> None:
        """Initialize the QPSK modulator.

        Args:
            normalize: If True, normalize constellation to unit energy
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # QPSK mapping table with Gray coding
        re_part = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float) * self._normalization
        im_part = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float) * self._normalization
        self.register_buffer("constellation", torch.complex(re_part, im_part))

        # Bit patterns for each symbol - Gray coded
        bit_patterns = torch.tensor(
            [
                [0.0, 0.0],  # First quadrant
                [0.0, 1.0],  # Fourth quadrant
                [1.0, 0.0],  # Second quadrant
                [1.0, 1.0],  # Third quadrant
            ],
            dtype=torch.float,
        )
        self.register_buffer("bit_patterns", bit_patterns)

        self._bits_per_symbol = 2  # QPSK has 2 bits per symbol

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bit pairs to QPSK symbols.

        Args:
            x: Input tensor of bits with shape (..., 2*N)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

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

        # Convert bit pairs to indices using Gray coding pattern
        indices = x_reshaped[..., 0].to(torch.long) * 2 + x_reshaped[..., 1].to(torch.long)

        # Handle empty tensor case
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


@ModulationRegistry.register_demodulator()
class QPSKDemodulator(BaseDemodulator):
    """Quadrature Phase-Shift Keying (QPSK) demodulator.

    Demodulates QPSK symbols back to bit pairs following Gray coding convention.
    """

    def __init__(self, normalize: bool = True, *args, **kwargs) -> None:
        """Initialize the QPSK demodulator.

        Args:
            normalize: If True, assume normalized constellation with unit energy
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self._normalization = 1 / np.sqrt(2) if normalize else 1.0

        # Create modulator to access constellation
        self.modulator = QPSKModulator(normalize)

        self._bits_per_symbol = 2  # QPSK has 2 bits per symbol

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate QPSK symbols.

        Args:
            y: Received tensor of QPSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]

        if noise_var is None:
            # Hard decision: Find the closest constellation point
            expanded_y = y.unsqueeze(-1)  # (..., N, 1)
            expanded_const = self.modulator.constellation.expand(*([1] * len(batch_shape)), symbol_shape, 4)  # (..., N, 4)

            # Calculate distances
            distances = torch.abs(expanded_y - expanded_const)
            closest_idx = torch.argmin(distances, dim=-1)  # (..., N)

            # Get bit patterns for each closest constellation point
            bits = torch.zeros((*batch_shape, symbol_shape, 2), dtype=torch.float, device=y.device)
            for i in range(4):
                mask = (closest_idx == i).unsqueeze(-1).expand(*batch_shape, symbol_shape, 2)
                bit_pattern = self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, 2)
                bits = torch.where(mask, bit_pattern, bits)

            # Reshape to final bit sequence
            return bits.reshape(*batch_shape, -1)
        else:
            # Support both scalar and tensor noise variance
            if not isinstance(noise_var, torch.Tensor):
                noise_var = torch.tensor(noise_var, device=y.device)

            # Handle broadcasting dimensions for noise_var
            if noise_var.dim() == 0:  # scalar
                noise_var = noise_var.expand(*batch_shape, symbol_shape)

            # Calculate LLRs for bit positions
            llrs = torch.zeros((*batch_shape, symbol_shape, 2), device=y.device)

            # For each bit position, compute the LLR using max-log approximation
            for bit_idx in range(2):
                # Separate constellation points for bit=0 and bit=1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get corresponding constellation points
                const_bit_0 = self.modulator.constellation[bit_0_mask]
                const_bit_1 = self.modulator.constellation[bit_1_mask]

                # Calculate minimum distances
                min_dist_0 = self._min_distance_to_points(y, const_bit_0, noise_var)
                min_dist_1 = self._min_distance_to_points(y, const_bit_1, noise_var)

                # LLR = log(P(bit=0|y)/P(bit=1|y))
                llrs[..., bit_idx] = min_dist_1 - min_dist_0

            # Reshape to final sequence
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
        y.unsqueeze(-1)  # (..., N, 1)

        # Fix the dimension mismatch by directly calculating distances for each point
        distances = torch.zeros((*batch_shape, symbol_shape, num_points), device=y.device)

        for i in range(num_points):
            point = points[i]
            # Calculate squared distance between each symbol and this point
            distances[..., i] = -torch.abs(y - point) ** 2 / noise_var

        # Return maximum (least negative) value for each symbol
        max_values, _ = torch.max(distances, dim=-1)
        return max_values


@ModulationRegistry.register_modulator()
class PSKModulator(BaseModulator):
    """General M-ary Phase-Shift Keying (PSK) modulator.

    Maps groups of bits to complex constellation points around the unit circle. Follows standard
    digital communications convention with Gray coding.
    """

    constellation: torch.Tensor  # Type annotation for the buffer
    bit_patterns: torch.Tensor  # Type annotation for the buffer
    bit_to_symbol_map: torch.Tensor  # Type annotation for mapping bit patterns to symbols

    def __init__(self, order: Literal[4, 8, 16, 32, 64] = 4, gray_coding: bool = True, constellation: Optional[torch.Tensor] = None, *args, **kwargs) -> None:
        """Initialize the PSK modulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether to use Gray coding for constellation mapping
            constellation: Optional custom constellation points (overrides order)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.gray_coding = gray_coding

        if constellation is not None:
            # Use custom constellation
            self.register_buffer("constellation", constellation)
            self.order = len(constellation)
            # Validate order is a power of 2
            if not (self.order > 0 and (self.order & (self.order - 1) == 0)):
                raise ValueError(f"Custom constellation length must be a power of 2, got {self.order}")
            self._bits_per_symbol: int = int(np.log2(self.order))
        else:
            # Validate order is a power of 2
            if not (order > 0 and (order & (order - 1) == 0)):
                raise ValueError(f"PSK order must be a power of 2, got {order}")

            self.order = order
            self._bits_per_symbol = int(np.log2(order))

            # Create standard PSK constellation
            self._create_constellation()

    def _create_constellation(self) -> None:
        """Create the PSK constellation mapping."""
        # Generate points evenly spaced around the unit circle
        # Standard convention: first point at angle 0 (real axis)
        angles = torch.arange(0, self.order) * (2 * np.pi / self.order)
        re_part = torch.cos(angles)
        im_part = torch.sin(angles)
        constellation = torch.complex(re_part, im_part)

        # Create bit pattern mapping
        bit_patterns = torch.zeros(self.order, self._bits_per_symbol)

        if self.gray_coding:
            # Apply Gray coding - standard digital communications convention
            # For each index i, calculate corresponding Gray code
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

        # Create mapping from bit patterns to constellation indices
        bit_to_symbol_map = torch.zeros(self.order, dtype=torch.long)

        # Map each bit pattern to its index in the constellation
        for i in range(self.order):
            # Create binary index from bit pattern
            idx = 0
            for j in range(self._bits_per_symbol):
                idx = idx * 2 + int(bit_patterns[i, j])

            if self.gray_coding:
                # For Gray coding, we map the bit pattern to the constellation point
                bit_to_symbol_map[idx] = i
            else:
                # For binary coding, the mapping is direct
                bit_to_symbol_map[i] = i

        self.register_buffer("constellation", constellation)
        self.register_buffer("bit_patterns", bit_patterns)
        self.register_buffer("bit_to_symbol_map", bit_to_symbol_map)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Modulate bit groups to PSK symbols.

        Args:
            x: Input tensor of bits with shape (..., M) or direct indices into the constellation
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Complex tensor of PSK symbols with shape (..., N)
        """
        # Handle scalar and 0-dim tensor inputs
        scalar_input = x.dim() == 0
        if scalar_input:
            x = x.unsqueeze(0)

        # Special case for direct constellation indices as scalar or single-element tensor
        if x.numel() == 1 and ((x == x.long()).all() and torch.all(x < self.order) and torch.all(x >= 0)):
            # This is a direct index into the constellation
            return self.constellation[x.long()].squeeze()  # Return scalar output for scalar input

        # Normal case: Binary bit values grouped into symbols
        # Ensure input contains binary values (0s and 1s)
        if torch.any((x != 0) & (x != 1)):
            # If there are non-binary values, check if they are valid indices
            if not ((x == x.long()).all() and torch.all(x < self.order) and torch.all(x >= 0)):
                raise ValueError("Input tensor must contain only binary values (0s and 1s)")

            # Special case for direct indices in a tensor
            if x.dim() == 1:
                # These are valid indices
                indices = x.long()
                return self.constellation[indices]

        # Get batch shape and bit length
        batch_shape = x.shape[:-1]
        bit_len = x.shape[-1]

        # Ensure input length is a multiple of bits_per_symbol
        if bit_len % self._bits_per_symbol != 0:
            raise ValueError(f"Input bit length must be a multiple of {self._bits_per_symbol} for {self.order}-PSK modulation")

        # Reshape to groups of bits
        x_reshaped = x.reshape(*batch_shape, -1, self._bits_per_symbol)

        # Convert bit groups to indices
        indices = torch.zeros((*batch_shape, x_reshaped.shape[-2]), dtype=torch.long, device=x.device)
        for i in range(self._bits_per_symbol):
            power = 2 ** (self._bits_per_symbol - 1 - i)
            indices = indices + (x_reshaped[..., i].long() * power)

        # Map bit pattern indices to constellation indices
        symbol_indices = self.bit_to_symbol_map[indices]

        # Map indices to symbols
        symbols = self.constellation[symbol_indices]

        # Handle scalar output if input was scalar
        if scalar_input and bit_len == self._bits_per_symbol:
            symbols = symbols.squeeze()

        return symbols

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


@ModulationRegistry.register_demodulator()
class PSKDemodulator(BaseDemodulator):
    """General M-ary Phase-Shift Keying (PSK) demodulator.

    Demodulates complex constellation points back to bits.
    """

    def __init__(self, order: Literal[4, 8, 16, 32, 64] = 4, gray_coding: bool = True, *args, **kwargs) -> None:
        """Initialize the PSK demodulator.

        Args:
            order: Modulation order (must be a power of 2)
            gray_coding: Whether Gray coding was used for constellation mapping
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.order = order
        self.gray_coding = gray_coding
        self._bits_per_symbol: int = int(np.log2(order))

        # Create modulator to access constellation
        self.modulator = PSKModulator(order, gray_coding)

    def forward(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None, *args, **kwargs) -> torch.Tensor:
        """Demodulate PSK symbols.

        Args:
            y: Received tensor of PSK symbols
            noise_var: Noise variance for soft demodulation (optional)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            If noise_var is provided, returns LLRs; otherwise, returns hard bit decisions
        """
        # Handle scalar input
        scalar_input = y.dim() == 0
        if scalar_input:
            y = y.unsqueeze(0)

        batch_shape = y.shape[:-1]
        symbol_shape = y.shape[-1]
        constellation = self.modulator.constellation

        if noise_var is None:
            # Hard decision: find closest constellation point
            expanded_y = y.unsqueeze(-1)  # (..., N, 1)
            expanded_const = constellation.unsqueeze(0).expand(*([1] * len(batch_shape)), symbol_shape, self.order)

            # Calculate distances to constellation points
            distances = torch.abs(expanded_y - expanded_const)
            closest_indices = torch.argmin(distances, dim=-1)  # (..., N)

            # Map to bit patterns
            bits = torch.zeros((*batch_shape, symbol_shape, self._bits_per_symbol), dtype=torch.float, device=y.device)
            for i in range(self.order):
                mask = (closest_indices == i).unsqueeze(-1).expand(*batch_shape, symbol_shape, self._bits_per_symbol)
                bit_pattern = self.modulator.bit_patterns[i].expand(*batch_shape, symbol_shape, self._bits_per_symbol)
                bits = torch.where(mask, bit_pattern, bits)

            # Reshape to final bit sequence
            result = bits.reshape(*batch_shape, -1).float()  # Ensure consistent float output

            # Handle scalar output if input was scalar
            if scalar_input:
                result = result.squeeze(0)

            return result
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
                # Get constellation points where bit is 0 or 1
                bit_0_mask = self.modulator.bit_patterns[:, bit_idx] == 0
                bit_1_mask = ~bit_0_mask

                # Get corresponding points
                const_bit_0 = constellation[bit_0_mask]
                const_bit_1 = constellation[bit_1_mask]

                # Process each symbol individually for clearer computation
                for b_idx in np.ndindex(batch_shape):
                    for s_idx in range(symbol_shape):
                        # Get the received symbol
                        if batch_shape:
                            sym = y[b_idx][s_idx]
                            nvar = noise_var[b_idx][s_idx]
                        else:
                            sym = y[s_idx]
                            nvar = noise_var[s_idx]

                        # Calculate distances to constellation points
                        dist_0 = torch.min(torch.abs(sym - const_bit_0) ** 2) / nvar
                        dist_1 = torch.min(torch.abs(sym - const_bit_1) ** 2) / nvar

                        # LLR = log(P(bit=0|y)/P(bit=1|y)) = log(exp(-dist_0)/exp(-dist_1)) = -dist_0 + dist_1
                        if batch_shape:
                            llrs[b_idx][s_idx][bit_idx] = -dist_0 + dist_1
                        else:
                            llrs[s_idx][bit_idx] = -dist_0 + dist_1

            # Reshape to final LLR sequence
            result = llrs.reshape(*batch_shape, -1)

            # Handle scalar output if input was scalar
            if scalar_input:
                result = result.squeeze(0)

            return result
