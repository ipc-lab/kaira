"""Golay code implementation for forward error correction.

This module implements binary Golay codes, a family of perfect linear error-correcting codes
with remarkable properties. The binary Golay code has the following parameters:

- Length: n = 23
- Dimension: k = 12
- Redundancy: r = 11
- Minimum distance: d = 7

In its extended version, the Golay code has the following parameters:

- Length: n = 24
- Dimension: k = 12
- Redundancy: r = 12
- Minimum distance: d = 8

Golay codes are perfect codes that can correct up to 3 errors and detect up to 7 errors
:cite:`lin2004error,moon2005error,richardson2008modern,golay1949notes`.
"""

from functools import lru_cache
from typing import Any, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder


def create_golay_parity_submatrix(extended: bool = False, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create the parity submatrix for a binary Golay code.

    The binary Golay code is defined by a specific parity submatrix which provides its
    unique error correction properties.

    Args:
        extended: Whether to create an extended Golay code. Default is False.
        dtype: The data type for tensor elements. Default is torch.float32.
        device: The device to place the resulting tensor on. Default is None (uses current device).

    Returns:
        The parity submatrix of the Golay code.
    """
    # Define the standard Golay code parity submatrix (12×11)
    parity_submatrix = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )

    # For extended Golay code, add an overall parity check column
    if extended:
        # Calculate the additional parity column (sum of rows mod 2)
        row_sums = torch.sum(parity_submatrix, dim=1) % 2
        # Add 1 to each sum per the definition
        last_column = (1 + row_sums) % 2
        # Append the last column
        extended_column = last_column.view(-1, 1)
        parity_submatrix = torch.cat([parity_submatrix, extended_column], dim=1)

    return parity_submatrix


# Generator polynomial for cyclic representation of Golay code
GOLAY_GENERATOR_POLYNOMIAL = 0b101011100011  # The value 0x2BB in hexadecimal (2787 in decimal)


@ModelRegistry.register_model("golay_code_encoder")
class GolayCodeEncoder(SystematicLinearBlockCodeEncoder):
    r"""Encoder for binary Golay codes.

    The binary Golay code is a perfect [23,12,7] linear error-correcting code that can
    correct up to 3 errors in a 23-bit word. The extended Golay code is a [24,12,8]
    code that can also correct up to 3 errors in a 24-bit word and detect up to 4 errors.

    These codes are named after Marcel J. E. Golay who discovered them in 1949
    :cite:`golay1949notes`. The binary Golay code is one of the few known perfect codes
    :cite:`lin2004error,moon2005error`.

    The parity submatrix for the binary Golay code is:

    .. math::

        P = \begin{bmatrix}
            1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 \\
            0 & 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
            0 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\
            1 & 0 & 1 & 1 & 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
            1 & 1 & 0 & 1 & 1 & 0 & 1 & 0 & 1 & 0 & 1 \\
            1 & 1 & 1 & 0 & 1 & 1 & 0 & 1 & 0 & 0 & 1 \\
            0 & 0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 0 & 1 \\
            0 & 1 & 0 & 1 & 0 & 1 & 1 & 1 & 0 & 0 & 1 \\
            0 & 1 & 1 & 0 & 1 & 0 & 1 & 0 & 0 & 1 & 1 \\
            1 & 0 & 0 & 1 & 1 & 0 & 0 & 1 & 0 & 1 & 1 \\
            1 & 0 & 1 & 0 & 0 & 0 & 1 & 1 & 1 & 0 & 1 \\
            1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 1 & 1 & 1
        \end{bmatrix}

    The extended Golay code adds an additional parity check column to ensure the overall
    parity of each codeword is even.

    Args:
        extended (bool, optional): Whether to use the extended version of the Golay code.
            Default is False.
        information_set (Union[List[int], torch.Tensor, str], optional): Information set
            specification. Default is "left".
        dtype (torch.dtype, optional): Data type for internal tensors. Default is torch.float32.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> encoder = GolayCodeEncoder()
        >>> print(f"Length: {encoder.length}, Dimension: {encoder.dimension}, Redundancy: {encoder.redundancy}")
        Length: 23, Dimension: 12, Redundancy: 11
        >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1.])
        >>> codeword = encoder(message)
        >>> print(codeword)
        tensor([1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1.])

        >>> # Using the extended version
        >>> ext_encoder = GolayCodeEncoder(extended=True)
        >>> print(f"Length: {ext_encoder.length}, Dimension: {ext_encoder.dimension}, Redundancy: {ext_encoder.redundancy}")
        Length: 24, Dimension: 12, Redundancy: 12
        >>> message = torch.tensor([1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1.])
        >>> codeword = ext_encoder(message)
        >>> print(codeword)
        tensor([1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.])
    """

    def __init__(self, extended: bool = False, information_set: Union[List[int], torch.Tensor, str] = "left", dtype: torch.dtype = torch.float32, **kwargs: Any):
        """Initialize the Golay code encoder.

        Args:
            extended (bool, optional): Whether to use the extended version of the Golay code.
                Default is False.
            information_set (Union[List[int], torch.Tensor, str], optional): Either indices of information positions,
                which must be a k-sublist of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
            dtype (torch.dtype, optional): Data type for internal tensors. Default is torch.float32.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Store parameters
        self._extended = extended
        self._dtype = dtype

        # Get device from kwargs if provided
        device = kwargs.get("device", None)

        # Create parity submatrix for Golay code
        parity_submatrix = create_golay_parity_submatrix(extended=extended, dtype=dtype, device=device)

        # Initialize the parent class with this parity submatrix
        super().__init__(parity_submatrix=parity_submatrix, information_set=information_set, **kwargs)

        # Calculate theoretical parameters
        self._theoretical_length = 24 if extended else 23
        self._theoretical_dimension = 12
        self._theoretical_redundancy = 12 if extended else 11
        self._error_correction_capability = 3  # Both Golay codes can correct up to 3 errors

        # Validate that the calculated dimensions match the theoretical ones
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate that the code dimensions match the theoretical values.

        Raises:
            ValueError: If calculated dimensions don't match theoretical expectations.
        """
        if self._length != self._theoretical_length:
            raise ValueError(f"Code length mismatch: calculated {self._length}, " f"expected {self._theoretical_length}")
        if self._dimension != self._theoretical_dimension:
            raise ValueError(f"Code dimension mismatch: calculated {self._dimension}, " f"expected {self._theoretical_dimension}")
        if self._redundancy != self._theoretical_redundancy:
            raise ValueError(f"Code redundancy mismatch: calculated {self._redundancy}, " f"expected {self._theoretical_redundancy}")

    @property
    def extended(self) -> bool:
        """Whether this is an extended Golay code."""
        return self._extended

    @property
    def error_correction_capability(self) -> int:
        """Number of errors the code can correct (3)."""
        return self._error_correction_capability

    @lru_cache(maxsize=None)
    def minimum_distance(self) -> int:
        """Calculate the minimum Hamming distance of the code.

        Returns:
            int: The minimum Hamming distance:
                - 7 for standard Golay code
                - 8 for extended Golay code
        """
        return 8 if self._extended else 7

    @classmethod
    def create_extended_golay_code(cls, **kwargs: Any) -> "GolayCodeEncoder":
        """Create an extended Golay code encoder.

        This is a convenience method for creating the extended version of the code.

        Args:
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            GolayCodeEncoder: Extended Golay code encoder.
        """
        kwargs["extended"] = True
        return cls(**kwargs)

    @classmethod
    def create_standard_golay_code(cls, **kwargs: Any) -> "GolayCodeEncoder":
        """Create a standard Golay code encoder.

        This is a convenience method for creating the standard version of the code.

        Args:
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            GolayCodeEncoder: Standard Golay code encoder.
        """
        kwargs["extended"] = False
        return cls(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the encoder.

        Returns:
            str: A string representation with key parameters.
        """
        return f"{self.__class__.__name__}(" f"extended={self._extended}, " f"length={self._length}, " f"dimension={self._dimension}, " f"redundancy={self._redundancy}, " f"error_correction_capability={self._error_correction_capability}, " f"dtype={self._dtype.__repr__()}" f")"
