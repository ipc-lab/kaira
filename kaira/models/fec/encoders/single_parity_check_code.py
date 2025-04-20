"""Single parity-check coding module for forward error correction.

This module implements single parity-check codes for binary data transmission.
A single parity-check code extends (n-1) information bits with a single parity bit,
creating a code of length n with dimension k = n-1.

The parity bit is chosen to ensure that the total number of 1s in the codeword is even,
providing the ability to detect (but not correct) a single error. Single parity-check
codes are among the simplest error detection codes, offering modest protection with
minimal redundancy.

For more theoretical background on single parity-check codes, see :cite:`lin2004error,moon2005error`.

This implementation follows standard conventions in coding theory for binary linear block codes,
with elements belonging to the binary field GF(2) :cite:`richardson2008modern`.
"""

from typing import Any

import torch

from kaira.models.registry import ModelRegistry

from .linear_block_code import LinearBlockCodeEncoder


def _generate_single_parity_check_matrix(dimension: int) -> torch.Tensor:
    """Generate the generator matrix for a single parity-check code.

    Args:
        dimension: The dimension k of the code (number of information bits)

    Returns:
        Generator matrix G for single parity-check code of shape (k, k+1)
    """
    # Create a k×k identity matrix for the systematic part
    identity = torch.eye(dimension, dtype=torch.int64)

    # Create a column of ones for the parity part
    parity_column = torch.ones((dimension, 1), dtype=torch.int64)

    # Concatenate the identity matrix and the parity column
    gen_matrix = torch.cat([identity, parity_column], dim=1)

    return gen_matrix


@ModelRegistry.register_model("single_parity_check_code_encoder")
class SingleParityCheckCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for single parity-check codes.

    A single parity-check code extends a message of k bits with a single parity bit,
    creating a codeword of length n = k + 1. The parity bit is chosen so that
    the total number of 1s in the codeword is even.

    The resulting code has the following properties:
    - Length: n = k + 1
    - Dimension: k
    - Minimum distance: d = 2

    This code can detect a single error but cannot correct any errors.
    Its dual is the repetition code.

    Attributes:
        dimension (int): The dimension k of the code (number of information bits)
        minimum_distance (int): The minimum Hamming distance of the code (always 2)

    Args:
        dimension (int): The dimension k of the code (number of information bits)
        *args: Variable positional arguments passed to the base class.
        **kwargs: Variable keyword arguments passed to the base class.
    """

    def __init__(self, dimension: int, *args: Any, **kwargs: Any):
        """Initialize the single parity-check encoder.

        Args:
            dimension: The dimension k of the code (number of information bits)
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If the dimension is not a positive integer
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be a positive integer, got {dimension}")

        # Generate the generator matrix for the code
        generator_matrix = _generate_single_parity_check_matrix(dimension)

        # Initialize the base class with the generator matrix
        super().__init__(generator_matrix, *args, **kwargs)

        # Store single parity-check specific parameters
        self.dimension = dimension
        self.minimum_distance = 2

    @classmethod
    def from_parameters(cls, dimension: int, *args: Any, **kwargs: Any) -> "SingleParityCheckCodeEncoder":
        """Create a single parity-check encoder from parameters.

        This is an alternative constructor that creates the encoder directly from
        the single parity-check parameters.

        Args:
            dimension: The dimension k of the code (number of information bits)
            *args: Variable positional arguments passed to the constructor.
            **kwargs: Variable keyword arguments passed to the constructor.

        Returns:
            A SingleParityCheckCodeEncoder instance
        """
        return cls(dimension, *args, **kwargs)

    @classmethod
    def from_length(cls, length: int, *args: Any, **kwargs: Any) -> "SingleParityCheckCodeEncoder":
        """Create a single parity-check encoder from the code length.

        This is an alternative constructor that creates the encoder given the
        desired code length n. The dimension k will be n-1.

        Args:
            length: The length n of the code
            *args: Variable positional arguments passed to the constructor.
            **kwargs: Variable keyword arguments passed to the constructor.

        Returns:
            A SingleParityCheckCodeEncoder instance

        Raises:
            ValueError: If the length is less than 2
        """
        if length < 2:
            raise ValueError(f"Code length must be at least 2, got {length}")

        # Dimension = length - 1
        dimension = length - 1

        return cls(dimension, *args, **kwargs)
