"""Repetition coding module for forward error correction.

This module implements repetition coding for binary data transmission, a simple error correction
technique where each bit is repeated multiple times. For decoding, a majority vote scheme is used
to determine the most likely transmitted bit value :cite:`lin2004error,moon2005error`.

Repetition coding provides a straightforward way to improve reliability at the expense of rate,
making it suitable for educational purposes and systems with strict reliability requirements but
modest rate requirements.

As a special case of linear block codes, repetition codes have a generator matrix consisting
of a single row of all ones, and a check matrix with (n-1) rows forming a basis for the
orthogonal complement of the all-ones vector :cite:`richardson2008modern`.
"""

from typing import Any

import torch
from scipy.special import comb

from kaira.models.registry import ModelRegistry

from .linear_block_code import LinearBlockCodeEncoder


@ModelRegistry.register_model("repetition_encoder")
class RepetitionCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for repetition coding that extends LinearBlockCodeEncoder.

    This encoder implements a repetition code, which is a special case of linear
    block codes where each bit is repeated n times. A repetition code has the
    following properties:

    - Length: n (the repetition factor)
    - Dimension: k = 1 (one information bit produces n coded bits)
    - Redundancy: r = n - 1 (number of redundant bits)
    - Minimum distance: d = n (can correct up to ⌊(n-1)/2⌋ errors)

    Its dual is the single parity-check code. The generator matrix is a
    single row of all ones [1, 1, ..., 1].

    Attributes:
        repetition_factor (int): The length n of the code. Must be a positive integer.

    Args:
        repetition_factor (int): Number of times to repeat each bit

    Examples:
        >>> import torch
        >>> encoder = RepetitionCodeEncoder(repetition_factor=5)
        >>> encoder.code_length, encoder.code_dimension, encoder.redundancy
        (5, 1, 4)
        >>> encoder.generator_matrix
        tensor([[1., 1., 1., 1., 1.]])
        >>> encoder(torch.tensor([[1.]]))
        tensor([[1., 1., 1., 1., 1.]])
    """

    def __init__(self, repetition_factor: int = 3, **kwargs: Any):
        """Initialize the repetition encoder.

        Args:
            repetition_factor: Number of times to repeat each bit. Must be a positive integer.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If repetition_factor is less than 1.
        """
        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")

        # Create the generator matrix for a repetition code: [1, 1, ..., 1]
        generator_matrix = torch.ones((1, repetition_factor), dtype=torch.float32)

        # Remove generator_matrix from kwargs if it exists to avoid duplicate
        kwargs_copy = kwargs.copy()
        if "generator_matrix" in kwargs_copy:
            del kwargs_copy["generator_matrix"]

        # Initialize the LinearBlockCodeEncoder parent with this generator matrix
        super().__init__(generator_matrix=generator_matrix, **kwargs_copy)

        # Store repetition factor as an attribute
        self.repetition_factor = repetition_factor

    def coset_leader_weight_distribution(self) -> torch.Tensor:
        """Calculate the coset leader weight distribution of the repetition code.

        For a repetition code of length n, the coset leader weight distribution
        is given by the binomial coefficients C(n,w) for w from 0 to ⌊n/2⌋,
        with a special case for n/2 when n is even.

        Returns:
            Tensor containing the coset leader weight distribution
        """
        n = self.repetition_factor
        distribution = torch.zeros(n + 1, dtype=torch.int64)

        # Fill in the distribution values
        for w in range((n + 1) // 2):
            distribution[w] = int(comb(n, w))

        # Special case when n is even
        if n % 2 == 0:
            distribution[n // 2] = int(comb(n, n // 2) // 2)

        return distribution

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            String representation with key parameters
        """
        return f"{self.__class__.__name__}(repetition_factor={self.repetition_factor})"
