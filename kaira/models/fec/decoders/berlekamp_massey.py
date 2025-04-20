"""Berlekamp-Massey decoder for BCH and Reed-Solomon codes.

This module implements the Berlekamp-Massey algorithm for decoding BCH and Reed-Solomon codes. The
algorithm efficiently solves the key equation for the error locator polynomial, which is then used
to find the locations of errors in the received codeword.
"""

from typing import Any, List, Tuple, Union

import torch

from kaira.models.fec.algebra import BinaryPolynomial
from kaira.models.fec.encoders.bch_code import BCHCodeEncoder
from kaira.models.fec.encoders.reed_solomon_code import ReedSolomonCodeEncoder

from ..utils import apply_blockwise
from .base import BlockDecoder


class BerlekampMasseyDecoder(BlockDecoder[Union[BCHCodeEncoder, ReedSolomonCodeEncoder]]):
    """Berlekamp-Massey decoder for BCH and Reed-Solomon codes.

    This decoder implements the Berlekamp-Massey algorithm for decoding BCH and Reed-Solomon codes.
    It is particularly efficient for these algebraic codes and can correct up to t = ⌊(d-1)/2⌋ errors,
    where d is the minimum distance of the code.

    The decoder works by:
    1. Computing the syndrome polynomial from the received word
    2. Using the Berlekamp-Massey algorithm to find the error locator polynomial
    3. Finding the roots of the error locator polynomial to determine error locations
    4. Correcting the errors in the received word
    5. Extracting the message bits from the corrected codeword

    Attributes:
        encoder (Union[BCHCodeEncoder, ReedSolomonCodeEncoder]): The encoder instance
        field (GaloisField): The finite field used by the code
        t (int): Error-correcting capability of the code

    Args:
        encoder (Union[BCHCodeEncoder, ReedSolomonCodeEncoder]): The encoder for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class
    """

    def __init__(self, encoder: Union[BCHCodeEncoder, ReedSolomonCodeEncoder], *args: Any, **kwargs: Any):
        """Initialize the Berlekamp decoder."""
        super().__init__(encoder, *args, **kwargs)

        if not isinstance(encoder, (BCHCodeEncoder, ReedSolomonCodeEncoder)):
            raise TypeError(f"Encoder must be a BCHCodeEncoder or ReedSolomonCodeEncoder, got {type(encoder).__name__}")

        self.field = encoder.field
        self.t = encoder.error_correction_capability

    def berlekamp_massey_algorithm(self, syndrome: List[Any]) -> List[Any]:
        """Implement the Berlekamp-Massey algorithm.

        This algorithm finds the error locator polynomial from the syndrome values.

        Args:
            syndrome: List of syndrome values in the Galois field

        Returns:
            Coefficients of the error locator polynomial
        """
        # Initialize variables
        field = self.field
        sigma = {-1: [field.one], 0: [field.one]}
        discrepancy = {-1: field.one, 0: syndrome[0]}
        degree = {-1: 0, 0: 0}

        # Main algorithm loop
        for j in range(self.t * 2 - 1):
            if discrepancy[j] == field.zero:
                degree[j + 1] = degree[j]
                sigma[j + 1] = sigma[j]
            else:
                # Find the most suitable previous iteration
                k, max_so_far = -1, -1
                for i in range(-1, j):
                    if discrepancy[i] != field.zero and i - degree[i] > max_so_far:
                        k, max_so_far = i, i - degree[i]

                # Calculate new polynomial degree
                degree[j + 1] = max(degree[j], degree[k] + j - k)

                # Initialize polynomial coefficients
                fst = [field.zero] * (degree[j + 1] + 1)
                fst[: degree[j] + 1] = sigma[j]
                snd = [field.zero] * (degree[j + 1] + 1)
                snd[j - k : degree[k] + j - k + 1] = sigma[k]

                # Calculate new polynomial coefficients
                sigma[j + 1] = [fst[i] + snd[i] * discrepancy[j] / discrepancy[k] for i in range(degree[j + 1] + 1)]

            # Calculate next discrepancy
            if j < (self.t * 2 - 2):
                discrepancy[j + 1] = syndrome[j + 1]
                for i in range(degree[j + 1]):
                    discrepancy[j + 1] += sigma[j + 1][i + 1] * syndrome[j - i]

        return sigma[self.t * 2 - 1]

    def _find_error_locations(self, error_locator_poly: List[Any]) -> List[int]:
        """Find the error locations by finding the roots of the error locator polynomial.

        Args:
            error_locator_poly: Coefficients of the error locator polynomial

        Returns:
            List of error positions in the codeword
        """
        # Use BinaryPolynomial to represent the error locator polynomial
        poly = BinaryPolynomial(0)
        for i, coef in enumerate(error_locator_poly):
            if coef != self.field.zero:
                poly.value |= 1 << i

        # Find the roots of the error locator polynomial
        roots = []
        for i in range(1, self.field.size):
            # Evaluate polynomial at alpha^i
            elem = self.field(i)
            value = poly.evaluate(elem)
            if value == self.field(0):
                roots.append(i)

        return roots

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using the Berlekamp algorithm.

        Args:
            received: Received codeword tensor. The last dimension should be
                    a multiple of the code length (n).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
                return_errors: If True, also return the estimated error patterns.

        Returns:
            Either:
            - Decoded tensor containing estimated messages
            - A tuple of (decoded tensor, error pattern tensor) if return_errors=True

        Raises:
            ValueError: If the last dimension of received is not a multiple of n.
        """
        return_errors = kwargs.get("return_errors", False)

        # Check input dimensions
        *leading_dims, L = received.shape
        if L % self.code_length != 0:
            raise ValueError(f"Last dimension ({L}) must be divisible by code length ({self.code_length})")

        # Process blockwise
        def decode_block(r_block):
            batch_size = r_block.shape[0]
            decoded = torch.zeros(batch_size, self.code_dimension, dtype=received.dtype, device=received.device)
            errors = torch.zeros_like(r_block)

            for i in range(batch_size):
                # Get the current received word
                r = r_block[i]

                # Convert to field elements
                r_field = [self.field.element(int(bit)) for bit in r]

                # Calculate syndrome
                syndrome = self.encoder.calculate_syndrome_polynomial(r_field)

                # Check if syndrome is zero (no errors)
                if all(s == self.field.zero for s in syndrome):
                    # No errors, just extract the message
                    decoded[i] = self.encoder.extract_message(r)
                    continue

                # Find error locator polynomial using Berlekamp-Massey algorithm
                error_locator = self.berlekamp_massey_algorithm(syndrome)

                # Find error locations
                error_positions = self._find_error_locations(error_locator)

                # Create error pattern
                error_pattern = torch.zeros_like(r)
                for pos in error_positions:
                    if 0 <= pos < self.code_length:
                        error_pattern[pos] = 1
                errors[i] = error_pattern

                # Correct errors
                corrected = (r + error_pattern) % 2

                # Extract message bits
                decoded[i] = self.encoder.extract_message(corrected)

            return (decoded, errors) if return_errors else decoded

        # Apply decoding blockwise
        return apply_blockwise(received, self.code_length, decode_block)
