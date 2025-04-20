"""Syndrome lookup decoder for forward error correction.

This module implements a syndrome table-based decoding approach for linear block codes. The decoder
works by precomputing a lookup table that maps each possible syndrome to its corresponding error
pattern (coset leader), which enables efficient decoding.
"""

from typing import Any, Dict, Tuple, Union

import torch

from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder

from ..utils import apply_blockwise
from .base import BlockDecoder


class SyndromeLookupDecoder(BlockDecoder[LinearBlockCodeEncoder]):
    """Syndrome lookup decoder for linear block codes.

    This decoder implements syndrome-based hard-decision decoding using a precomputed table
    of coset leaders. It is efficient for small to medium sized codes but becomes impractical
    for larger codes due to the exponential growth of the syndrome table.

    The decoder works by:
    1. Calculating the syndrome of the received codeword
    2. Looking up the most likely error pattern (coset leader) for that syndrome
    3. Correcting the received word by XORing it with the error pattern
    4. Extracting the message bits from the corrected codeword

    Attributes:
        encoder (LinearBlockCodeEncoder): The encoder instance
        _syndrome_table (Dict[int, torch.Tensor]): Lookup table mapping syndromes to error patterns

    Args:
        encoder (LinearBlockCodeEncoder): The encoder for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class
    """

    def __init__(self, encoder: LinearBlockCodeEncoder, *args: Any, **kwargs: Any):
        """Initialize the syndrome table decoder."""
        super().__init__(encoder, *args, **kwargs)

        if not isinstance(encoder, LinearBlockCodeEncoder):
            raise TypeError(f"Encoder must be a LinearBlockCodeEncoder, got {type(encoder).__name__}")

        # Build syndrome table during initialization
        self._syndrome_table = self._build_syndrome_table()

    def _build_syndrome_table(self) -> Dict[int, torch.Tensor]:
        """Build the syndrome lookup table.

        Creates a dictionary mapping each possible syndrome to its corresponding
        coset leader (the error pattern with the minimum Hamming weight).

        Returns:
            Dictionary mapping syndrome bit patterns (as integers) to error pattern tensors
        """
        table: Dict[int, torch.Tensor] = {}

        # Start with zero-weight error pattern (no errors)
        zero_error = torch.zeros(self.code_length, dtype=torch.int)
        zero_syndrome = self.encoder.calculate_syndrome(zero_error)
        syndrome_int = self._syndrome_to_int(zero_syndrome)
        table[syndrome_int] = zero_error

        # Continue with weight-1 error patterns, then weight-2, etc., until all syndromes are covered
        for weight in range(1, self.code_length + 1):
            # If we've found all possible syndromes, we can stop
            if len(table) == 2**self.redundancy:
                break

            # Generate all error patterns of current weight
            for error_pattern in self._generate_error_patterns(weight):
                syndrome = self.encoder.calculate_syndrome(error_pattern)
                syndrome_int = self._syndrome_to_int(syndrome)

                # Only add to table if this syndrome hasn't been seen before
                if syndrome_int not in table:
                    table[syndrome_int] = error_pattern

        return table

    def _generate_error_patterns(self, weight: int) -> torch.Tensor:
        """Generate all possible error patterns with a given Hamming weight.

        Args:
            weight: The Hamming weight (number of 1s) in the error patterns

        Returns:
            Tensor containing all error patterns with the specified weight
        """
        if weight == 0:
            return torch.zeros((1, self.code_length), dtype=torch.int)

        # This is a simplified implementation that generates patterns sequentially
        # For a production system, this should be optimized for large code lengths
        patterns = []

        # Helper function for recursive generation
        def generate_recursive(current: torch.Tensor, ones_left: int, pos: int):
            if ones_left == 0:
                patterns.append(current.clone())
                return

            if pos + ones_left > self.code_length:
                return

            # Skip this position
            generate_recursive(current, ones_left, pos + 1)

            # Use this position
            current[pos] = 1
            generate_recursive(current, ones_left - 1, pos + 1)
            current[pos] = 0

        current = torch.zeros(self.code_length, dtype=torch.int)
        generate_recursive(current, weight, 0)

        return torch.stack(patterns)

    def _syndrome_to_int(self, syndrome: torch.Tensor) -> int:
        """Convert a syndrome tensor to an integer for table lookup.

        Args:
            syndrome: Binary syndrome tensor

        Returns:
            Integer representation of the syndrome
        """
        result = 0
        for i, bit in enumerate(syndrome):
            if bit:
                result |= 1 << i
        return result

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using the syndrome table.

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

                # Calculate syndrome
                syndrome = self.encoder.calculate_syndrome(r)
                syndrome_int = self._syndrome_to_int(syndrome)

                # Look up error pattern
                error_pattern = self._syndrome_table.get(syndrome_int, torch.zeros(self.code_length, dtype=torch.int))
                errors[i] = error_pattern

                # Correct errors
                corrected = (r + error_pattern) % 2

                # Extract message bits (assuming systematic encoder)
                decoded[i] = self.encoder.extract_message(corrected)

            return (decoded, errors) if return_errors else decoded

        # Apply decoding blockwise
        return apply_blockwise(received, self.code_length, decode_block)
