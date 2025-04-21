"""Syndrome lookup decoder for forward error correction.

This module implements a syndrome table-based decoding approach for linear block codes. The decoder
works by precomputing a lookup table that maps each possible syndrome to its corresponding error
pattern (coset leader), which enables efficient decoding.

The syndrome lookup approach is a classic decoding method for linear block codes, providing
optimal hard-decision decoding when using the standard array and coset leaders. It is particularly
effective for codes with smaller block lengths, as the syndrome table size grows exponentially
with the code's redundancy.

:cite:`lin2004error`
:cite:`moon2005error`
:cite:`macwilliams1977theory`
"""

from typing import Any, Dict, Tuple, Union

import torch

from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder

from ..utils import apply_blockwise
from .base import BaseBlockDecoder


class SyndromeLookupDecoder(BaseBlockDecoder[LinearBlockCodeEncoder]):
    """Syndrome lookup decoder for linear block codes.

    This decoder implements syndrome-based hard-decision decoding using a precomputed table
    of coset leaders. It is efficient for small to medium sized codes but becomes impractical
    for larger codes due to the exponential growth of the syndrome table :cite:`lin2004error`.

    The standard array decoding approach provides maximum-likelihood decoding for binary
    symmetric channels when errors are equally likely in all positions. The decoder works by:

    1. Calculating the syndrome of the received codeword: s = H·r^T
    2. Looking up the most likely error pattern (coset leader) for that syndrome
    3. Correcting the received word by XORing it with the error pattern: v = r + e
    4. Extracting the message bits from the corrected codeword

    Attributes:
        encoder (LinearBlockCodeEncoder): The encoder instance providing code parameters
                                         and syndrome calculation functions
        _syndrome_table (Dict[int, torch.Tensor]): Lookup table mapping syndromes to
                                                  their corresponding error patterns

    Args:
        encoder (LinearBlockCodeEncoder): The encoder for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Raises:
        TypeError: If the encoder is not a LinearBlockCodeEncoder

    Examples:
        >>> from kaira.models.fec.encoders import LinearBlockCodeEncoder
        >>> from kaira.models.fec.decoders import SyndromeLookupDecoder
        >>> import torch
        >>>
        >>> # Create a (7,4) Hamming code encoder and decoder
        >>> G = torch.tensor([
        ...     [1, 0, 0, 0, 1, 1, 0],
        ...     [0, 1, 0, 0, 1, 0, 1],
        ...     [0, 0, 1, 0, 0, 1, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ], dtype=torch.float)
        >>> encoder = LinearBlockCodeEncoder(generator_matrix=G)
        >>> decoder = SyndromeLookupDecoder(encoder)
        >>>
        >>> # Encode a message
        >>> message = torch.tensor([1., 0., 1., 1.])
        >>> codeword = encoder(message)
        >>>
        >>> # Introduce an error
        >>> received = codeword.clone()
        >>> received[2] = 1 - received[2]  # Flip a bit
        >>>
        >>> # Decode
        >>> decoded = decoder(received)
        >>> print(torch.all(decoded == message))
        True
    """

    def __init__(self, encoder: LinearBlockCodeEncoder, *args: Any, **kwargs: Any):
        """Initialize the syndrome lookup decoder.

        This constructor sets up the decoder and builds the syndrome lookup table,
        which maps each possible syndrome to its corresponding error pattern with
        minimum Hamming weight (coset leader).

        Args:
            encoder: The encoder instance for the code being decoded
            *args: Variable positional arguments passed to the base class
            **kwargs: Variable keyword arguments passed to the base class

        Raises:
            TypeError: If the encoder is not a LinearBlockCodeEncoder

        Note:
            For large codes, building the syndrome table can be computationally expensive,
            as it requires exploring a large space of error patterns. The table size is
            2^r where r is the code's redundancy.
        """
        # Check encoder type before calling super().__init__
        if not isinstance(encoder, LinearBlockCodeEncoder):
            raise TypeError(f"Encoder must be a LinearBlockCodeEncoder, got {type(encoder).__name__}")

        self._validate_encoder_type(encoder)

        super().__init__(encoder, *args, **kwargs)

        # Build syndrome table during initialization
        self._syndrome_table = self._build_syndrome_table()

    def _validate_encoder_type(self, encoder: LinearBlockCodeEncoder) -> None:
        """Validate that the encoder is of the correct type.

        This method exists to support testing and can be overridden in tests.

        Args:
            encoder: The encoder to validate

        Raises:
            TypeError: If the encoder is not valid for this decoder
        """
        encoder_class_name = encoder.__class__
        if not issubclass(encoder.__class__, LinearBlockCodeEncoder):
            raise TypeError(f"Encoder must be a LinearBlockCodeEncoder, got {encoder_class_name.__name__}")

    def _build_syndrome_table(self) -> Dict[int, torch.Tensor]:
        """Build the syndrome lookup table for maximum likelihood decoding.

        Creates a dictionary mapping each possible syndrome to its corresponding
        coset leader (the error pattern with the minimum Hamming weight in its coset).
        This implements the standard array decoding approach from coding theory.

        The construction is performed by:
        1. Starting with the zero error pattern (corresponding to no errors)
        2. Adding all error patterns of weight 1
        3. Adding all error patterns of weight 2, and so on

        For each error pattern, if its syndrome isn't already in the table, it becomes
        the coset leader for that syndrome.

        Returns:
            Dictionary mapping syndrome bit patterns (as integers) to error pattern tensors

        Note:
            For an (n,k) linear code, there are 2^(n-k) possible syndromes, each associated
            with a unique coset of the code. This method finds the minimum weight vector
            in each coset.
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

        Creates all binary vectors of length n (code length) with exactly 'weight'
        ones. These represent all possible error patterns with 'weight' bit flips.

        Args:
            weight: The Hamming weight (number of 1s) in the error patterns

        Returns:
            Tensor containing all error patterns with the specified weight

        Note:
            The number of patterns generated is binomial(n,weight), which can be
            very large for moderate values of n and weight.
        """
        if weight == 0:
            return torch.zeros((1, self.code_length), dtype=torch.int)

        # For weight 1, we can simply create the standard basis vectors
        if weight == 1:
            patterns = torch.zeros((self.code_length, self.code_length), dtype=torch.int)
            for i in range(self.code_length):
                patterns[i, i] = 1
            return patterns

        # For higher weights, use a combinatorial approach
        patterns = []
        current = torch.zeros(self.code_length, dtype=torch.int)

        def generate_recursive(current: torch.Tensor, ones_left: int, start_pos: int):
            if ones_left == 0:
                patterns.append(current.clone())
                return

            # Try each position from start_pos to the end
            for pos in range(start_pos, self.code_length - ones_left + 1):
                current[pos] = 1
                generate_recursive(current, ones_left - 1, pos + 1)
                current[pos] = 0  # Backtrack

        generate_recursive(current, weight, 0)
        return torch.stack(patterns)

    def _syndrome_to_int(self, syndrome: torch.Tensor) -> int:
        """Convert a syndrome tensor to an integer for table lookup.

        Transforms a binary syndrome vector into an integer value that can be
        used as a key in the syndrome lookup table dictionary.

        Args:
            syndrome: Binary syndrome tensor of shape (r,) where r is the redundancy

        Returns:
            Integer representation of the syndrome (treating syndrome as binary number)

        Example:
            If syndrome = [1, 0, 1], this returns 5 (binary 101)
        """
        # Make sure we're dealing with a 1D tensor
        if syndrome.dim() > 1:
            if syndrome.size(0) == 1:
                syndrome = syndrome.squeeze(0)
            else:
                raise ValueError(f"Expected 1D syndrome tensor, got shape {syndrome.shape}")

        # Convert to integers for binary operations
        syndrome = syndrome.int()

        result = 0
        for i, bit in enumerate(syndrome):
            if bit.item() == 1:
                result |= 1 << i
        return result

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using the syndrome lookup table.

        This method implements the complete syndrome-based decoding process:
        1. Calculate the syndrome of the received word
        2. Look up the corresponding error pattern in the syndrome table
        3. Correct the received word by adding (XORing) the error pattern
        4. Extract the message from the corrected codeword

        Args:
            received: Received codeword tensor with shape (..., n) or (..., m*n)
                     where n is the code length and m is some multiple
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
                return_errors: If True, also return the estimated error patterns

        Returns:
            Either:
            - Decoded tensor containing estimated messages with shape (..., k) or (..., m*k)
            - A tuple of (decoded tensor, error pattern tensor) if return_errors=True

        Raises:
            ValueError: If the last dimension of received is not a multiple of the code length

        Note:
            This decoder provides maximum likelihood (ML) decoding for the binary
            symmetric channel (BSC) when all error patterns of the same weight
            are equally likely.
        """
        return_errors = kwargs.get("return_errors", False)

        # Check input dimensions
        *leading_dims, L = received.shape
        if L % self.code_length != 0:
            raise ValueError(f"Last dimension ({L}) must be divisible by code length ({self.code_length})")

        # Handle 1D tensor input for single codeword
        if not leading_dims:  # This is a 1D tensor (a single codeword)
            # Add batch dimension for processing
            batched_received = received.unsqueeze(0)

            # Process directly without using apply_blockwise
            batch_size = 1
            decoded = torch.zeros(batch_size, self.code_dimension, dtype=received.dtype, device=received.device)
            errors = torch.zeros_like(batched_received)

            # Calculate syndrome
            syndrome = self.encoder.calculate_syndrome(received)
            syndrome_int = self._syndrome_to_int(syndrome)

            # Look up error pattern
            error_pattern = self._syndrome_table.get(syndrome_int, torch.zeros(self.code_length, dtype=torch.int))
            errors[0] = error_pattern

            # Correct errors
            corrected = (received + error_pattern) % 2

            # Extract message bits
            decoded[0] = self.encoder.extract_message(corrected)

            # Remove batch dimension for output
            if return_errors:
                return decoded.squeeze(0), errors.squeeze(0)
            return decoded.squeeze(0)

        # For tensors with leading dimensions, process blockwise
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

                # Extract message bits
                decoded[i] = self.encoder.extract_message(corrected)

            return (decoded, errors) if return_errors else decoded

        # Apply decoding blockwise
        result = apply_blockwise(received, self.code_length, decode_block)

        # If we're returning errors and handling multi-block tensors
        # apply_blockwise will return a tuple that we need to handle specially
        if return_errors and L > self.code_length:
            decoded_parts = []
            error_parts = []

            # Handle batch dimension cases
            *_, blocks, _ = received.shape
            for i in range(blocks):
                decoded, errors = result[:, i]
                decoded_parts.append(decoded)
                error_parts.append(errors)

            # Stack the parts along the appropriate dimension
            decoded_tensor = torch.cat(decoded_parts, dim=-1)
            error_tensor = torch.cat(error_parts, dim=-1)

            return decoded_tensor, error_tensor

        return result
