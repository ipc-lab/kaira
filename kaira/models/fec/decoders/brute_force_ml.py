"""Brute Force Maximum Likelihood decoder for forward error correction.

This module implements a brute-force maximum likelihood decoding approach for block codes by
searching through all possible codewords to find the one closest to the received word in terms of
Hamming distance, which is equivalent to maximum likelihood decoding in symmetric channels.
"""

from typing import Any, Tuple, Union

import torch

from kaira.models.fec.encoders.block_code import BlockCodeEncoder

from ..utils import apply_blockwise
from .base import BlockDecoder


class BruteForceMLDecoder(BlockDecoder[BlockCodeEncoder]):
    """Brute Force Maximum Likelihood decoder for block codes.

    This decoder implements a brute-force maximum likelihood approach by searching
    through all possible codewords to find the one that is closest to the received
    word in terms of Hamming distance. It is optimal in the sense that it performs
    maximum-likelihood decoding for symmetric channels, but becomes computationally
    infeasible for larger codes.

    The decoder works by:
    1. Generating all possible codewords (or using a precomputed codebook)
    2. Computing the Hamming distance between the received word and each codeword
    3. Selecting the codeword with the minimum distance (maximum likelihood)
    4. Extracting the message bits from the selected codeword

    Attributes:
        encoder (BlockCodeEncoder): The encoder instance
        _codebook (torch.Tensor): Precomputed tensor of all codewords
        _message_map (torch.Tensor): Mapping from codeword indices to message indices

    Args:
        encoder (BlockCodeEncoder): The encoder for the code being decoded
        precompute_codebook (bool): Whether to precompute the entire codebook during initialization
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class
    """

    def __init__(self, encoder: BlockCodeEncoder, precompute_codebook: bool = True, *args: Any, **kwargs: Any):
        """Initialize the brute force ML decoder."""
        super().__init__(encoder, *args, **kwargs)

        if precompute_codebook:
            self._codebook, self._message_map = self._generate_codebook()
        else:
            self._codebook = None
            self._message_map = None

    def _generate_codebook(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all possible codewords for the code.

        This method generates all 2^k possible messages and encodes them to
        create the full codebook of the code.

        Returns:
            Tuple containing:
            - Tensor of shape (2^k, n) containing all codewords
            - Tensor of shape (2^k, k) containing all messages
        """
        k = self.code_dimension
        n = self.code_length

        # Generate all possible messages (2^k)
        num_messages = 2**k
        messages = torch.zeros((num_messages, k), dtype=torch.int)

        # Fill in binary representations
        for i in range(num_messages):
            for j in range(k):
                messages[i, k - j - 1] = (i >> j) & 1

        # Encode all messages to get codewords
        codewords = torch.zeros((num_messages, n), dtype=torch.int)
        for i in range(num_messages):
            codewords[i] = self.encoder(messages[i].unsqueeze(0)).squeeze(0)

        return codewords, messages

    def _hamming_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Hamming distance between two binary vectors.

        Args:
            x: First binary vector
            y: Second binary vector

        Returns:
            Tensor containing the Hamming distance
        """
        return torch.sum((x != y).to(torch.int), dim=-1)

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using maximum likelihood decoding.

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

            # Generate codebook if needed
            if self._codebook is None:
                codebook, message_map = self._generate_codebook()
            else:
                codebook = self._codebook
                message_map = self._message_map

            # Move codebook to same device as received
            codebook = codebook.to(received.device)
            message_map = message_map.to(received.device)

            for i in range(batch_size):
                # Get the current received word
                r = r_block[i]

                # Compute Hamming distance to all codewords
                distances = self._hamming_distance(r.unsqueeze(0).expand(codebook.shape[0], -1), codebook)

                # Find the closest codeword (maximum likelihood)
                min_idx = torch.argmin(distances)
                closest_codeword = codebook[min_idx]

                # Compute the error pattern
                error_pattern = (r != closest_codeword).to(torch.int)
                errors[i] = error_pattern

                # Get the corresponding message
                decoded[i] = message_map[min_idx]

            return (decoded, errors) if return_errors else decoded

        # Apply decoding blockwise
        return apply_blockwise(received, self.code_length, decode_block)
