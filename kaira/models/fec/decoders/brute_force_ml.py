"""Brute Force Maximum Likelihood decoder for forward error correction.

This module implements a brute-force maximum likelihood decoding approach for block codes by
searching through all possible codewords to find the one closest to the received word in terms of
Hamming distance, which is equivalent to maximum likelihood decoding in symmetric channels.

The maximum likelihood decoding principle selects the codeword that has the highest probability
of having been transmitted, given the received word. For binary symmetric channels, this is
equivalent to finding the codeword with the minimum Hamming distance to the received word.
While this approach guarantees optimal decoding performance, its computational complexity
grows exponentially with the code dimension, making it practical only for small codes.

:cite:`lin2004error`
:cite:`moon2005error`
:cite:`proakis2008digital`
"""

from typing import Any, Tuple, Union

import torch

from kaira.models.fec.encoders.base import BaseBlockCodeEncoder

from ..utils import apply_blockwise
from .base import BaseBlockDecoder


class BruteForceMLDecoder(BaseBlockDecoder[BaseBlockCodeEncoder]):
    """Brute Force Maximum Likelihood decoder for block codes.

    This decoder implements a brute-force maximum likelihood approach by searching
    through all possible codewords to find the one that is closest to the received
    word in terms of Hamming distance. It is optimal in the sense that it performs
    maximum-likelihood decoding for symmetric channels :cite:`proakis2008digital`, but
    becomes computationally infeasible for larger codes.

    The brute force ML decoding principle is based on finding:
    argmin_{c ∈ C} d(r, c)
    where d(r, c) is the Hamming distance between the received word r and codeword c,
    and C is the set of all codewords in the code.

    The decoder works by:
    1. Generating all possible codewords (or using a precomputed codebook)
    2. Computing the Hamming distance between the received word and each codeword
    3. Selecting the codeword with the minimum distance (maximum likelihood)
    4. Extracting the message bits from the selected codeword

    Attributes:
        encoder (BaseBlockCodeEncoder): The encoder instance providing encoding functionality
        _codebook (torch.Tensor): Precomputed tensor of all codewords with shape (2^k, n)
        _message_map (torch.Tensor): Mapping from codeword indices to message bits with shape (2^k, k)

    Args:
        encoder (BaseBlockCodeEncoder): The encoder for the code being decoded
        precompute_codebook (bool): Whether to precompute the entire codebook during initialization.
                                   Default is True, which is more efficient for multiple decoding
                                   operations but requires more memory.
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Examples:
        >>> from kaira.models.fec.encoders import LinearBlockCodeEncoder
        >>> from kaira.models.fec.decoders import BruteForceMLDecoder
        >>> import torch
        >>>
        >>> # Create a simple (7,4) Hamming code encoder and decoder
        >>> G = torch.tensor([
        ...     [1, 0, 0, 0, 1, 1, 0],
        ...     [0, 1, 0, 0, 1, 0, 1],
        ...     [0, 0, 1, 0, 0, 1, 1],
        ...     [0, 0, 0, 1, 1, 1, 1]
        ... ], dtype=torch.float)
        >>> encoder = LinearBlockCodeEncoder(generator_matrix=G)
        >>> decoder = BruteForceMLDecoder(encoder)
        >>>
        >>> # Encode a message
        >>> message = torch.tensor([1., 0., 1., 1.])
        >>> codeword = encoder(message)
        >>>
        >>> # Introduce two bit errors
        >>> received = codeword.clone()
        >>> received[2] = 1 - received[2]
        >>> received[5] = 1 - received[5]
        >>>
        >>> # Decode using ML decoding
        >>> decoded = decoder(received)
        >>> print(torch.all(decoded == message))
        True
    """

    def __init__(self, encoder: BaseBlockCodeEncoder, precompute_codebook: bool = True, *args: Any, **kwargs: Any):
        """Initialize the brute force ML decoder.

        Sets up the decoder with an encoder instance and optionally precomputes
        the complete codebook for more efficient decoding operations.

        Args:
            encoder: The encoder instance for the code being decoded
            precompute_codebook: Whether to generate all possible codewords during
                                initialization (True) or on-demand (False)
            *args: Variable positional arguments passed to the base class
            **kwargs: Variable keyword arguments passed to the base class

        Note:
            Precomputing the codebook requires O(2^k * n) memory, where k is the
            code dimension and n is the code length. This can be prohibitive for
            larger codes, so set precompute_codebook=False for such cases.
        """
        super().__init__(encoder, *args, **kwargs)

        if precompute_codebook:
            self._codebook, self._message_map = self._generate_codebook()
        else:
            self._codebook = None
            self._message_map = None

    def _generate_codebook(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all possible codewords for the code.

        This method enumerates all 2^k possible messages (where k is the code dimension)
        and encodes each one to create the complete codebook of the code. It also
        maintains a mapping from each codeword back to its original message.

        The codebook is essential for the brute force ML decoding approach, as it
        allows the decoder to compare the received word with all valid codewords
        to find the most likely transmitted codeword.

        Returns:
            Tuple containing:
            - Tensor of shape (2^k, n) containing all possible codewords
            - Tensor of shape (2^k, k) containing all possible messages

        Note:
            For codes with dimension k > 20, this function may require excessive
            memory and computation time. Consider alternative decoding methods
            for such large codes.
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

        The Hamming distance counts the number of positions at which the
        corresponding elements of two vectors are different. It is a key
        metric in coding theory and is used by the ML decoder to determine
        the closest codeword to a received word.

        Args:
            x: First binary vector or batch of vectors
            y: Second binary vector or batch of vectors of the same shape as x

        Returns:
            Tensor containing the Hamming distance(s) along the last dimension

        Example:
            >>> x = torch.tensor([1, 0, 1, 0, 1])
            >>> y = torch.tensor([1, 1, 0, 0, 1])
            >>> self._hamming_distance(x, y)
            tensor(2)
        """
        return torch.sum((x != y).to(torch.int), dim=-1)

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords using maximum likelihood decoding.

        This method implements the complete maximum likelihood decoding process:
        1. Compare the received word with every possible codeword
        2. Find the codeword that minimizes the Hamming distance to the received word
        3. Return the message bits corresponding to that codeword

        This provides optimal decoding performance for the binary symmetric channel (BSC)
        in terms of minimizing the word error probability.

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
            This decoder provides optimal (maximum likelihood) decoding for the binary
            symmetric channel at the cost of exponential complexity in the code dimension k.
            For larger codes, consider using more efficient decoders that may sacrifice
            some performance for computational tractability.
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
