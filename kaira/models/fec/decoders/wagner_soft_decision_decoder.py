"""Wagner soft-decision decoder for single parity-check codes.

This module implements the Wagner algorithm for soft-decision decoding of single parity-check
codes. The algorithm efficiently decodes single parity-check codes by making decisions based on the
reliability of received soft values, providing optimal performance in AWGN channels.
"""

from typing import Any, Tuple, Union

import torch

from kaira.models.fec.encoders.block_code import BlockCodeEncoder

from ..utils import apply_blockwise
from .base import BlockDecoder


class WagnerSoftDecisionDecoder(BlockDecoder[BlockCodeEncoder]):
    """Wagner soft-decision decoder for single parity-check codes.

    This decoder implements the Wagner algorithm, which is designed specifically
    for single parity-check codes with soft-decision inputs. It works by:
    1. Making hard decisions based on the sign of the received values
    2. Checking if the parity constraint is satisfied
    3. If not, flipping the bit with the smallest absolute value (i.e., the least reliable bit)

    This approach is optimal for single parity-check codes under AWGN channels.

    Attributes:
        encoder (BlockCodeEncoder): The encoder instance

    Args:
        encoder (BlockCodeEncoder): The encoder for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class
    """

    def __init__(self, encoder: BlockCodeEncoder, *args: Any, **kwargs: Any):
        """Initialize the Wagner soft-decision decoder."""
        super().__init__(encoder, *args, **kwargs)

        # Verify that this is a single parity-check code
        if self.code_length != self.code_dimension + 1:
            raise ValueError(f"Wagner decoder is only applicable to single parity-check codes. " f"Expected code_length = code_dimension + 1, " f"got code_length = {self.code_length}, code_dimension = {self.code_dimension}")

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received soft values using the Wagner algorithm.

        Args:
            received: Received soft-decision tensor. The last dimension should be
                    a multiple of the code length (n). For soft inputs, positive values
                    represent 0 bits and negative values represent 1 bits.
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
            decoded = torch.zeros(batch_size, self.code_dimension, dtype=torch.int, device=received.device)

            for i in range(batch_size):
                # Get the current received word (soft values)
                r = r_block[i]

                # Make hard decisions based on sign
                hard_decisions = (r < 0).to(torch.int)

                # Check parity (even parity expected)
                if torch.sum(hard_decisions) % 2 != 0:
                    # Find the least reliable bit
                    least_reliable_idx = torch.argmin(torch.abs(r))

                    # Flip the least reliable bit
                    hard_decisions[least_reliable_idx] = 1 - hard_decisions[least_reliable_idx]

                # Extract message bits (assuming systematic form where parity is the last bit)
                decoded[i] = hard_decisions[: self.code_dimension]

            if return_errors:
                # For return_errors, we need to calculate the error pattern
                # This is the difference between our hard decisions and the original hard decisions
                original_hard = (r_block < 0).to(torch.int)
                errors = (hard_decisions != original_hard).to(torch.int)
                return decoded, errors
            else:
                return decoded

        # Apply decoding blockwise
        return apply_blockwise(received, self.code_length, decode_block)
