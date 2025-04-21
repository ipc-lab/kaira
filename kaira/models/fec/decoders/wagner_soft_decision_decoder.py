"""Wagner soft-decision decoder for single parity-check codes.

This module implements the Wagner algorithm for soft-decision decoding of single parity-check
codes. The algorithm efficiently decodes single parity-check codes by making decisions based on the
reliability of received soft values, providing optimal performance in AWGN channels.

The Wagner algorithm is a classic technique for soft-decision decoding of single parity-check codes.
It leverages reliability information from the channel to make optimal decisions, providing
significant performance gains over hard-decision decoding approaches. This algorithm is especially
valuable in applications involving concatenated codes where single parity-check codes serve as
component codes.

:cite:`wagner1986simple`
:cite:`hagenauer1989iterative`
:cite:`moon2005error`
"""

from typing import Any, Tuple, Union

import torch

from kaira.models.fec.encoders.base import BaseBlockCodeEncoder

from .base import BaseBlockDecoder


class WagnerSoftDecisionDecoder(BaseBlockDecoder[BaseBlockCodeEncoder]):
    """Wagner soft-decision decoder for single parity-check codes.

    This decoder implements the Wagner algorithm :cite:`wagner1986simple`, which is designed
    specifically for single parity-check codes with soft-decision inputs. It leverages
    reliability information to make optimal decoding decisions under the assumption of an
    AWGN channel.

    The Wagner algorithm works by:
    1. Making initial hard decisions based on the sign of the received soft values
    2. Checking if the parity constraint is satisfied by these decisions
    3. If not, flipping the bit with the smallest absolute value (least reliable bit)

    This simple but elegant approach achieves optimal maximum likelihood decoding for
    single parity-check codes with soft inputs :cite:`moon2005error,hagenauer1989iterative`.

    Attributes:
        encoder (BaseBlockCodeEncoder): The encoder instance providing code parameters
                                       and encoding functionality

    Args:
        encoder (BaseBlockCodeEncoder): The encoder for the code being decoded. Must represent
                                       a single parity-check code where code_length = code_dimension + 1
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Raises:
        ValueError: If the encoder does not represent a single parity-check code

    Examples:
        >>> from kaira.models.fec.encoders import SingleParityCheckCodeEncoder
        >>> from kaira.models.fec.decoders import WagnerSoftDecisionDecoder
        >>> import torch
        >>>
        >>> # Create a (4,3) single parity-check code encoder and decoder
        >>> encoder = SingleParityCheckCodeEncoder(code_dimension=3)
        >>> decoder = WagnerSoftDecisionDecoder(encoder)
        >>>
        >>> # Encode a message
        >>> message = torch.tensor([1., 0., 1.])
        >>> codeword = encoder(message)
        >>> print(codeword)  # Output: tensor([1., 0., 1., 0.])
        >>>
        >>> # Simulate soft decision values from AWGN channel
        >>> # Positive values represent 0, negative values represent 1
        >>> # The magnitudes represent reliability
        >>> soft_received = torch.tensor([-2.1, 1.5, -1.8, 0.2])
        >>>
        >>> # Decode using the Wagner algorithm
        >>> decoded = decoder(soft_received)
        >>> print(decoded)  # Output: tensor([1, 0, 1])
    """

    def __init__(self, encoder: BaseBlockCodeEncoder, *args: Any, **kwargs: Any):
        """Initialize the Wagner soft-decision decoder.

        Sets up the decoder with an encoder instance and verifies that it
        represents a single parity-check code, which is required for the
        Wagner algorithm.

        Args:
            encoder: The encoder instance for the code being decoded
            *args: Variable positional arguments passed to the base class
            **kwargs: Variable keyword arguments passed to the base class

        Raises:
            ValueError: If the encoder does not represent a single parity-check code,
                      i.e., if code_length ≠ code_dimension + 1
        """
        super().__init__(encoder, *args, **kwargs)

        # Verify that this is a single parity-check code
        if self.code_length != self.code_dimension + 1:
            raise ValueError(f"Wagner decoder is only applicable to single parity-check codes. " f"Expected code_length = code_dimension + 1, " f"got code_length = {self.code_length}, code_dimension = {self.code_dimension}")

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received soft values using the Wagner algorithm.

        This method implements the complete Wagner soft-decision decoding process:
        1. Make initial hard decisions based on the sign of received values
        2. Check if the parity constraint is satisfied
        3. If not, find the least reliable bit (smallest absolute value) and flip it
        4. Extract the message bits from the corrected codeword

        The algorithm assumes soft-input values where:
        - Positive values represent a likelihood of '0' bits
        - Negative values represent a likelihood of '1' bits
        - The magnitude of the value represents the reliability of the decision

        Args:
            received: Received soft-decision tensor with shape (..., n) or (..., m*n)
                     where n is the code length and m is some multiple.
                     Values represent soft bit likelihoods (e.g., LLRs) where:
                     - Positive values suggest bit=0
                     - Negative values suggest bit=1
                     - Magnitude indicates reliability
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
                     return_errors: If True, also return the estimated error patterns compared to the initial hard decisions

        Returns:
            Either:
            - Decoded tensor containing estimated messages with shape (..., k) or (..., m*k)
            - A tuple of (decoded tensor, error pattern tensor) if return_errors=True

        Raises:
            ValueError: If the last dimension of received is not a multiple of the code length

        Note:
            This decoder provides optimal maximum likelihood performance for single parity-check
            codes under AWGN channels. It is computationally efficient, requiring only O(n)
            operations for an (n,n-1) single parity-check code.
        """
        return_errors = kwargs.get("return_errors", False)

        # Check input dimensions
        *leading_dims, L = received.shape
        if L % self.code_length != 0:
            raise ValueError(f"Last dimension ({L}) must be divisible by code length ({self.code_length})")

        # If no batch dimension, add one for consistency
        if not leading_dims:
            received = received.unsqueeze(0)
            added_batch = True
        else:
            added_batch = False

        # Reshape to blocks if there are multiple code blocks
        num_blocks = received.size(-1) // self.code_length
        if num_blocks > 1:
            # Reshape from (..., L) to (..., num_blocks, code_length)
            received = received.view(*received.size()[:-1], num_blocks, self.code_length)
        else:
            # Add block dimension: (..., code_length) to (..., 1, code_length)
            received = received.unsqueeze(-2)

        # At this point received has shape (..., num_blocks, code_length)
        # Make hard decisions based on sign
        hard_decisions = (received < 0).to(torch.int)

        # Store original hard decisions if needed for error calculation
        if return_errors:
            original_hard = hard_decisions.clone()

        # Check parity (even parity expected)
        parity_sums = hard_decisions.sum(dim=-1) % 2

        # For blocks with odd parity, find and flip the least reliable bit
        for indices in torch.nonzero(parity_sums == 1, as_tuple=False):
            # Get batch and block indices
            *batch_indices, block_idx = indices.tolist()

            # Find the least reliable bit in this block
            block_values = received[batch_indices + [block_idx]]
            least_reliable_idx = torch.argmin(torch.abs(block_values))

            # Flip the least reliable bit
            hard_decisions[batch_indices + [block_idx, least_reliable_idx]] = 1 - hard_decisions[batch_indices + [block_idx, least_reliable_idx]]

        # Extract message bits (assuming systematic form where message bits come first)
        decoded = hard_decisions[..., : self.code_dimension]

        # Calculate error pattern if required
        if return_errors:
            errors = (hard_decisions != original_hard).to(torch.int)

            # Reshape back to original dimensions
            if num_blocks > 1:
                errors = errors.reshape(*errors.size()[:-2], -1)
                decoded = decoded.reshape(*decoded.size()[:-2], -1)
            else:
                errors = errors.squeeze(-2)
                decoded = decoded.squeeze(-2)

            # Remove batch dimension if it was added
            if added_batch:
                errors = errors.squeeze(0)
                decoded = decoded.squeeze(0)

            return decoded, errors
        else:
            # Reshape back to original dimensions
            if num_blocks > 1:
                decoded = decoded.reshape(*decoded.size()[:-2], -1)
            else:
                decoded = decoded.squeeze(-2)

            # Remove batch dimension if it was added
            if added_batch:
                decoded = decoded.squeeze(0)

            return decoded
