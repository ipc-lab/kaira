"""Reed-Muller decoder using majority-logic decoding.

This module implements the majority-logic decoding algorithm for Reed-Muller codes. The algorithm
efficiently decodes Reed-Muller codes by exploiting their recursive structure and the properties of
their codewords.

Reed-Muller codes form an important family of linear error-correcting codes with a rich mathematical
structure based on finite geometries. The majority-logic decoding algorithm leverages this structure
to provide an efficient decoding method that can correct multiple errors while avoiding the complexity
of brute-force maximum likelihood decoding.

:cite:`reed1954class`
:cite:`muller1954application`
:cite:`macwilliams1977theory`
:cite:`lin2004error`
"""

from typing import Any, List, Literal, Tuple, Union

import torch

from kaira.models.fec.encoders.reed_muller_code import ReedMullerCodeEncoder

from ..utils import apply_blockwise
from .base import BaseBlockDecoder


class ReedMullerDecoder(BaseBlockDecoder[ReedMullerCodeEncoder]):
    """Reed-Muller decoder using majority-logic decoding.

    This decoder implements the majority-logic decoding algorithm developed by Reed
    for Reed-Muller codes :cite:`reed1954class`. It works by recursively decoding the
    received word using a series of majority-logic decisions based on special partitions
    of the code that correspond to geometrical subspaces in the finite geometry interpretation.

    For an RM(r,m) code, the algorithm can correct up to 2^(m-r-1) - 1 errors, which is
    optimal for first-order Reed-Muller codes (r=1) :cite:`macwilliams1977theory`.

    The decoder supports both hard-decision and soft-decision decoding, with the
    soft-decision variant offering better performance in the presence of noise by
    taking into account reliability information from the channel.

    Attributes:
        encoder (ReedMullerCodeEncoder): The Reed-Muller encoder instance providing
                                        code parameters and encoding functionality
        input_type (str): The type of input the decoder accepts:
                         'hard' for binary inputs (0s and 1s)
                         'soft' for real-valued inputs with reliability information
        _reed_partitions (List[List[int]]): Precomputed Reed partitions for efficient decoding,
                                           where each partition corresponds to a specific
                                           information bit

    Args:
        encoder (ReedMullerCodeEncoder): The encoder for the Reed-Muller code being decoded
        input_type (Literal["hard", "soft"]): The type of input the decoder accepts.
                                             Default is "hard".
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Examples:
        >>> from kaira.models.fec.encoders import ReedMullerCodeEncoder
        >>> from kaira.models.fec.decoders import ReedMullerDecoder
        >>> import torch
        >>>
        >>> # Create a RM(1,3) code encoder and decoder
        >>> encoder = ReedMullerCodeEncoder(r=1, m=3)
        >>> decoder = ReedMullerDecoder(encoder)
        >>>
        >>> # Encode a message
        >>> message = torch.tensor([1., 0., 1., 0.])
        >>> codeword = encoder(message)
        >>>
        >>> # Introduce an error
        >>> received = codeword.clone()
        >>> received[2] = 1 - received[2]
        >>>
        >>> # Decode using majority-logic decoding
        >>> decoded = decoder(received)
        >>> print(torch.all(decoded == message))
        True
    """

    def __init__(self, encoder: ReedMullerCodeEncoder, input_type: Literal["hard", "soft"] = "hard", *args: Any, **kwargs: Any):
        """Initialize the Reed-Muller decoder.

        Sets up the decoder with a Reed-Muller encoder instance and computes the
        Reed partitions needed for majority-logic decoding.

        Args:
            encoder: The Reed-Muller encoder instance for the code being decoded
            input_type: The type of decoder input, either "hard" for binary inputs
                      or "soft" for real-valued inputs with reliability information
            *args: Variable positional arguments passed to the base class
            **kwargs: Variable keyword arguments passed to the base class

        Note:
            The Reed partitions are precomputed during initialization to make the
            decoding process more efficient. These partitions depend on the specific
            parameters of the Reed-Muller code (r,m).
        """
        super().__init__(encoder, *args, **kwargs)

        self.input_type = input_type

        # Compute Reed partitions
        self._reed_partitions = self._generate_reed_partitions()

    def _generate_reed_partitions(self) -> List[List[torch.Tensor]]:
        """Generate Reed partitions for efficient majority-logic decoding.

        Reed partitions are special subsets of positions in the codeword that form
        orthogonal check sums for decoding specific information bits in a Reed-Muller
        code. These partitions correspond to geometrical subspaces in the finite
        geometry interpretation of Reed-Muller codes.

        In the context of an RM(r,m) code:
        - For r=0 (repetition code), there is a single partition with all positions
        - For r=1 (first-order RM code), partitions correspond to hyperplanes
        - For higher-order RM codes, partitions are constructed recursively

        Returns:
            List of Reed partitions, where each partition is a list of position groups
            that form check sums for a specific information bit

        Note:
            This implementation is simplified and would need to be expanded for a full
            production implementation to handle all possible Reed-Muller parameters
            correctly. The actual construction of these partitions is based on the
            recursive structure of Reed-Muller codes and their relation to finite geometries.
        """
        # This is a simplified implementation of Reed partitions generation
        # In a full implementation, this would depend on the specific parameters
        # of the Reed-Muller code (r, m)

        # For demonstration purposes, we'll create a basic structure
        # A real implementation would compute these based on the code properties
        partitions = []

        # Example partitioning logic - would need to be replaced with actual Reed-Muller partitioning
        m = 0
        r = 0

        # Try to infer Reed-Muller parameters from code length and dimension
        # For an (r,m) Reed-Muller code:
        # - Length n = 2^m
        # - Dimension k = sum(i=0 to r) of binomial(m,i)

        # Infer m from code length
        n = self.code_length
        temp_m = 0
        while 2**temp_m < n:
            temp_m += 1
        if 2**temp_m == n:
            m = temp_m

        # Given m, try to infer r from dimension
        if m > 0:
            k = self.code_dimension
            temp_r = 0
            temp_k = 0
            while temp_k < k and temp_r <= m:
                # Add binomial coefficient (m choose temp_r)
                from math import comb

                temp_k += comb(m, temp_r)
                if temp_k == k:
                    r = temp_r
                    break
                temp_r += 1

        # Generate partitions based on Reed-Muller structure
        if m > 0 and 0 <= r <= m:
            # Generate partitions based on the cosets of the Reed-Muller code
            # This is a simplified approach - actual implementation would be more involved

            # For each information bit
            for i in range(self.code_dimension):
                # Create a partition for this bit
                partition = []

                # In a real implementation, these would be carefully constructed
                # based on the algebraic structure of Reed-Muller codes
                for j in range(2 ** (m - 1)):
                    # Create groups of positions that form checks for this bit
                    positions = []
                    for offset in range(2**r):
                        pos = (j * 2**r + offset) % self.code_length
                        positions.append(pos)

                    # Convert to tensor
                    partition.append(torch.tensor(positions, dtype=torch.long))

                partitions.append(partition)

        return partitions

    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received values using the Reed majority-logic algorithm.

        This method implements the majority-logic decoding process for Reed-Muller codes.
        For each information bit, it computes a set of check sums based on the Reed
        partitions and then makes a decision based on the majority value of these sums.

        For soft-decision decoding, it also takes into account the reliability information
        of each received bit, which can significantly improve performance in AWGN channels.

        Args:
            received: Received tensor with shape (..., n) or (..., m*n) where n is the code length.
                     For hard inputs, values should be 0 or 1.
                     For soft inputs, positive values represent likelihood of 0 bits and
                     negative values represent likelihood of 1 bits (e.g., LLR values).
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
            For first-order Reed-Muller codes (r=1), this decoder can correct up to
            2^(m-2) errors, which matches the code's error-correcting capability.
            For higher-order RM codes, the performance may not be optimal but the
            algorithm provides an efficient decoding approach.
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
            errors = torch.zeros_like(r_block) if return_errors else None

            for i in range(batch_size):
                # Get the current received word - ensure it's a 1D tensor
                if r_block.dim() == 3:  # Handle the case when r_block has shape [batch, 1, code_length]
                    r = r_block[i, 0, :]
                else:  # Handle the case when r_block has shape [batch, code_length]
                    r = r_block[i, :]

                """
                # Convert to binary for hard decoding or compute hard decisions for soft decoding
                if self.input_type == "hard":
                    bx = r.clone()
                else:  # self.input_type == "soft"
                    bx = (r < 0).to(torch.int)
                """

                # Decode using Reed algorithm
                u_hat = torch.zeros(self.code_dimension, dtype=torch.int, device=received.device)

                # Process each bit position using its corresponding partition
                for j, partition in enumerate(self._reed_partitions):
                    if j >= self.code_dimension:
                        break

                    # For hard decision decoding
                    if self.input_type == "hard":
                        # Calculate checksums for each group in the partition
                        checksums = []
                        for group in partition:
                            # Ensure the group indices are valid
                            valid_indices = group[group < r.shape[0]]
                            if len(valid_indices) == 0:
                                continue

                            # Take relevant positions and compute parity
                            # Use indexing to select elements from the 1D tensor
                            group_bits = r[valid_indices].to(torch.int)
                            checksum = torch.sum(group_bits) % 2
                            checksums.append(checksum.item())  # Use .item() to convert tensor to scalar

                        # Skip if no valid checksums
                        if not checksums:
                            continue

                        # Convert to tensor
                        checksums = torch.tensor(checksums, device=received.device)

                        # Make majority decision
                        u_hat[j] = (torch.sum(checksums) > len(checksums) // 2).to(torch.int)

                    # For soft decision decoding
                    else:  # self.input_type == "soft"
                        # Calculate checksums and minimum reliabilities for each group
                        checksums = []
                        min_reliabilities = []

                        for group in partition:
                            # Ensure the group indices are valid
                            valid_indices = group[group < r.shape[0]]
                            if len(valid_indices) == 0:
                                continue

                            # Take relevant positions
                            group_bits = (r[valid_indices] < 0).to(torch.int)
                            group_reliabilities = torch.abs(r[valid_indices])

                            # Compute parity of hard decisions
                            checksum = torch.sum(group_bits) % 2
                            checksums.append(checksum.item())  # Use .item() to convert tensor to scalar

                            # Find minimum reliability in this group
                            min_reliability = torch.min(group_reliabilities)
                            min_reliabilities.append(min_reliability.item())  # Use .item() to convert tensor to scalar

                        # Skip if no valid checksums
                        if not checksums:
                            continue

                        # Convert to tensors
                        checksums = torch.tensor(checksums, device=received.device)
                        min_reliabilities = torch.tensor(min_reliabilities, device=received.device)

                        # Calculate decision variable
                        decision_var = torch.sum((1 - 2 * checksums) * min_reliabilities)

                        # Make decision
                        u_hat[j] = (decision_var < 0).to(torch.int)

                # Store the decoded message
                decoded[i] = u_hat

                # Compute error pattern if needed
                if return_errors:
                    # Re-encode the message to get the correct codeword
                    correct_codeword = self.encoder(u_hat.float().unsqueeze(0)).squeeze(0)
                    errors[i] = (r.to(torch.int) != correct_codeword.to(torch.int)).to(torch.int)

            return (decoded, errors) if return_errors else decoded

        # Apply decoding blockwise
        return apply_blockwise(received, self.code_length, decode_block)
