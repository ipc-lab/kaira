"""Reed-Muller coding module for forward error correction.

This module implements Reed-Muller codes for binary data transmission, a powerful class of
error-correcting codes defined by two parameters: r (order) and m (length parameter).
Reed-Muller codes are versatile linear block codes with well-defined structure and
theoretical properties. They include several other codes as special cases, such as
repetition codes and Hamming codes.

Reed-Muller codes are useful in scenarios requiring reliable transmission with
different error correction capabilities based on application requirements.
For more theoretical background on Reed-Muller codes, see :cite:`lin2004error,moon2005error`.

The implementation follows standard conventions in coding theory for binary linear block codes,
with elements belonging to the binary field GF(2) :cite:`richardson2008modern`.
"""

from itertools import combinations, product
from typing import Any, List, Tuple

import torch

from kaira.models.registry import ModelRegistry

from .linear_block_code import LinearBlockCodeEncoder


def _generate_evaluation_vectors(m: int) -> torch.Tensor:
    """Generate the evaluation vectors for Reed-Muller code.

    Args:
        m: Length parameter (resulting in 2^m code length)

    Returns:
        Tensor of shape (m, 2^m) containing evaluation vectors
    """
    v = torch.zeros((m, 2**m), dtype=torch.int64)
    for i in range(m):
        block_size = 2 ** (m - i - 1)
        # Create a block pattern of 0s followed by 1s
        block = torch.cat([torch.zeros(block_size, dtype=torch.int64), torch.ones(block_size, dtype=torch.int64)])
        # Assign in natural order: row i
        v[i] = block.repeat(2**i)

    return v


def _generate_reed_muller_matrix(r: int, m: int) -> torch.Tensor:
    """Generate the generator matrix for an RM(r,m) code.

    Args:
        r: Order parameter (0 ≤ r < m)
        m: Length parameter (resulting in 2^m code length)

    Returns:
        Generator matrix G for Reed-Muller code
    """
    if not (0 <= r < m):
        raise ValueError(f"Parameters must satisfy 0 ≤ r < m, got r={r}, m={m}")

    # Get evaluation vectors
    v = _generate_evaluation_vectors(m)

    # Build generator matrix
    rows = []

    # Add rows for each r-tuple combination of evaluation vectors
    for order in range(r, 0, -1):
        for indices in combinations(range(m), order):
            # Multiply the selected evaluation vectors elementwise
            row = v[list(indices)].prod(dim=0)
            rows.append(row)

    # Add all-ones row (corresponds to order 0)
    rows.append(torch.ones(2**m, dtype=torch.int64))

    # Stack all rows to form the generator matrix
    return torch.stack(rows) % 2  # Ensure binary values


def calculate_reed_muller_dimension(r: int, m: int) -> int:
    """Calculate the dimension (k) of a Reed-Muller code RM(r,m).

    Args:
        r: Order parameter (0 ≤ r < m)
        m: Length parameter

    Returns:
        Dimension k of the code
    """
    if not (0 <= r < m):
        raise ValueError(f"Parameters must satisfy 0 ≤ r < m, got r={r}, m={m}")

    # Dimension is sum_{i=0}^r binomial(m, i)
    from math import comb

    return sum(comb(m, i) for i in range(r + 1))


@ModelRegistry.register_model("reed_muller_code_encoder")
class ReedMullerCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for Reed-Muller codes.

    Reed-Muller codes are a family of linear error-correcting codes with parameters (r,m)
    where:
    - r is the order (0 ≤ r < m)
    - m is the length parameter

    The resulting code has the following properties:
    - Length: n = 2^m
    - Dimension: k = sum_{i=0}^r (m choose i)
    - Minimum distance: d = 2^(m-r)

    Special cases:
    - RM(0,m) is a repetition code
    - RM(m-1,m) is a single parity-check code
    - RM(1,m) is a first-order Reed-Muller code (also known as a lengthened simplex code)
    - RM(m-2,m) is an extended Hamming code

    This implementation follows the standard approach to Reed-Muller coding described in
    error control coding literature :cite:`lin2004error,moon2005error`.

    Attributes:
        order (int): The order r of the Reed-Muller code (0 ≤ r < m)
        length_param (int): The length parameter m (code length will be 2^m)
        minimum_distance (int): The minimum Hamming distance of the code (2^(m-r))

    Args:
        order (int): The order r of the Reed-Muller code (0 ≤ r < m)
        length_param (int): The length parameter m
        *args: Variable positional arguments passed to the base class.
        **kwargs: Variable keyword arguments passed to the base class.
    """

    def __init__(self, order: int, length_param: int, *args: Any, **kwargs: Any):
        """Initialize the Reed-Muller encoder.

        Args:
            order: The order r of the Reed-Muller code (0 ≤ r < m)
            length_param: The length parameter m (code length will be 2^m)
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If the parameters do not satisfy 0 ≤ r < m
        """
        if not (0 <= order < length_param):
            raise ValueError(f"Parameters must satisfy 0 ≤ r < m, got r={order}, m={length_param}")

        # Generate the generator matrix for the code and cast to float for tensor operations
        matrix = _generate_reed_muller_matrix(order, length_param)
        generator_matrix = matrix.to(torch.float)

        # Initialize the base class with the (float) generator matrix
        super().__init__(generator_matrix, *args, **kwargs)

        # Store Reed-Muller specific parameters
        self.order = order
        self.length_param = length_param
        self.minimum_distance = 2 ** (length_param - order)

    def __repr__(self) -> str:
        """Formal string representation including key parameters."""
        return f"ReedMullerCodeEncoder(order={self.order}, length_param={self.length_param}, " f"length={self.code_length}, dimension={self.code_dimension})"

    @classmethod
    def from_parameters(cls, order: int, length_param: int, *args: Any, **kwargs: Any) -> "ReedMullerCodeEncoder":
        """Create a Reed-Muller encoder from parameters.

        This is an alternative constructor that creates the encoder directly from
        the Reed-Muller parameters.

        Args:
            order: The order r of the Reed-Muller code (0 ≤ r < m)
            length_param: The length parameter m
            *args: Variable positional arguments passed to the constructor.
            **kwargs: Variable keyword arguments passed to the constructor.

        Returns:
            A ReedMullerCodeEncoder instance
        """
        return cls(order, length_param, *args, **kwargs)

    def get_reed_partitions(self) -> List[torch.Tensor]:
        """Get the Reed partitions of the code.

        Reed partitions are useful for certain decoding algorithms, particularly
        majority-logic decoding.

        Returns:
            List of tensors representing the Reed partitions
        """
        r, m = self.order, self.length_param
        reed_partitions = []

        # Generate all binary vectors of various lengths
        binary_vectors = []
        for ell in range(m + 1):
            vectors = torch.tensor(list(product([0, 1], repeat=ell)), dtype=torch.int64)
            if vectors.size(0) > 0:  # Handle empty case for ell=0
                binary_vectors.append(torch.flip(vectors, dims=[1]))
            else:
                binary_vectors.append(torch.zeros((1, 0), dtype=torch.int64))

        # Generate Reed partitions
        for ell in range(r, -1, -1):
            for indices in combinations(range(m), ell):
                # Convert indices to tensor
                set_I = torch.tensor(list(indices), dtype=torch.int64)

                # Get complement set (indices not in set_I)
                set_E = torch.tensor([i for i in range(m) if i not in indices], dtype=torch.int64)

                # Calculate the components
                set_S = torch.matmul(binary_vectors[ell], torch.pow(2, set_I))
                set_Q = torch.matmul(binary_vectors[m - ell], torch.pow(2, set_E))

                # Form the partition
                partition = set_S.unsqueeze(1) + set_Q.unsqueeze(0)
                reed_partitions.append(partition)

        return reed_partitions

    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode codeword(s) by brute-force nearest-codeword search.

        This method implements a maximum-likelihood decoder for Reed-Muller codes
        by finding the closest valid codeword to the received word(s).

        Args:
            x: The received codeword(s). Can be a single vector or a batch.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tuple containing:
                - Decoded message(s)
                - Syndrome (difference between closest valid codeword and received word)
        """
        # Make input a batch
        if x.dim() == 1:
            y2d = x.unsqueeze(0)
            single = True
        else:
            y2d = x
            single = False
        device = y2d.device

        # Enumerate all possible messages (2^k of them)
        import itertools

        msgs = torch.tensor(list(itertools.product([0, 1], repeat=self.code_dimension)), dtype=torch.float, device=device)  # (M, k)

        # Generate their codewords
        cws = (msgs @ self.generator_matrix) % 2  # (M, n)

        # Compute Hamming distances to each received y
        diff = (cws.unsqueeze(1) != y2d.unsqueeze(0)).float()
        dists = diff.sum(dim=2)  # (M, B)
        best = dists.argmin(dim=0)  # (B,)

        # Pick best messages and their codewords
        decoded = msgs[best]  # (B, k)
        pred_cw = cws[best]  # (B, n)
        syndrome = (pred_cw != y2d).float()  # (B, n)

        if single:
            return decoded[0], syndrome[0]
        return decoded, syndrome

    def calculate_syndrome(self, y: torch.Tensor):
        """Return the syndrome (error pattern) for given codeword(s).

        For Reed-Muller codes, we define the syndrome as the difference between
        the received word and the nearest valid codeword.

        Args:
            y: The received codeword(s)

        Returns:
            Syndrome tensor (difference between nearest valid codeword and received word)
        """
        _, syn = self.inverse_encode(y)
        return syn
