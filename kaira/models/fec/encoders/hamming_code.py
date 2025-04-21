"""Hamming code implementation for forward error correction.

This module implements Hamming codes, a family of linear error-correcting codes that can detect
up to two-bit errors and correct single-bit errors. For a given parameter μ ≥ 2, a Hamming code
has the following parameters:

- Length: n = 2^μ - 1
- Dimension: k = 2^μ - μ - 1
- Redundancy: m = μ
- Minimum distance: d = 3

In its extended version, the Hamming code has the following parameters:

- Length: n = 2^μ
- Dimension: k = 2^μ - μ - 1
- Redundancy: m = μ + 1
- Minimum distance: d = 4

Hamming codes are perfect codes, meaning they achieve the theoretical limit for the number
of correctable errors given their length and dimension :cite:`lin2004error,moon2005error`.
"""

import itertools
from functools import lru_cache
from typing import Any, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder


def create_hamming_parity_submatrix(mu: int, extended: bool = False, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create the parity submatrix for a Hamming code.

    The parity submatrix has columns that are all possible non-zero binary μ-tuples.
    For extended Hamming codes, an additional row of all ones is added :cite:`lin2004error`.

    Args:
        mu: The parameter μ of the code. Must satisfy μ ≥ 2.
        extended: Whether to create an extended Hamming code. Default is False.
        dtype: The data type for tensor elements. Default is torch.float32.
        device: The device to place the resulting tensor on. Default is None (uses current device).

    Returns:
        The parity submatrix of the Hamming code.
    """
    # Validate input
    if mu < 2:
        raise ValueError("'mu' must be at least 2")

    # Calculate dimensions
    k = 2**mu - mu - 1  # Dimension (information length)
    m = mu  # Redundancy (parity length)

    # Create empty parity submatrix
    parity_submatrix = torch.zeros((k, m), dtype=dtype, device=device)

    # Optimized implementation for small mu values (common case)
    if mu <= 8:  # Arbitrary threshold based on practical use cases
        # Create all possible weight-1 binary tuples directly

        # Each column of the check matrix is a non-zero binary μ-tuple
        # For Hamming codes, we can generate these systematically

        # Start counter for filling parity submatrix
        row_idx = 0

        # Generate all weight 2+ combinations
        for w in range(2, mu + 1):
            for indices in itertools.combinations(range(mu), w):
                # Create a tuple with 1s at the specified positions
                row = torch.zeros(mu, dtype=dtype, device=device)
                row.index_fill_(0, torch.tensor(indices, device=device), 1.0)
                parity_submatrix[row_idx, :] = row
                row_idx += 1
    else:
        # For very large mu values, use the original implementation
        # Create all binary tuples of length μ (except all zeros)
        nonzero_tuples = []
        for w in range(1, mu + 1):
            for indices in itertools.combinations(range(mu), w):
                binary_tuple = torch.zeros(mu, dtype=dtype, device=device)
                binary_tuple[list(indices)] = 1
                nonzero_tuples.append(binary_tuple)

        # Construct check matrix with all nonzero tuples as columns
        check_matrix = torch.stack(nonzero_tuples, dim=1)

        # Create systematic parity submatrix by rearranging columns
        # The parity submatrix P consists of the columns of the check matrix
        # corresponding to the information set
        i = 0
        for w in range(2, mu + 1):
            for indices in itertools.combinations(range(mu), w):
                tuple_idx = nonzero_tuples.index(torch.zeros(mu, dtype=dtype, device=device).index_put_([list(indices)], torch.ones(len(indices), device=device)))
                parity_submatrix[i, :] = check_matrix[:, tuple_idx].T
                i += 1

    # For extended Hamming code, add an overall parity check
    if extended:
        # Add a row of all ones to the parity submatrix
        parity_extension = torch.ones((k, 1), dtype=dtype, device=device)
        parity_submatrix = torch.cat([parity_submatrix, parity_extension], dim=1)

    return parity_submatrix


@ModelRegistry.register_model("hamming_code_encoder")
class HammingCodeEncoder(SystematicLinearBlockCodeEncoder):
    r"""Encoder for Hamming codes.

    Hamming codes are linear error-correcting codes that can detect up to two-bit errors
    and correct single-bit errors. They are perfect codes, meaning they achieve the
    theoretical limit for the number of correctable errors given their length and dimension
    :cite:`lin2004error,richardson2008modern`.

    For a given parameter μ ≥ 2, a Hamming code has the following parameters:
    - Length: n = 2^μ - 1
    - Dimension: k = 2^μ - μ - 1
    - Redundancy: m = μ
    - Minimum distance: d = 3

    In its extended version, the Hamming code has the following parameters:
    - Length: n = 2^μ
    - Dimension: k = 2^μ - μ - 1
    - Redundancy: m = μ + 1
    - Minimum distance: d = 4

    The implementation follows standard techniques in error control coding literature
    :cite:`lin2004error,moon2005error,sklar2001digital`.

    Args:
        mu (int): The parameter μ of the code. Must satisfy μ ≥ 2.
        extended (bool, optional): Whether to use the extended version of the Hamming code.
            Default is False.
        information_set (Union[List[int], torch.Tensor, str], optional): Information set
            specification. Default is "left".
        dtype (torch.dtype, optional): Data type for internal tensors. Default is torch.float32.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> encoder = HammingCodeEncoder(mu=3)
        >>> print(f"Length: {encoder.length}, Dimension: {encoder.dimension}, Redundancy: {encoder.redundancy}")
        Length: 7, Dimension: 4, Redundancy: 3
        >>> message = torch.tensor([1., 0., 1., 1.])
        >>> codeword = encoder(message)
        >>> print(codeword)
        tensor([1., 0., 1., 1., 0., 1., 1.])

        >>> # Using the extended version
        >>> ext_encoder = HammingCodeEncoder(mu=3, extended=True)
        >>> print(f"Length: {ext_encoder.length}, Dimension: {ext_encoder.dimension}, Redundancy: {ext_encoder.redundancy}")
        Length: 8, Dimension: 4, Redundancy: 4
        >>> message = torch.tensor([1., 0., 1., 1.])
        >>> codeword = ext_encoder(message)
        >>> print(codeword)
        tensor([1., 0., 1., 1., 0., 1., 1., 0.])
    """

    def __init__(self, mu: int, extended: bool = False, information_set: Union[List[int], torch.Tensor, str] = "left", dtype: torch.dtype = torch.float32, **kwargs: Any):
        """Initialize the Hamming code encoder.

        Args:
            mu: The parameter μ of the code. Must satisfy μ ≥ 2.
            extended: Whether to use the extended version of the Hamming code.
                Default is False.
            information_set: Either indices of information positions, which must be a k-sublist
                of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
            dtype: Data type for internal tensors. Default is torch.float32.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If mu < 2.
        """
        if mu < 2:
            raise ValueError("'mu' must be at least 2")

        # Store parameters
        self._mu = mu
        self._extended = extended
        self._dtype = dtype

        # Calculate theoretical parameters based on mu
        self._theoretical_length = 2**mu - 1 if not extended else 2**mu
        self._theoretical_dimension = 2**mu - mu - 1
        self._theoretical_redundancy = mu if not extended else mu + 1

        # Get device from kwargs if provided
        device = kwargs.get("device", None)

        # Create parity submatrix for Hamming code
        parity_submatrix = create_hamming_parity_submatrix(mu=mu, extended=extended, dtype=dtype, device=device)

        # Initialize the parent class with this parity submatrix
        super().__init__(parity_submatrix=parity_submatrix, information_set=information_set, **kwargs)

        # Validate that the calculated dimensions match the theoretical ones
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate that the code dimensions match the theoretical values."""
        if self._length != self._theoretical_length:
            raise ValueError(f"Code length mismatch: calculated {self._length}, " f"expected {self._theoretical_length}")
        if self._dimension != self._theoretical_dimension:
            raise ValueError(f"Code dimension mismatch: calculated {self._dimension}, " f"expected {self._theoretical_dimension}")
        if self._redundancy != self._theoretical_redundancy:
            raise ValueError(f"Code redundancy mismatch: calculated {self._redundancy}, " f"expected {self._theoretical_redundancy}")

    @property
    def mu(self) -> int:
        """Get the parameter μ of the code."""
        return self._mu

    @property
    def extended(self) -> bool:
        """Get whether this is an extended Hamming code."""
        return self._extended

    @lru_cache(maxsize=None)
    def minimum_distance(self) -> int:
        """Calculate the minimum Hamming distance of the code.

        Returns:
            The minimum Hamming distance:
            - 3 for standard Hamming code
            - 4 for extended Hamming code
        """
        return 4 if self._extended else 3

    def __repr__(self) -> str:
        """Return a string representation of the encoder.

        Returns:
            A string representation with key parameters
        """
        return f"{self.__class__.__name__}(" f"mu={self._mu}, " f"extended={self._extended}, " f"length={self._length}, " f"dimension={self._dimension}, " f"redundancy={self._redundancy}, " f"dtype={self._dtype.__repr__()}" f")"

    def inverse_encode(self, y):
        """Decode a codeword back to its original message, correcting single-bit errors.

        Args:
            y: A tensor of codewords to decode. The last dimension should match the code length.
                Supports batch processing with arbitrary batch dimensions.

        Returns:
            tuple: (decoded_message, syndrome)
                - decoded_message: The decoded information bits with corrected errors
                - syndrome: The syndrome vectors for each codeword
        """
        # Calculate syndrome
        syndrome = self.calculate_syndrome(y)

        # Prepare shapes
        original_dims = y.size()[:-1]
        y_reshaped = y.reshape(-1, self.code_length).clone()
        syndrome_reshaped = syndrome.reshape(-1, self.redundancy)

        # Vectorized error correction
        error_positions = torch.tensor([self._syndrome_to_error_position(s) for s in syndrome_reshaped], device=y.device)

        # Create mask for valid error positions (less than code_length)
        valid_errors = error_positions < self.code_length

        # Apply corrections using vectorized operations
        if valid_errors.any():
            # Create batch indices for the samples with errors
            batch_indices = torch.nonzero(valid_errors, as_tuple=True)[0]
            # Get corresponding error positions
            pos = error_positions[valid_errors]

            # Flip the bits at error positions
            for i, p in zip(batch_indices, pos):
                y_reshaped[i, p] = 1 - y_reshaped[i, p]

        # Extract information bits
        decoded = y_reshaped[..., self.information_set]
        decoded = decoded.reshape(*original_dims, self.code_dimension)

        return decoded, syndrome

    def _syndrome_to_error_position(self, syndrome):
        """Convert a syndrome to an error position by matching check matrix columns."""
        # check_matrix shape: (m, n)
        # compare syndrome to each column of check_matrix
        H = self.check_matrix  # shape (m, n)
        # ensure float dtype
        syn = syndrome.float()
        for j in range(self.code_length):
            col = H[:, j].float()
            if torch.equal(col, syn):
                return j
        return self.code_length
