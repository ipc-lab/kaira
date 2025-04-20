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
from typing import Any, List, Optional, Tuple, Union

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

    Attributes:
        mu (int): The parameter μ of the code. Must satisfy μ ≥ 2.
        extended (bool): Whether this is an extended Hamming code.
        length (int): Code length (n)
        dimension (int): Code dimension (k)
        redundancy (int): Code redundancy (r = n - k)
        parity_submatrix (torch.Tensor): The parity submatrix P of the code
        generator_matrix (torch.Tensor): The generator matrix G of the code
        check_matrix (torch.Tensor): The parity check matrix H of the code

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
        """Get the parameter μ of the code.

        Returns:
            The parameter μ
        """
        return self._mu

    @property
    def extended(self) -> bool:
        """Get whether this is an extended Hamming code.

        Returns:
            True if this is an extended Hamming code, False otherwise
        """
        return self._extended

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type used for internal tensors.

        Returns:
            The data type
        """
        return self._dtype

    @lru_cache(maxsize=None)
    def minimum_distance(self) -> int:
        """Calculate the minimum Hamming distance of the code.

        Returns:
            The minimum Hamming distance:
            - 3 for standard Hamming code
            - 4 for extended Hamming code
        """
        return 4 if self._extended else 3

    def correct_single_error(self, received_word: torch.Tensor) -> torch.Tensor:
        """Correct a single error in the received word.

        This method utilizes the error-correcting capability of Hamming codes to correct
        a single bit error in the received word. For the extended code, it can correct
        a single error and detect double errors :cite:`lin2004error,moon2005error`.

        Args:
            received_word: The received word tensor with shape (..., n) or (..., b*n)
                where n is the code length and b is a positive integer.

        Returns:
            The corrected word tensor with the same shape as the input.
        """
        # Calculate syndrome
        syndrome = self.calculate_syndrome(received_word)

        # For standard Hamming codes
        if not self._extended:
            # Reshape inputs to handle batch dimensions
            orig_shape = received_word.shape
            if received_word.dim() > 1:
                # Handle batched input
                received_word = received_word.reshape(-1, self._length)
                syndrome = syndrome.reshape(-1, self._redundancy)
            else:
                # Add batch dimension for consistent processing
                received_word = received_word.unsqueeze(0)
                syndrome = syndrome.unsqueeze(0)

            batch_size = received_word.size(0)
            corrected = received_word.clone()

            # Process each syndrome in the batch
            for i in range(batch_size):
                syn = syndrome[i]

                # No error if syndrome is all zeros
                if torch.all(syn == 0):
                    continue

                # Convert syndrome to column index in the parity check matrix
                # For Hamming codes, syndrome directly gives error position in non-systematic form
                # Convert to our actual code positions based on check matrix
                error_pos = torch.where(torch.all(self.check_matrix.T == syn.unsqueeze(1), dim=0))[0]

                if len(error_pos) > 0:
                    # Flip the bit at the error position
                    corrected[i, error_pos[0]] = 1 - corrected[i, error_pos[0]]

            # Reshape back to original shape
            corrected = corrected.reshape(orig_shape)
            return corrected

        else:
            # For extended Hamming code
            # First check overall parity (last syndrome bit)
            # Reshape inputs to handle batch dimensions
            orig_shape = received_word.shape
            if received_word.dim() > 1:
                # Handle batched input
                received_word = received_word.reshape(-1, self._length)
                syndrome = syndrome.reshape(-1, self._redundancy)
            else:
                # Add batch dimension for consistent processing
                received_word = received_word.unsqueeze(0)
                syndrome = syndrome.unsqueeze(0)

            batch_size = received_word.size(0)
            corrected = received_word.clone()

            # Process each syndrome in the batch
            for i in range(batch_size):
                syn = syndrome[i]
                overall_parity = syn[-1]  # Last bit is overall parity
                trimmed_syn = syn[:-1]  # First bits are standard Hamming syndrome

                # Case 1: No errors
                if torch.all(syn == 0):
                    continue

                # Case 2: Double error (standard syndrome is 0, but overall parity is wrong)
                if torch.all(trimmed_syn == 0) and overall_parity != 0:
                    # Can't correct double error - leave as is
                    continue

                # Case 3: Single error (can correct)
                # Convert syndrome to column index in the parity check matrix
                # For standard part of syndrome, identify the position
                error_pos = torch.where(torch.all(self.check_matrix[:-1, :].T == trimmed_syn.unsqueeze(1), dim=0))[0]

                if len(error_pos) > 0:
                    # Flip the bit at the error position
                    corrected[i, error_pos[0]] = 1 - corrected[i, error_pos[0]]

            # Reshape back to original shape
            corrected = corrected.reshape(orig_shape)
            return corrected

    def decode(self, received_word: torch.Tensor) -> torch.Tensor:
        """Decode a received word using Hamming decoding.

        For Hamming codes, decoding consists of extracting the message part
        after applying the error correction.

        Args:
            received_word: The received word tensor with shape (..., n) or (..., b*n)
                where n is the code length and b is a positive integer.

        Returns:
            The decoded message tensor with shape (..., k) or (..., b*k).
        """
        # First correct any single error in the received word
        corrected = self.correct_single_error(received_word)

        # Then use the parent class's decoding logic to extract the message
        return self.project_word(corrected)

    def encode_decode_pipeline(self, message: torch.Tensor, error_rate: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a complete encode-transmit-decode pipeline with optional errors.

        This is a convenience method that encodes a message, optionally introduces
        bit-flip errors, and then attempts to decode and correct the received word.
        The approach follows standard techniques for binary symmetric channels
        :cite:`sklar2001digital,richardson2008modern`.

        Args:
            message: The message tensor to encode
            error_rate: Probability of bit flip for each bit in the codeword (default: 0.0)

        Returns:
            Tuple containing:
                - Original message
                - Decoded message
                - Syndrome of the received word
        """
        # Encode the message
        codeword = self(message)

        # Introduce bit-flip errors based on the error rate
        if error_rate > 0.0:
            noise = torch.bernoulli(torch.full(codeword.shape, error_rate, dtype=codeword.dtype, device=codeword.device))
            received_word = (codeword + noise) % 2
        else:
            received_word = codeword

        # Decode the received word
        decoded_message = self.decode(received_word)

        # Calculate the syndrome of the received word
        syndrome = self.calculate_syndrome(received_word)

        return message, decoded_message, syndrome

    def __repr__(self) -> str:
        """Return a string representation of the encoder.

        Returns:
            A string representation with key parameters
        """
        return f"{self.__class__.__name__}(" f"mu={self._mu}, " f"extended={self._extended}, " f"length={self._length}, " f"dimension={self._dimension}, " f"redundancy={self._redundancy}, " f"dtype={self._dtype.__repr__()}" f")"
