"""Systematic linear block coding module for forward error correction.

This module implements systematic linear block coding for binary data transmission, a specific
form of linear block coding where the information bits appear unchanged in predefined positions
of the codeword. The remaining positions, called the parity set, contain parity bits calculated
from the information bits :cite:`lin2004error,moon2005error,richardson2008modern`.

Systematic codes have the advantage that the original message can be directly extracted from
the codeword without decoding, making them practical for many applications. Their generator
matrices have a specific structure where the columns indexed by the information set form an
identity matrix.
"""

from typing import Any, List, Tuple, Union

import torch

from kaira.models.registry import ModelRegistry

from ..utils import apply_blockwise
from .linear_block_code import LinearBlockCodeEncoder


def create_systematic_generator_matrix(parity_submatrix: torch.Tensor, information_set: Union[List[int], torch.Tensor, str] = "left") -> torch.Tensor:
    """Create a systematic generator matrix from a parity submatrix and information set.

    Args:
        parity_submatrix: The parity submatrix P of shape (k, m) where k is the dimension
            and m is the redundancy.
        information_set: Either indices of information positions, which must be a k-sublist
            of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.

    Returns:
        A systematic generator matrix of shape (k, n)

    Raises:
        ValueError: If information_set is invalid.
    """
    k, m = parity_submatrix.shape
    n = k + m

    # Create a generator matrix of the proper size
    generator_matrix = torch.zeros((k, n), dtype=parity_submatrix.dtype)

    # Get information and parity sets
    information_indices, parity_indices = get_information_and_parity_sets(k, n, information_set)

    # Set the identity matrix at the information positions
    generator_matrix[:, information_indices] = torch.eye(k, dtype=parity_submatrix.dtype)

    # Set the parity submatrix at the parity positions
    generator_matrix[:, parity_indices] = parity_submatrix

    return generator_matrix


def get_information_and_parity_sets(k: int, n: int, information_set: Union[List[int], torch.Tensor, str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine the information and parity sets for a systematic code.

    Args:
        k: Code dimension (information length)
        n: Code length
        information_set: Either indices of information positions, which must be a k-sublist
            of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.

    Returns:
        Tuple containing:
            - information_indices: Tensor of information set indices
            - parity_indices: Tensor of parity set indices

    Raises:
        ValueError: If information_set is invalid.
    """
    # Process the information set
    if isinstance(information_set, str):
        if information_set == "left":
            information_indices = torch.arange(k)
            parity_indices = torch.arange(k, n)
        elif information_set == "right":
            information_indices = torch.arange(n - k, n)
            parity_indices = torch.arange(n - k)
        else:
            raise ValueError("If string, information_set must be 'left' or 'right'")
    else:
        # Convert to tensor if it's a list
        if isinstance(information_set, list):
            information_indices = torch.tensor(information_set)
        else:
            information_indices = information_set

        # Validate information indices
        if information_indices.size(0) != k or information_indices.min() < 0 or information_indices.max() >= n:
            raise ValueError(f"information_set must be a {k}-sublist of [0...{n})")

        # Calculate parity indices as the complement
        all_indices = torch.arange(n)
        parity_indices = torch.tensor([i for i in all_indices if i not in information_indices])

    return information_indices, parity_indices


@ModelRegistry.register_model("systematic_linear_block_code_encoder")
class SystematicLinearBlockCodeEncoder(LinearBlockCodeEncoder):
    r"""Encoder for systematic linear block coding.

    A systematic linear block code is a linear block code in which the information bits
    can be found in predefined positions in the codeword, called the information set K,
    which is a k-sublist of [0...n). The remaining positions are called the parity set M,
    which is an m-sublist of [0...n).

    In this case, the generator matrix has the property that the columns indexed by K
    are equal to I_k (identity matrix), and the columns indexed by M are equal to P
    (the parity submatrix). The check matrix has the property that the columns indexed
    by M are equal to I_m, and the columns indexed by K are equal to P^T.

    This implementation follows the standard approach to systematic linear block coding
    described in the error control coding literature :cite:`lin2004error,moon2005error,sklar2001digital`.

    Args:
        parity_submatrix (torch.Tensor): The parity submatrix for the code.
        information_set: Either indices of information positions, which must be a k-sublist
            of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
    """

    def __init__(self, parity_submatrix: torch.Tensor, information_set: Union[List[int], torch.Tensor, str] = "left", **kwargs: Any):
        """Initialize the systematic linear block encoder.

        Args:
            parity_submatrix: The parity submatrix P for the code.
                Must be a binary matrix of shape (k, m) where k is the message length
                and m is the redundancy.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If parity_submatrix or information_set are invalid.
        """
        # Ensure parity submatrix is a torch tensor
        if not isinstance(parity_submatrix, torch.Tensor):
            parity_submatrix = torch.tensor(parity_submatrix, dtype=torch.float32)

        # Store local copies of key attributes
        self._parity_submatrix = parity_submatrix
        k, m = self._parity_submatrix.shape
        n = k + m

        # Store the original information_set configuration if it's a string
        self._info_set_config = information_set if isinstance(information_set, str) else None

        # Store the information and parity sets before using them to create the generator matrix
        self._information_set, self._parity_set = get_information_and_parity_sets(k, n, information_set)

        # Create the systematic generator matrix
        generator_matrix = create_systematic_generator_matrix(parity_submatrix=parity_submatrix, information_set=information_set)

        # Make a copy of kwargs to avoid modifying the original
        kwargs_copy = kwargs.copy()
        if "generator_matrix" in kwargs_copy:
            del kwargs_copy["generator_matrix"]

        # Initialize the parent class with this generator matrix
        # This will set up _length, _dimension, _redundancy, check_matrix, and generator_right_inverse
        super().__init__(generator_matrix=generator_matrix, **kwargs_copy)

        # After parent initialization, register buffers with different names to avoid conflicts with properties
        self.register_buffer("_info_set_buffer", self._information_set)
        self.register_buffer("_parity_set_buffer", self._parity_set)
        self.register_buffer("_parity_submatrix_buffer", self._parity_submatrix)

    @property
    def information_set(self) -> torch.Tensor:
        """Either indices of information positions, which must be a k-sublist of [0...n), or one of
        the strings 'left' or 'right'.

        Default is 'left'.
        """
        return self._info_set_buffer

    @property
    def parity_submatrix(self) -> torch.Tensor:
        """Parity submatrix P of the code."""
        return self._parity_submatrix_buffer

    @property
    def parity_set(self) -> torch.Tensor:
        """Parity set M of the code."""
        return self._parity_set_buffer

    def project_word(self, x: torch.Tensor) -> torch.Tensor:
        """Project a codeword onto the information set.

        This extracts the information bits directly from a codeword without
        decoding, which is a key advantage of systematic codes.

        Args:
            x: Input tensor of shape (..., codeword_length) or (..., b*codeword_length)
               where b is a positive integer.

        Returns:
            Projected tensor of shape (..., message_length) or (..., b*message_length)

        Raises:
            ValueError: If the last dimension of the input is not a multiple of n.
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of n
        if last_dim_size % self._length != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"the code length {self._length}")

        # Define projection function to apply to blocks
        def projection_fn(reshaped_x):
            # Extract information bits directly from the corresponding positions
            return reshaped_x[..., self.information_set]

        # Use apply_blockwise to handle the projection
        return apply_blockwise(x, self._length, projection_fn)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode the input tensor using systematic encoding.

        For systematic codes, encoding can be done efficiently by placing information
        bits directly in the information positions and calculating parity bits only.
        This implementation is optimized compared to the general matrix multiplication
        used in the parent class.

        Args:
            x: The input tensor of shape (..., message_length) or (..., b*message_length)
               where b is a positive integer.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Encoded tensor of shape (..., codeword_length) or (..., b*codeword_length)

        Raises:
            ValueError: If the last dimension of the input is not a multiple of k.
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of k
        if last_dim_size % self._dimension != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"the code dimension {self._dimension}")

        # Define systematic encoding function to apply to blocks
        def systematic_encode_fn(reshaped_x):
            # Compute parity bits
            parity_bits = torch.matmul(reshaped_x, self.parity_submatrix.to(reshaped_x.dtype)) % 2

            # Create output tensor of the right shape
            batch_shape = reshaped_x.shape[:-1]
            codewords = torch.zeros((*batch_shape, self._length), dtype=reshaped_x.dtype, device=reshaped_x.device)

            # Place information bits directly
            codewords[..., self.information_set] = reshaped_x

            # Place parity bits
            codewords[..., self.parity_set] = parity_bits

            return codewords

        # Use apply_blockwise to handle the encoding
        return apply_blockwise(x, self._dimension, systematic_encode_fn)

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            String representation with key parameters
        """
        return f"{self.__class__.__name__}(" f"parity_submatrix=tensor(...), " f"information_set=tensor({self._information_set.tolist()}), " f"dimension={self._dimension}, " f"length={self._length}, " f"redundancy={self._redundancy}" f")"
