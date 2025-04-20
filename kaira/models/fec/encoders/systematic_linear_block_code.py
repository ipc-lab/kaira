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

    Attributes:
        parity_submatrix (torch.Tensor): The parity submatrix P of the code
        information_set (torch.Tensor): Indices of the information positions
        parity_set (torch.Tensor): Indices of the parity positions
        generator_matrix (torch.Tensor): The generator matrix G of the code
        check_matrix (torch.Tensor): The parity check matrix H of the code
        length (int): Code length (n)
        dimension (int): Code dimension (k)
        redundancy (int): Code redundancy (r = n - k)

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
            information_set: Either indices of information positions, which must be a k-sublist
                of [0...n), or one of the strings 'left' or 'right'. Default is 'left'.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If parity_submatrix or information_set are invalid.
        """
        # Ensure parity submatrix is a torch tensor
        if not isinstance(parity_submatrix, torch.Tensor):
            parity_submatrix = torch.tensor(parity_submatrix, dtype=torch.float32)

        self._parity_submatrix = parity_submatrix
        self._k, self._m = self._parity_submatrix.shape
        self._n = self._k + self._m

        # Create the systematic generator matrix
        generator_matrix = create_systematic_generator_matrix(parity_submatrix=parity_submatrix, information_set=information_set)

        # Make a copy of kwargs to avoid modifying the original
        kwargs_copy = kwargs.copy()
        if "generator_matrix" in kwargs_copy:
            del kwargs_copy["generator_matrix"]

        # Initialize the parent class with this generator matrix
        # This will set up _length, _dimension, _redundancy, check_matrix, and generator_right_inverse
        super().__init__(generator_matrix=generator_matrix, **kwargs_copy)

        # Store the information and parity sets
        self._information_set, self._parity_set = get_information_and_parity_sets(self._dimension, self._length, information_set)

        # Register information set and parity set as buffers
        self.register_buffer("information_set", self._information_set)
        self.register_buffer("parity_set", self._parity_set)

        # Register parity submatrix as a buffer
        self.register_buffer("parity_submatrix", self._parity_submatrix)

    @property
    def parity_submatrix(self) -> torch.Tensor:
        """Get the parity submatrix P of the code.

        Returns:
            The parity submatrix P
        """
        return self._parity_submatrix

    @property
    def parity_set(self) -> torch.Tensor:
        """Get the parity set M of the code.

        Returns:
            Indices of the parity positions
        """
        return self._parity_set

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
        return self.apply_blockwise(x, self._length, projection_fn)

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
            parity_bits = torch.matmul(reshaped_x, self.parity_submatrix) % 2

            # Create output tensor of the right shape
            batch_shape = reshaped_x.shape[:-1]
            codewords = torch.zeros((*batch_shape, self._length), dtype=reshaped_x.dtype, device=reshaped_x.device)

            # Place information bits directly
            codewords[..., self.information_set] = reshaped_x

            # Place parity bits
            codewords[..., self.parity_set] = parity_bits

            return codewords

        # Use apply_blockwise to handle the encoding
        return self.apply_blockwise(x, self._dimension, systematic_encode_fn)

    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the syndrome of a received word.

        For systematic codes, the syndrome can be calculated more efficiently as:
        s = r_M + r_K * P^T (mod 2)
        where r_M are the parity bits and r_K are the information bits.

        Args:
            x: Received word tensor of shape (..., codeword_length) or (..., b*codeword_length)
               where b is a positive integer.

        Returns:
            Syndrome tensor of shape (..., redundancy) or (..., b*redundancy)

        Raises:
            ValueError: If the last dimension of the input is not a multiple of n.
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of n
        if last_dim_size % self._length != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"the code length {self._length}")

        # Define systematic syndrome calculation function to apply to blocks
        def systematic_syndrome_fn(reshaped_x):
            # Extract information bits
            r_info = reshaped_x[..., self.information_set]

            # Extract parity bits
            r_parity = reshaped_x[..., self.parity_set]

            # Calculate syndrome efficiently using the parity submatrix
            syndrome = (r_parity + torch.matmul(r_info, self.parity_submatrix.transpose(0, 1))) % 2

            return syndrome

        # Use apply_blockwise to handle the syndrome calculation
        return self.apply_blockwise(x, self._length, systematic_syndrome_fn)

        # Alternative implementation using parent class:
        # return super().calculate_syndrome(x)

    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the input tensor using systematic decoding.

        For systematic codes, decoding is more efficient because the message bits are
        already present in the information positions of the codeword. We can extract
        them directly and calculate the syndrome for error detection.

        This is more efficient than the general matrix multiplication approach used
        in the parent class.

        Args:
            x: The input tensor of shape (..., codeword_length) or (..., b*codeword_length)
               where b is a positive integer.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tuple containing:
                - Decoded tensor of shape (..., b*k). Has the same shape as the input, with the last
                  dimension reduced from b*n to b*k, where b is a positive integer.
                - Syndrome tensor for error detection of shape (..., b*r), where r is the redundancy.

        Raises:
            ValueError: If the last dimension of the input is not a multiple of n.
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of n
        if last_dim_size % self._length != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"the code length {self._length}")

        # Extract message using projection (advantage of systematic codes)
        decoded = self.project_word(x)

        # Calculate syndrome using our optimized systematic method
        syndrome = self.calculate_syndrome(x)

        return decoded, syndrome

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            String representation with key parameters
        """
        return f"{self.__class__.__name__}(" f"parity_submatrix=tensor(...), " f"information_set=tensor({self._information_set.tolist()}), " f"dimension={self._dimension}, " f"length={self._length}, " f"redundancy={self._redundancy}" f")"
