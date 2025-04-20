"""Linear block coding module for forward error correction.

This module implements linear block coding for binary data transmission, a fundamental
error correction technique where a message is encoded into a code word using generator
and check matrices. Linear block codes provide a systematic approach to adding redundancy
for error correction :cite:`lin2004error,moon2005error`.

The implementation follows common conventions in coding theory with particular focus
on binary linear block codes, which are characterized by generator and check matrices
whose elements belong to the binary field GF(2) :cite:`richardson2008modern`.
"""

from typing import Any, Tuple

import torch

from kaira.models.registry import ModelRegistry

from .block_code import BlockCodeEncoder


def compute_null_space_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the null space matrix of the input matrix.

    Args:
        matrix: Input matrix

    Returns:
        Matrix whose rows form a basis for the null space of the input matrix
    """
    # Use SVD to compute the null space
    U, S, V = torch.linalg.svd(matrix, full_matrices=True)

    # Count non-zero singular values with small tolerance
    tol = S.max() * max(matrix.size()) * torch.finfo(matrix.dtype).eps
    rank = torch.sum(S > tol).item()

    # Extract the null space from the right singular vectors
    if rank < V.size(1):
        null_space = V[rank:].T
        # Convert to binary
        return (null_space.abs() > 0.5).type(matrix.dtype)
    else:
        # No null space, return empty matrix
        return torch.zeros((0, matrix.size(1)), dtype=matrix.dtype)


def compute_reduced_row_echelon_form(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the reduced row echelon form of the matrix.

    Args:
        matrix: Input matrix

    Returns:
        Reduced row echelon form of the matrix
    """
    try:
        # Try to use PyTorch's built-in RREF function if available
        rref_matrix, _ = torch.linalg.matrix_rank(matrix, method="complete")

        # Convert to binary
        return (rref_matrix.abs() > 0.5).type(matrix.dtype)

    except (AttributeError, RuntimeError):
        # Fall back to manual implementation if torch.linalg.matrix_rank with method='complete' is not available
        # Make a copy to avoid modifying the original
        A = matrix.clone()
        rows, cols = A.size()

        # Initialize pivot position
        pivot_row = 0

        # Process each column
        for col in range(cols):
            # Find pivot row
            max_idx = torch.argmax(torch.abs(A[pivot_row:, col])) + pivot_row if pivot_row < rows else -1

            # If no pivot found in this column, move to the next column
            if max_idx == -1 or A[max_idx, col].abs() < 1e-10:
                continue

            # Swap rows to bring pivot to the top
            if max_idx != pivot_row:
                A[pivot_row], A[max_idx] = A[max_idx].clone(), A[pivot_row].clone()

            # Scale pivot row to have a 1 at the pivot
            pivot_val = A[pivot_row, col]
            A[pivot_row] = A[pivot_row] / pivot_val

            # Eliminate the pivot column from all other rows
            for i in range(rows):
                if i != pivot_row:
                    factor = A[i, col]
                    A[i] = A[i] - factor * A[pivot_row]

            pivot_row += 1
            if pivot_row == rows:
                break

        # Convert to binary
        return (A.abs() > 0.5).type(matrix.dtype)


def compute_right_pseudo_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the right pseudo-inverse of a matrix.

    For a generator matrix G, the right pseudo-inverse G_right_inv satisfies G * G_right_inv = I

    Args:
        matrix: Input matrix

    Returns:
        Right pseudo-inverse of the matrix
    """
    # Use built-in torch.linalg.pinv function which computes the Moore-Penrose pseudo-inverse
    # This is more numerically stable and optimized than manual SVD computation
    pseudo_inv = torch.linalg.pinv(matrix)

    # Convert to binary
    return (pseudo_inv.abs() > 0.5).type(matrix.dtype)


@ModelRegistry.register_model("linear_block_code_encoder")
class LinearBlockCodeEncoder(BlockCodeEncoder):
    """Encoder for linear block coding.

    This encoder transforms binary input messages into codewords according to
    the specified generator matrix. It serves as the encoding component of
    a linear block code system.

    The encoder applies the formula: c = mG, where:
    - c is the codeword
    - m is the message
    - G is the generator matrix

    This implementation follows the standard approach to linear block coding described in the
    error control coding literature :cite:`lin2004error,moon2005error,sklar2001digital`.

    Attributes:
        generator_matrix (torch.Tensor): The generator matrix G of the code
        generator_right_inverse (torch.Tensor): The right pseudo-inverse of the generator matrix
        check_matrix (torch.Tensor): The parity check matrix H

    Args:
        generator_matrix (torch.Tensor): The generator matrix for encoding.
            Must be a binary matrix of shape (k, n) where k is the message length
            and n is the codeword length.
        *args: Variable positional arguments passed to the base class.
        **kwargs: Variable keyword arguments passed to the base class.
    """

    def __init__(self, generator_matrix: torch.Tensor, *args: Any, **kwargs: Any):
        """Initialize the linear block encoder.

        Args:
            generator_matrix (torch.Tensor): The generator matrix for encoding.
                Must be a binary matrix of shape (k, n) where k is the message length
                and n is the codeword length.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Ensure generator matrix is a torch tensor
        if not isinstance(generator_matrix, torch.Tensor):
            generator_matrix = torch.tensor(generator_matrix)

        # Extract dimensions from generator matrix
        dimension, length = generator_matrix.size()

        # Initialize the base class with dimensions
        super().__init__(code_length=length, code_dimension=dimension)

        # Register buffer for the generator matrix
        self.register_buffer("generator_matrix", generator_matrix)

        # Create generator matrix right inverse for decoding
        self._generator_right_inverse = compute_right_pseudo_inverse(generator_matrix)

        # Register buffer for the generator right inverse
        self.register_buffer("generator_right_inverse", self._generator_right_inverse)

        # Compute check matrix for syndrome calculation
        self._check_matrix = compute_null_space_matrix(generator_matrix)

        # Register buffer for the check matrix
        self.register_buffer("check_matrix", self._check_matrix)

    @property
    def parity_check_matrix(self) -> torch.Tensor:
        """Get the check matrix H of the code.

        The check matrix H satisfies the property: GH^T = 0

        Returns:
            The check matrix H of the code
        """
        return self.check_matrix

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Applies the encoding mapping Enc: B^k → B^n of the code.

        This method takes one or more sequences of messages and returns their
        corresponding codeword sequences. The encoding process follows standard linear
        block code principles :cite:`lin2004error,richardson2008modern`.

        Args:
            x: The input tensor. Can be either a single sequence whose length is a multiple of k,
               or a multidimensional tensor where the last dimension is a multiple of k.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The output tensor. Has the same shape as the input, with the last dimension
            expanded from b*k to b*n, where b is a positive integer.

        Raises:
            ValueError: If the last dimension of the input is not a multiple of k.
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of k
        if last_dim_size % self.code_dimension != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of the code dimension {self.code_dimension}")

        # Define encoding function to apply to blocks
        def encode_fn(reshaped_x):
            # Apply matrix multiplication to the last dimension
            return torch.matmul(reshaped_x, self.generator_matrix) % 2

        # Use apply_blockwise to handle the encoding
        return self.apply_blockwise(x, self.code_dimension, encode_fn)

    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the syndrome of a received word.

        The syndrome is computed as s = xH^T and is used to detect errors.
        A non-zero syndrome indicates the presence of errors :cite:`lin2004error,moon2005error`.
        This approach is a fundamental technique in error detection and correction
        for linear block codes :cite:`sklar2001digital`.

        Args:
            x: Received word tensor of shape (..., codeword_length) or (..., b*codeword_length)
               where b is a positive integer.

        Returns:
            Syndrome tensor of shape (..., redundancy) or (..., b*redundancy)
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of n
        if last_dim_size % self.code_length != 0:
            raise ValueError(f"Input codeword length {last_dim_size} must be a multiple of the code length {self.code_length}")

        # Define syndrome calculation function to apply to blocks
        def syndrome_fn(reshaped_x):
            # Apply matrix multiplication with check matrix transposed
            return torch.matmul(reshaped_x, self.check_matrix.transpose(0, 1)) % 2

        # Use apply_blockwise to handle the syndrome calculation
        return self.apply_blockwise(x, self.code_length, syndrome_fn)

    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the input tensor using the generator matrix right inverse.

        This method takes one or more sequences of codewords and returns their
        corresponding decoded messages along with syndromes. The decoding approach
        follows standard techniques in error control coding literature :cite:`lin2004error,sklar2001digital`.

        Args:
            x: The input tensor. Can be either a single sequence whose length is a multiple of n,
               or a multidimensional tensor where the last dimension is a multiple of n.
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
        if last_dim_size % self.code_length != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of the code length {self.code_length}")

        # Calculate syndrome using the calculate_syndrome method which already uses apply_blockwise
        syndrome = self.calculate_syndrome(x)

        # Define decoding function to apply to blocks
        def decode_fn(reshaped_x):
            # Apply matrix multiplication with generator right inverse
            return torch.matmul(reshaped_x, self.generator_right_inverse) % 2

        # Use apply_blockwise to handle the decoding
        decoded = self.apply_blockwise(x, self.code_length, decode_fn)

        return decoded, syndrome
