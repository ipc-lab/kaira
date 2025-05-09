"""Utility functions for forward error correction.

This module provides common utility functions used across different encoder and decoder
implementations in the kaira.models.fec package. These utilities handle binary data manipulation,
distance calculations, and tensor processing operations that are fundamental to error correction coding.

Functions:
    hamming_distance: Calculate bit differences between binary tensors
    hamming_weight: Count number of 1s in binary tensors
    to_binary_tensor: Convert integers to binary tensor representation
    from_binary_tensor: Convert binary tensors back to integers
    apply_blockwise: Process tensor data in blocks of specified size

These functions are optimized for PyTorch operations and support both CPU and GPU computation.

    :cite:`moon2005error`
"""

from typing import Callable, Optional

import torch


def hamming_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the Hamming distance between two binary tensors.

    The Hamming distance is the number of positions where corresponding elements differ,
    which is a fundamental metric in error correction coding to quantify errors.

    Args:
        x: First binary tensor of shape (..., N)
        y: Second binary tensor of shape (..., N)

    Returns:
        Tensor containing Hamming distances along the last dimension

    Examples:
        >>> import torch
        >>> x = torch.tensor([1, 0, 1, 0])
        >>> y = torch.tensor([1, 1, 0, 0])
        >>> hamming_distance(x, y)
        tensor(2)
    """
    return torch.sum((x != y).to(torch.int), dim=-1)


def hamming_weight(x: torch.Tensor) -> torch.Tensor:
    """Calculate the Hamming weight (number of ones) in a binary tensor.

    The Hamming weight is the number of non-zero elements, which is useful for
    determining the number of 1s in a codeword or error pattern.

    Args:
        x: Binary tensor of shape (..., N)

    Returns:
        Tensor containing Hamming weights along the last dimension

    Examples:
        >>> import torch
        >>> x = torch.tensor([1, 0, 1, 1, 0])
        >>> hamming_weight(x)
        tensor(3)
    """
    return torch.sum(x, dim=-1)


def to_binary_tensor(x: int, length: int, device=None, dtype=torch.int) -> torch.Tensor:
    """Convert an integer to its binary representation as a tensor.

    Supports custom device and dtype, and handles negative values by using absolute.

    This utility is useful for converting numerical values to their binary form
    for processing with binary error correction codes.

    Args:
        x: Integer to convert
        length: Length of the binary representation (padded with leading zeros if needed)
        device: Device to place the tensor on (CPU or GPU)
        dtype: Data type of the resulting tensor

    Returns:
        Binary tensor representation of the integer with shape (length,)

    Examples:
        >>> to_binary_tensor(10, 6)  # Decimal 10 = Binary 001010
        tensor([0, 0, 1, 0, 1, 0])
    """
    x_abs = abs(x)
    result = torch.zeros(length, dtype=dtype, device=device)
    for i in range(length):
        result[length - i - 1] = (x_abs >> i) & 1
    return result


def from_binary_tensor(x: torch.Tensor) -> int:
    """Convert a binary tensor to an integer.

    This is the inverse operation of to_binary_tensor, converting a binary
    representation back to its integer value.

    Args:
        x: Binary tensor to convert, with shape (...) where the last dimension
           represents the binary digits

    Returns:
        Integer representation of the binary tensor

    Examples:
        >>> x = torch.tensor([0, 0, 1, 0, 1, 0])  # Binary 001010
        >>> from_binary_tensor(x)
        10
    """
    result = 0
    for i, bit in enumerate(x.flip(dims=[-1])):
        if bit:
            result |= 1 << i
    return result


def apply_blockwise(x: torch.Tensor, block_size: int, fn: Callable) -> torch.Tensor:
    """Apply a function blockwise to the last dimension of a tensor.

    This utility is essential for block coding operations where data needs to be
    processed in fixed-size chunks, such as in systematic codes or interleaved coding.

    Args:
        x: Input tensor with shape (..., L) where L is a multiple of block_size
        block_size: Size of each block in the last dimension
        fn: Function to apply to each block. Should accept a tensor and return
            a transformed tensor preserving the batch dimensions

    Returns:
        Tensor with transformed blocks or tuple of tensors if fn returns a tuple

    Raises:
        AssertionError: If the last dimension is not divisible by block_size

    Examples:
        >>> x = torch.tensor([1, 0, 1, 0, 1, 1])
        >>> # Apply NOT operation to each block of size 2
        >>> apply_blockwise(x, 2, lambda b: 1 - b)
        tensor([0, 1, 0, 1, 0, 0])
    """
    *leading_dims, L = x.shape
    assert L % block_size == 0, f"Last dimension ({L}) must be divisible by block_size ({block_size})"

    # Reshape to expose blocks: (..., L) -> (..., L//block_size, block_size)
    new_shape = (*leading_dims, L // block_size, block_size)
    x_reshaped = x.view(*new_shape)

    # Apply function along the last dimension (block)
    result = fn(x_reshaped)

    # Check if the result is a tuple (like when return_errors=True)
    if isinstance(result, tuple):
        # Process each part of the tuple independently
        processed_results = []
        for res_part in result:
            # Flatten each part back to original structure
            processed_results.append(res_part.view(*leading_dims, -1))
        return tuple(processed_results)
    else:
        # Flatten the result back to original structure
        return result.view(*leading_dims, -1)


def Taylor_arctanh(vector: torch.Tensor, num_series: int = 105):
    """Approximate the inverse hyperbolic tangent (arctanh) using a Taylor series expansion.

    Args:
        vector (torch.Tensor): Input tensor for which the arctanh is to be approximated.
        num_series (int): Number of terms in the Taylor series to include (default: 105).

    Returns:
        torch.Tensor: Tensor containing the approximated arctanh values for the input.

    Notes:
        The Taylor series for arctanh(x) is given by:
        arctanh(x) = x + (x^3)/3 + (x^5)/5 + (x^7)/7 + ...
        This function computes the series up to the specified number of terms.
    """
    ans = vector
    for i in range(1, num_series):
        ans = ans + 1 / (2 * i + 1) * torch.pow(vector, 2 * i + 1)
    return ans


def sign_to_bin(x: torch.Tensor) -> torch.Tensor:
    """Convert sign values (-1/+1) to binary values (1/0).

    This function maps values from the sign domain to the binary domain:
    - Sign +1 maps to binary 0
    - Sign -1 maps to binary 1

    It's commonly used in soft-decision decoding to convert LLR sign information
    to binary codeword bits.

    Args:
        x: Input tensor with sign values (-1/+1)

    Returns:
        Tensor with binary values (0/1)
    """
    return 0.5 * (1 - x)


def row_reduction(matrix: torch.Tensor, num_cols: Optional[int] = None):
    """Perform row reduction on a binary matrix using PyTorch.

    Args:
        matrix: Binary matrix of shape (m, n) to be row reduced, m <= n.
        num_cols: Number of columns to consider for row reduction. Defaults to all columns.

    Returns:
        Tuple containing:
            - Row-reduced matrix.
            - Rank of the matrix (number of pivot rows).
    """
    device = matrix.device
    if num_cols is None:
        num_cols = matrix.shape[1]
    matrix_row_reduced = matrix.clone()  # Create a copy to avoid modifying the input
    p = 0  # Pivot row index
    for j in range(num_cols):
        # Find the first non-zero element in the current column starting from row p
        idxs = p + torch.nonzero(matrix_row_reduced[p:, j], as_tuple=False).view(-1).to(device)  # .squeeze().to(device)

        if idxs.numel() == 0:  # If no non-zero element is found, continue to the next column
            continue

        # Swap the current row with the row containing the first non-zero element
        idxs = idxs[0]
        matrix_row_reduced[[p, idxs], :] = matrix_row_reduced[[idxs, p], :]

        # Perform row reduction on all other rows in the current column
        non_pivot_idxs = torch.nonzero(matrix_row_reduced[:, j], as_tuple=False).view(-1).tolist()
        non_pivot_idxs.remove(p)
        matrix_row_reduced[non_pivot_idxs, :] ^= matrix_row_reduced[p, :]

        p += 1  # Move to the next pivot row
        if p == matrix_row_reduced.shape[0]:  # Stop if all rows have been processed
            break

    return matrix_row_reduced, p
