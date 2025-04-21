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

from typing import Callable

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
        Tensor with transformed blocks

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

    # Flatten the result back to original structure
    return result.view(*leading_dims, -1)
