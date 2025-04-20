"""Utility functions for decoders.

This module provides common utility functions that are used across different decoder
implementations in the kaira.models.fec.decoders package.
"""

from typing import Callable

import torch


def hamming_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the Hamming distance between two binary tensors.

    Args:
        x: First binary tensor
        y: Second binary tensor

    Returns:
        Hamming distance (number of positions where the bits differ)
    """
    return torch.sum((x != y).to(torch.int), dim=-1)


def hamming_weight(x: torch.Tensor) -> torch.Tensor:
    """Calculate the Hamming weight (number of ones) in a binary tensor.

    Args:
        x: Binary tensor

    Returns:
        Hamming weight (number of ones)
    """
    return torch.sum(x, dim=-1)


def to_binary_tensor(x: int, length: int, device=None) -> torch.Tensor:
    """Convert an integer to its binary representation as a tensor.

    Args:
        x: Integer to convert
        length: Length of the binary representation
        device: Device to place the tensor on

    Returns:
        Binary tensor representation of the integer
    """
    result = torch.zeros(length, dtype=torch.int, device=device)
    for i in range(length):
        result[length - i - 1] = (x >> i) & 1
    return result


def from_binary_tensor(x: torch.Tensor) -> int:
    """Convert a binary tensor to an integer.

    Args:
        x: Binary tensor to convert

    Returns:
        Integer representation of the binary tensor
    """
    result = 0
    for i, bit in enumerate(x.flip(dims=[-1])):
        if bit:
            result |= 1 << i
    return result


def apply_blockwise(x: torch.Tensor, block_size: int, fn: Callable) -> torch.Tensor:
    """Apply a function blockwise to the last dimension of a tensor.

    This utility method is useful for processing tensors in block-sized chunks,
    which is common in block coding operations.

    Args:
        x: Input tensor with shape (..., L) where L is a multiple of block_size
        block_size: Size of each block in the last dimension
        fn: Function to apply to each block. Should accept a tensor and return
            a transformed tensor preserving the batch dimensions.

    Returns:
        Tensor with transformed blocks

    Raises:
        AssertionError: If the last dimension is not divisible by block_size
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
