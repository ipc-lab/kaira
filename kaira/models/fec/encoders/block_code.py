"""Block code module for forward error correction.

This module implements block coding for digital communications, which is a fundamental
error correction technique where a message is encoded into a codeword by adding redundancy.
Block codes provide systematic approaches to detect and correct errors that might occur
during transmission over noisy channels.

The implementation provides base classes for various types of block codes, following
standard conventions in coding theory :cite:`lin2004error,moon2005error,richardson2008modern`.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union

import torch

from kaira.models.base import BaseModel


class BlockCodeEncoder(BaseModel, ABC):
    """Base class for block code encoders.

    This abstract class provides a common interface and functionality for all types of
    block code encoders. It serves as a foundation for specific implementations like
    linear block codes, cyclic codes, BCH codes, etc.

    Block codes transform k information bits into n coded bits (n > k), providing
    error detection and correction capabilities :cite:`lin2004error,moon2005error`.

    Attributes:
        code_length (int): The length of the codeword (n)
        code_dimension (int): The dimension of the code (k)

    Args:
        code_length (int): The length of the codeword (n)
        code_dimension (int): The dimension of the code (k)
        *args: Variable positional arguments passed to the base class.
        **kwargs: Variable keyword arguments passed to the base class.
    """

    def __init__(self, code_length: int, code_dimension: int, *args: Any, **kwargs: Any):
        """Initialize the block code encoder."""
        super().__init__(*args, **kwargs)

        if code_length <= 0:
            raise ValueError(f"Code length must be positive, got {code_length}")
        if code_dimension <= 0:
            raise ValueError(f"Code dimension must be positive, got {code_dimension}")
        if code_dimension > code_length:
            raise ValueError(f"Code dimension ({code_dimension}) must not exceed code length ({code_length})")

        self._length = code_length
        self._dimension = code_dimension
        self._redundancy = code_length - code_dimension

    @property
    def code_length(self) -> int:
        """Get the code length (n).

        Returns:
            The length of the code (number of bits in a codeword)
        """
        return self._length

    @property
    def code_dimension(self) -> int:
        """Get the code dimension (k).

        Returns:
            The dimension of the code (number of information bits)
        """
        return self._dimension

    @property
    def redundancy(self) -> int:
        """Get the code redundancy (r = n - k).

        Returns:
            The redundancy of the code (number of parity bits)
        """
        return self._redundancy

    @property
    def parity_bits(self) -> int:
        """Get the number of parity bits (r = n - k).

        Returns:
            The number of parity bits in the code
        """
        return self._redundancy

    @property
    def code_rate(self) -> float:
        """Get the code rate (k/n).

        Returns:
            The rate of the code (ratio of information bits to total bits)
        """
        return self._dimension / self._length

    @staticmethod
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

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the encoding operation to the input tensor.

        Args:
            x: Input tensor containing message bits. The last dimension should be
               a multiple of the code dimension (k).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Encoded tensor with codewords. Has the same shape as the input except
            the last dimension is expanded by a factor of n/k.

        Raises:
            ValueError: If the last dimension of x is not a multiple of k.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @abstractmethod
    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode a received codeword back to a message.

        This method is the inverse of the encoding process (though not all errors may be correctable).

        Args:
            x: Input tensor containing received codewords. The last dimension should be
               a multiple of the code length (n).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Either:
            - Decoded tensor containing messages
            - A tuple of (decoded tensor, syndrome or error information)

        Raises:
            ValueError: If the last dimension of x is not a multiple of n.
        """
        raise NotImplementedError("Subclasses must implement inverse_encode method")

    @abstractmethod
    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the syndrome of a received codeword.

        The syndrome provides information about errors that may have occurred during
        transmission. A zero syndrome indicates that the received word is a valid codeword,
        though not necessarily the transmitted one.

        Args:
            x: Received codeword tensor

        Returns:
            Syndrome tensor
        """
        raise NotImplementedError("Subclasses must implement calculate_syndrome method")
