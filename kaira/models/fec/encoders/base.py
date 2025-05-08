"""Block code module for forward error correction.

This module implements block coding for digital communications, which is a fundamental
error correction technique where a message is encoded into a codeword by adding redundancy.
Block codes provide systematic approaches to detect and correct errors that might occur
during transmission over noisy channels.

The implementation follows standard conventions in coding theory and provides base classes
for various types of block codes with well-defined interfaces for encoding, decoding,
and error detection processes.

    :cite:`lin2004error`
    :cite:`moon2005error`
    :cite:`richardson2008modern`
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch

from kaira.models.base import BaseModel


class BaseBlockCodeEncoder(BaseModel, ABC):
    """Base class for block code encoders.

    This abstract class provides a common interface and functionality for all types of
    block code encoders. It serves as a foundation for specific implementations like
    linear block codes, cyclic codes, BCH codes, etc.

    Block codes transform k information bits into n coded bits (n > k), providing
    error detection and correction capabilities. The redundancy added during encoding
    enables the receiver to detect and possibly correct errors introduced by the channel.


    Args:
        code_length (int): The length of the codeword (n)
        code_dimension (int): The dimension of the code (k)
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Raises:
        ValueError: If code parameters are invalid (e.g., non-positive or dimension > length)

    Note:
        All concrete implementations must override the forward, inverse_encode, and
        calculate_syndrome methods to provide specific encoding and decoding behavior.
    """

    def __init__(self, code_length: int, code_dimension: int, *args: Any, **kwargs: Any):
        """Initialize the block code encoder with specified parameters.

        Sets up the basic code parameters and validates that they meet the requirements for a valid
        block code (positive length, positive dimension, dimension <= length).
        """
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
        self.device = kwargs.get("device", "cpu")

    @property
    def code_length(self) -> int:
        """Get the codeword length (n).

        Returns:
            The number of bits in each codeword after encoding
        """
        return self._length

    @property
    def code_dimension(self) -> int:
        """Get the code dimension (k).

        Returns:
            The number of information bits encoded in each codeword
        """
        return self._dimension

    @property
    def redundancy(self) -> int:
        """Get the code redundancy (r = n - k).

        Returns:
            The number of redundant bits added during encoding
        """
        return self._redundancy

    @property
    def parity_bits(self) -> int:
        """Get the number of parity bits (synonym for redundancy).

        Returns:
            The number of parity/check bits in each codeword
        """
        return self._redundancy

    @property
    def code_rate(self) -> float:
        """Get the rate of the code (k/n).

        The code rate is a measure of efficiency, representing the proportion
        of the total bits that carry information (as opposed to redundancy).

        Returns:
            The ratio of information bits to total bits (between 0 and 1)
        """
        return self._dimension / self._length

    def extract_message(self, codeword: torch.Tensor) -> torch.Tensor:
        """Extract the message bits from a codeword.

        By default, this calls inverse_encode and returns just the decoded message.
        Subclasses can override this method to provide more efficient implementations.

        Args:
            codeword: Codeword tensor with shape (..., n) where n is the code length

        Returns:
            Extracted message tensor with shape (..., k) where k is the code dimension

        Note:
            This implementation assumes the inverse_encode method can handle a single
            codeword correctly. Specific code types may override this with more
            efficient implementations.
        """
        # Use inverse_encode and discard the syndrome information if returned
        result = self.inverse_encode(codeword)
        if isinstance(result, tuple):
            return result[0]
        return result

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the encoding operation to the input tensor.

        Transforms message bits into codewords by adding redundancy according to
        the specific encoding scheme implemented by the subclass.

        Args:
            x: Input tensor containing message bits. The last dimension should be
               a multiple of the code dimension (k).
            *args: Additional positional arguments for specific encoder implementations.
            **kwargs: Additional keyword arguments for specific encoder implementations.

        Returns:
            Encoded tensor with codewords. Has the same shape as the input except
            the last dimension is expanded by a factor of n/k.

        Raises:
            ValueError: If the last dimension of x is not a multiple of k.

        Note:
            The specific encoding method depends on the subclass implementation.
            For example, linear codes use matrix multiplication, while other codes
            may use different algorithms.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @abstractmethod
    def inverse_encode(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode a received codeword back to a message.

        This method is the inverse of the encoding process, attempting to recover
        the original message from a potentially corrupted codeword. The ability to
        correct errors depends on the specific code properties and the error pattern.

        Args:
            x: Input tensor containing received codewords. The last dimension should be
               a multiple of the code length (n).
            *args: Additional positional arguments for specific decoder implementations.
            **kwargs: Additional keyword arguments for specific decoder implementations.

        Returns:
            Either:
            - Decoded tensor containing messages, or
            - A tuple of (decoded tensor, syndrome or error information)

            The return type depends on the specific implementation and parameters.

        Raises:
            ValueError: If the last dimension of x is not a multiple of n.

        Note:
            This method may not recover the original message perfectly if the number
            of errors exceeds the error-correcting capability of the code.
        """
        raise NotImplementedError("Subclasses must implement inverse_encode method")

    @abstractmethod
    def calculate_syndrome(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the syndrome of a received codeword.

        The syndrome provides information about errors that may have occurred during
        transmission. A zero syndrome indicates that the received word is a valid codeword
        (though not necessarily the transmitted one if the number of errors is large).

        The syndrome pattern helps identify the error pattern and is used in the
        decoding process to correct errors.

        Args:
            x: Received codeword tensor with shape (..., n) or (..., m*n)
               where n is the code length and m is some multiple.

        Returns:
            Syndrome tensor that characterizes the error pattern (if any).
            A zero syndrome indicates a valid codeword.

        Note:
            For linear codes, the syndrome is computed as H·r where H is the parity-check
            matrix and r is the received codeword vector.
        """
        raise NotImplementedError("Subclasses must implement calculate_syndrome method")
