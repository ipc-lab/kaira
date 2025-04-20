"""Base decoders module for forward error correction.

This module implements base decoder classes for various forward error correction techniques.
Decoders are responsible for recovering the original message from received codewords that
may contain errors introduced during transmission over noisy channels.

The module provides a type-generic architecture that ensures correct pairing between encoders
and their corresponding decoders, while maintaining flexibility for different decoding
algorithms and implementation strategies.

:cite:`lin2004error`
:cite:`moon2005error`
:cite:`richardson2008modern`
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar, Union

import torch

from kaira.models.base import BaseModel

from ..encoders.base import BaseBlockCodeEncoder

T = TypeVar("T", bound=BaseBlockCodeEncoder)


class BaseBlockDecoder(BaseModel, Generic[T], ABC):
    """Base class for block code decoders.

    This abstract class provides a common interface and functionality for all types of
    block code decoders. It serves as a foundation for specific implementations like
    syndrome decoders, maximum likelihood decoders, algebraic decoders, and soft-decision
    decoders.

    The class uses a generic type parameter T to ensure type safety when pairing
    encoders with their corresponding decoders. This allows the compiler to catch
    type mismatches at development time rather than during runtime.

    Attributes:
        encoder (T): The encoder instance associated with this decoder, providing
                     access to code parameters and encoding/syndrome calculation methods

    Args:
        encoder (T): The encoder instance for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class

    Note:
        All concrete implementations must override the forward method to implement
        specific decoding algorithms. Decoders may operate on hard-decision (binary)
        or soft-decision (real-valued reliability information) inputs depending on
        their implementation.
    """

    def __init__(self, encoder: T, *args: Any, **kwargs: Any):
        """Initialize the block code decoder with an encoder instance.

        The encoder provides essential information about the code parameters and may be used by the
        decoder to perform syndrome calculations or other encoding-related operations during the
        decoding process.
        """
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    @property
    def code_length(self) -> int:
        """Get the code length (n).

        The code length is the total number of bits in each codeword,
        including both information bits and redundancy bits.

        Returns:
            The length of the code (number of bits in a codeword)
        """
        return self.encoder.code_length

    @property
    def code_dimension(self) -> int:
        """Get the code dimension (k).

        The code dimension is the number of information bits in each codeword,
        representing the actual data being transmitted.

        Returns:
            The dimension of the code (number of information bits)
        """
        return self.encoder.code_dimension

    @property
    def redundancy(self) -> int:
        """Get the code redundancy (r = n - k).

        The redundancy represents the number of parity or check bits added
        to the information bits to enable error detection and correction.

        Returns:
            The redundancy of the code (number of parity bits)
        """
        return self.encoder.redundancy

    @property
    def code_rate(self) -> float:
        """Get the code rate (k/n).

        The code rate is the ratio of information bits to the total bits,
        indicating the coding efficiency. Higher rates mean more efficient
        use of the channel but typically lower error correction capability.

        Returns:
            The rate of the code (ratio of information bits to total bits)
        """
        return self.encoder.code_rate

    @abstractmethod
    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords to recover the original messages.

        This method implements the decoding algorithm to estimate the original
        message from a potentially corrupted received codeword. Different decoder
        implementations will use different algorithms based on the code structure
        and desired performance characteristics.

        Args:
            received: Received codeword tensor with shape (..., n) or (..., m*n)
                    where n is the code length and m is some multiple.
            *args: Additional positional arguments for specific decoder implementations.
            **kwargs: Additional keyword arguments for specific decoder implementations.

        Returns:
            Either:
            - Decoded tensor containing estimated messages with shape (..., k) or (..., m*k)
            - A tuple of (decoded tensor, additional decoding information such as syndromes,
              reliability metrics, or error patterns)

        Raises:
            ValueError: If the last dimension of received is not a multiple of n.

        Note:
            The decoding may not perfectly recover the original message if the number
            of errors exceeds the error-correcting capability of the code.
        """
        raise NotImplementedError("Subclasses must implement forward method")
