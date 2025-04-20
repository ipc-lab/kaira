"""Base decoders module for forward error correction.

This module implements base decoder classes for various forward error correction techniques.
Decoders are responsible for recovering the original message from received codewords that
may contain errors introduced during transmission over noisy channels.

The implementation provides base classes for various types of decoders, following
standard conventions in coding theory :cite:`lin2004error,moon2005error,richardson2008modern`.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar, Union

import torch

from kaira.models.base import BaseModel

from ..encoders.block_code import BlockCodeEncoder

T = TypeVar("T", bound=BlockCodeEncoder)


class BlockDecoder(BaseModel, Generic[T], ABC):
    """Base class for block code decoders.

    This abstract class provides a common interface and functionality for all types of
    block code decoders. It serves as a foundation for specific implementations like
    syndrome decoders, Viterbi decoders, BCJR decoders, etc.

    Attributes:
        encoder (T): The encoder instance associated with this decoder

    Args:
        encoder (T): The encoder instance for the code being decoded
        *args: Variable positional arguments passed to the base class
        **kwargs: Variable keyword arguments passed to the base class
    """

    def __init__(self, encoder: T, *args: Any, **kwargs: Any):
        """Initialize the block code decoder."""
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    @property
    def code_length(self) -> int:
        """Get the code length (n).

        Returns:
            The length of the code (number of bits in a codeword)
        """
        return self.encoder.code_length

    @property
    def code_dimension(self) -> int:
        """Get the code dimension (k).

        Returns:
            The dimension of the code (number of information bits)
        """
        return self.encoder.code_dimension

    @property
    def redundancy(self) -> int:
        """Get the code redundancy (r = n - k).

        Returns:
            The redundancy of the code (number of parity bits)
        """
        return self.encoder.redundancy

    @property
    def code_rate(self) -> float:
        """Get the code rate (k/n).

        Returns:
            The rate of the code (ratio of information bits to total bits)
        """
        return self.encoder.code_rate

    @abstractmethod
    def forward(self, received: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode received codewords to recover the original messages.

        Args:
            received: Received codeword tensor. The last dimension should be
                    a multiple of the code length (n).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Either:
            - Decoded tensor containing estimated messages
            - A tuple of (decoded tensor, additional decoding information)

        Raises:
            ValueError: If the last dimension of received is not a multiple of n.
        """
        raise NotImplementedError("Subclasses must implement forward method")
