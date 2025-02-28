"""Digital Binary Channel Implementations for Discrete Communications.

This module provides implementations of classic binary channels used in information theory and
digital communications. These channels model discrete errors that occur in binary transmission
systems.
"""

import torch

from kaira.utils import to_tensor

from .base import BaseChannel


class BinarySymmetricChannel(BaseChannel):
    """Binary Symmetric Channel (BSC) with symmetric bit flip probability.

    The Binary Symmetric Channel is a fundamental model in information theory where
    each bit is independently flipped with probability p. It represents a binary channel
    with symmetric crossover characteristics.

    Mathematical Model:
        P(y=0|x=1) = P(y=1|x=0) = p (crossover probability)
        P(y=0|x=0) = P(y=1|x=1) = 1-p

    Args:
        crossover_prob (float): Probability of bit flip (0 ≤ p ≤ 0.5)

    Example:
        >>> channel = BinarySymmetricChannel(crossover_prob=0.1)
        >>> x = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        >>> y = channel(x)  # Some bits will be flipped with 10% probability
    """

    def __init__(self, crossover_prob):
        super().__init__()
        if not 0 <= crossover_prob <= 1:
            raise ValueError("Crossover probability must be between 0 and 1")
        self.crossover_prob = to_tensor(crossover_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random bit flips to the input signal.

        Args:
            x (torch.Tensor): Binary input tensor with values in {0,1} or {-1,1}

        Returns:
            torch.Tensor: Binary output tensor with some bits flipped
        """
        # Check if input uses {-1,1} format
        neg_one_format = (x == -1).any()

        # Convert to {0,1} format if needed
        if neg_one_format:
            x = (x + 1) / 2

        # Generate random values for potential bit flips
        noise = torch.rand_like(x.float())

        # Apply bit flips where random values are less than crossover probability
        flips = (noise < self.crossover_prob).float()
        y = (x + flips) % 2

        # Convert back to original format if needed
        if neg_one_format:
            y = 2 * y - 1

        return y


class BinaryErasureChannel(BaseChannel):
    """Binary Erasure Channel (BEC) with symbol erasure probability.

    The Binary Erasure Channel is a channel model where bits are either received correctly
    or erased (lost) with probability p. It's commonly used to model packet loss in
    communication systems.

    Mathematical Model:
        P(y=e|x=0) = P(y=e|x=1) = p (erasure probability)
        P(y=0|x=0) = P(y=1|x=1) = 1-p (correct transmission)
        P(y=1|x=0) = P(y=0|x=1) = 0 (no bit flips)

    Args:
        erasure_prob (float): Probability of bit erasure (0 ≤ p ≤ 1)
        erasure_symbol (float): Symbol to use for erasures (default: -1)

    Example:
        >>> channel = BinaryErasureChannel(erasure_prob=0.2)
        >>> x = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        >>> y = channel(x)  # Some bits will be replaced with -1 (erasure)
    """

    def __init__(self, erasure_prob, erasure_symbol=-1):
        super().__init__()
        if not 0 <= erasure_prob <= 1:
            raise ValueError("Erasure probability must be between 0 and 1")
        self.erasure_prob = to_tensor(erasure_prob)
        self.erasure_symbol = erasure_symbol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random erasures to the input signal.

        Args:
            x (torch.Tensor): Binary input tensor with values in {0,1}

        Returns:
            torch.Tensor: Output tensor with some positions replaced by erasure_symbol
        """
        # Generate random values for deciding erasures
        erasure_mask = torch.rand_like(x.float()) < self.erasure_prob

        # Create output tensor by keeping original values where not erased
        y = x.clone().float()

        # Apply erasures
        y[erasure_mask] = self.erasure_symbol

        return y


class BinaryZChannel(BaseChannel):
    """Z Channel (asymmetric binary channel) with one-sided error probability.

    The Z Channel is an asymmetric binary channel where only one type of bit flip
    can occur: 1→0 errors happen with probability p, while 0→1 errors never occur.
    This models systems where one type of error is much more likely than the other.

    Mathematical Model:
        P(y=0|x=1) = p (1→0 error probability)
        P(y=1|x=1) = 1-p
        P(y=0|x=0) = 1 (no errors)
        P(y=1|x=0) = 0

    Args:
        error_prob (float): Probability of 1→0 bit flip (0 ≤ p ≤ 1)

    Example:
        >>> channel = BinaryZChannel(error_prob=0.3)
        >>> x = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
        >>> y = channel(x)  # Some 1s may flip to 0s, but 0s never flip to 1s
    """

    def __init__(self, error_prob):
        super().__init__()
        if not 0 <= error_prob <= 1:
            raise ValueError("Error probability must be between 0 and 1")
        self.error_prob = to_tensor(error_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one-sided (1→0) random bit flips to the input signal.

        Args:
            x (torch.Tensor): Binary input tensor with values in {0,1}

        Returns:
            torch.Tensor: Binary output tensor with some 1s flipped to 0s
        """
        # Check if input uses {-1,1} format
        neg_one_format = (x == -1).any()

        # Convert to {0,1} format if needed
        if neg_one_format:
            x = (x + 1) / 2

        # Generate mask for ones in the input
        ones_mask = x == 1

        # Create output tensor starting with the input
        y = x.clone().float()

        # For positions with 1s, generate random values and flip to 0 with probability error_prob
        if ones_mask.any():
            flip_mask = torch.rand_like(x[ones_mask]) < self.error_prob
            y[ones_mask][flip_mask] = 0

        # Convert back to original format if needed
        if neg_one_format:
            y = 2 * y - 1

        return y
