"""Bit Error Rate (BER) metric.

BER is one of the most fundamental performance metrics in digital communications, providing
a measure of the reliability of the entire system :cite:`proakis2007digital` :cite:`ziemer2006principles`.
"""

from typing import Optional

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("ber")
class BitErrorRate(BaseMetric):
    """Bit Error Rate (BER) metric.

    BER measures the number of bit errors divided by the total number of bits transmitted. Lower
    values indicate better performance. BER is one of the most common figure of merit used to assess systems
    that transmit digital data from one location to another :cite:`proakis2007digital` and serves
    as the cornerstone for performance evaluation in communications :cite:`barry2003digital`.
    """

    def __init__(self, threshold: float = 0.5, name: Optional[str] = None):
        """Initialize the BER metric.

        Args:
            threshold (float): Threshold for binary decision (default: 0.5)
            name (Optional[str]): Optional name for the metric
        """
        super().__init__(name=name or "BER")
        self.threshold = threshold
        self.register_buffer("total_bits", torch.tensor(0))
        self.register_buffer("error_bits", torch.tensor(0))

    def forward(self, transmitted: Tensor, received: Tensor) -> Tensor:
        """Calculate BER between transmitted and received bit sequences.

        Args:
            transmitted (Tensor): Original transmitted bits
            received (Tensor): Received bits

        Returns:
            Tensor: BER values
        """
        # Threshold received values to get binary decisions
        transmitted_bits = (transmitted > self.threshold).float()
        received_bits = (received > self.threshold).float()

        # Count errors (XOR will be 1 where bits differ)
        errors = (transmitted_bits ^ received_bits).float()

        # Calculate error rates per sample
        error_rate = errors.mean(dim=-1)

        return error_rate

    def update(self, transmitted: Tensor, received: Tensor) -> None:
        """Update the internal state with a batch of samples.

        Args:
            transmitted (Tensor): Original transmitted bits
            received (Tensor): Received bits
        """
        transmitted_bits = (transmitted > self.threshold).float()
        received_bits = (received > self.threshold).float()

        # Count errors
        errors = (transmitted_bits ^ received_bits).float()

        # Update cumulative statistics
        self.total_bits += torch.prod(torch.tensor(transmitted.shape))
        self.error_bits += errors.sum()

    def compute(self) -> Tensor:
        """Compute the accumulated BER.

        Returns:
            Tensor: Accumulated BER
        """
        return self.error_bits / self.total_bits if self.total_bits > 0 else torch.tensor(0.0)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_bits.zero_()
        self.error_bits.zero_()


# Alias for backward compatibility
BER = BitErrorRate
