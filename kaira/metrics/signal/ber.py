"""Bit Error Rate (BER) metric.

BER is one of the most fundamental performance metrics in digital communications, providing
a measure of the reliability of the entire system :cite:`proakis2007digital` :cite:`ziemer2006principles`.
"""

from typing import Any, Optional

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

    def __init__(self, threshold: float = 0.5, name: Optional[str] = None, *args: Any, **kwargs: Any):
        """Initialize the BER metric.

        Args:
            threshold (float): Threshold for binary decision (default: 0.5)
            name (Optional[str]): Optional name for the metric
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "BER", *args, **kwargs)  # Pass args and kwargs
        self.threshold = threshold
        self.register_buffer("total_bits", torch.tensor(0))
        self.register_buffer("error_bits", torch.tensor(0))

    def forward(self, transmitted: Tensor, received: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate BER between transmitted and received bit sequences.

        Args:
            transmitted (Tensor): Original transmitted bits
            received (Tensor): Received bits
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            Tensor: BER value as a scalar tensor, or a tensor of BER values for each batch element
        """
        # Note: *args and **kwargs are not directly used here
        # but are included for interface consistency.

        if transmitted.numel() == 0 or received.numel() == 0:
            return torch.tensor(0.0)

        # Handle complex values by concatenating real and imaginary parts
        if transmitted.is_complex():
            transmitted_real = transmitted.real
            transmitted_imag = transmitted.imag
            transmitted = torch.cat([transmitted_real, transmitted_imag], dim=-1)
        if received.is_complex():
            received_real = received.real
            received_imag = received.imag
            received = torch.cat([received_real, received_imag], dim=-1)

        # Check for batch dimension
        # TODO: implement is_batched
        is_batched = False  # transmitted.dim() > 1 and transmitted.size(0) > 1 and transmitted.size(1) > 1

        # Threshold received values to get binary decisions
        transmitted_bits = (transmitted > self.threshold).bool()
        received_bits = (received > self.threshold).bool()

        # Count errors (using not equal comparison instead of XOR)
        errors = (transmitted_bits != received_bits).float()

        if is_batched:
            # Calculate error rate per batch element
            batch_errors = []
            for i in range(transmitted.size(0)):
                batch_error_sum = errors[i].sum().item()
                batch_total_bits = float(transmitted[i].numel())
                batch_error_rate = batch_error_sum / batch_total_bits if batch_total_bits > 0 else 0.0
                batch_errors.append(batch_error_rate)
            return torch.tensor(batch_errors)
        else:
            # Calculate overall error rate more precisely
            num_errors = errors.sum().item()
            total_bits = float(transmitted.numel())
            error_rate = torch.tensor(num_errors / total_bits if total_bits > 0 else 0.0)
            return error_rate

    def update(self, transmitted: Tensor, received: Tensor, *args: Any, **kwargs: Any) -> None:
        """Update the internal state with a batch of samples.

        Args:
            transmitted (Tensor): Original transmitted bits
            received (Tensor): Received bits
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).
        """
        # Note: *args and **kwargs are not directly used here
        # but are included for interface consistency.

        if transmitted.numel() == 0 or received.numel() == 0:
            return

        # Handle complex values by concatenating real and imaginary parts
        if transmitted.is_complex():
            transmitted_real = transmitted.real
            transmitted_imag = transmitted.imag
            transmitted = torch.cat([transmitted_real, transmitted_imag], dim=-1)
        if received.is_complex():
            received_real = received.real
            received_imag = received.imag
            received = torch.cat([received_real, received_imag], dim=-1)

        transmitted_bits = (transmitted > self.threshold).bool()
        received_bits = (received > self.threshold).bool()

        # Count errors using not equal comparison
        errors = (transmitted_bits != received_bits).float()

        # Update cumulative statistics
        self.total_bits += torch.prod(torch.tensor(transmitted.shape)).long()
        self.error_bits += errors.sum().long()  # Convert to long to match buffer type

    def compute(self) -> Tensor:
        """Compute the accumulated BER.

        Returns:
            Tensor: Accumulated BER
        """
        return self.error_bits.float() / max(self.total_bits, 1)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_bits.zero_()
        self.error_bits.zero_()


# Alias for backward compatibility
BER = BitErrorRate
