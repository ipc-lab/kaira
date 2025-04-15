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

    Attributes:
        threshold (float): Threshold for binary decision (default: 0.5).
        total_bits (Tensor): Accumulated total number of bits processed.
        error_bits (Tensor): Accumulated number of bit errors.
    """

    is_differentiable = False
    higher_is_better = False

    def __init__(self, threshold: float = 0.5, name: Optional[str] = None, *args: Any, **kwargs: Any):
        """Initialize the BER metric.

        Args:
            threshold (float): Threshold for binary decision (default: 0.5)
            name (Optional[str]): Optional name for the metric
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "BER")  # Pass only name
        self.threshold = threshold
        self.register_buffer("total_bits", torch.tensor(0, dtype=torch.long))
        self.register_buffer("error_bits", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the Bit Error Rate for the current batch.

        Args:
            x (Tensor): The transmitted/original tensor.
            y (Tensor): The received/predicted tensor.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tensor: Bit error rate for the batch.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Handle complex inputs by comparing real and imaginary parts separately
        if x.is_complex() or y.is_complex():
            if not x.is_complex() or not y.is_complex():
                raise ValueError("Both inputs must be complex if one is complex.")
            # Treat real and imaginary parts as separate bits
            x_real = (x.real > self.threshold).bool()
            x_imag = (x.imag > self.threshold).bool()
            y_real = (y.real > self.threshold).bool()
            y_imag = (y.imag > self.threshold).bool()

            errors_real = (x_real != y_real).float()
            errors_imag = (x_imag != y_imag).float()

            num_errors = errors_real.sum().item() + errors_imag.sum().item()
            total_bits = float(x.numel() * 2)  # Each complex number represents 2 "bits" (real/imag)
        else:
            # Threshold received values to get binary decisions
            x_bits = (x > self.threshold).bool()
            y_bits = (y > self.threshold).bool()

            # Count errors
            errors = (x_bits != y_bits).float()
            num_errors = errors.sum().item()
            total_bits = float(x.numel())

        error_rate = torch.tensor(num_errors / total_bits if total_bits > 0 else 0.0)
        return error_rate

    def update(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> None:
        """Update accumulated statistics with results from a new batch.

        Args:
            x (Tensor): The transmitted/original tensor for the current batch.
            y (Tensor): The received/predicted tensor for the current batch.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Handle complex inputs
        if x.is_complex() or y.is_complex():
            if not x.is_complex() or not y.is_complex():
                raise ValueError("Both inputs must be complex if one is complex for update.")
            x_real = (x.real > self.threshold).bool()
            x_imag = (x.imag > self.threshold).bool()
            y_real = (y.real > self.threshold).bool()
            y_imag = (y.imag > self.threshold).bool()

            errors_real = (x_real != y_real).float()
            errors_imag = (x_imag != y_imag).float()

            batch_errors = errors_real.sum().long() + errors_imag.sum().long()
            batch_bits = x.numel() * 2
        else:
            x_bits = (x > self.threshold).bool()
            y_bits = (y > self.threshold).bool()
            errors = (x_bits != y_bits).float()
            batch_errors = errors.sum().long()
            batch_bits = x.numel()

        # Update cumulative statistics
        self.total_bits += batch_bits
        self.error_bits += batch_errors

    def compute(self) -> Tensor:
        """Compute the accumulated BER.

        Returns:
            Tensor: Accumulated BER
        """
        # Ensure float division
        return self.error_bits.float() / max(self.total_bits.item(), 1)

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.total_bits.zero_()
        self.error_bits.zero_()


# Alias for backward compatibility
BER = BitErrorRate
