"""Error Vector Magnitude (EVM) metric.

EVM is a key performance indicator used in digital communication systems to quantify the difference
between the ideal transmitted signal and the received signal. It provides a comprehensive measure
of signal quality by considering both magnitude and phase errors.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("evm")
class ErrorVectorMagnitude(BaseMetric):
    """Error Vector Magnitude (EVM) metric.

    EVM measures the difference between the ideal constellation points and the received
    constellation points, expressed as a percentage. It captures both magnitude and phase
    errors in the received signal. Lower EVM values indicate better signal quality.

    EVM is calculated as:
    EVM(%) = sqrt(E[||error_vector||^2] / E[||reference_vector||^2]) * 100

    where error_vector = received_signal - reference_signal

    Attributes:
        normalize (bool): Whether to normalize by reference signal power (default: True).
        mode (str): EVM calculation mode ('rms', 'peak', or 'percentile').
        percentile (float): Percentile value when mode is 'percentile' (default: 95.0).
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(self, normalize: bool = True, mode: str = "rms", percentile: float = 95.0, name: Optional[str] = None, *args: Any, **kwargs: Any):
        """Initialize the EVM metric.

        Args:
            normalize (bool): Whether to normalize by reference signal power (default: True).
            mode (str): EVM calculation mode ('rms', 'peak', or 'percentile').
            percentile (float): Percentile value when mode is 'percentile' (default: 95.0).
            name (Optional[str]): Optional name for the metric.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "EVM")
        self.normalize = normalize
        self.mode = mode.lower()
        self.percentile = percentile

        if self.mode not in ["rms", "peak", "percentile"]:
            raise ValueError(f"Mode must be 'rms', 'peak', or 'percentile', got '{mode}'")

        if not 0 < percentile <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the Error Vector Magnitude for the current batch.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tensor: Error Vector Magnitude as a percentage.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Handle empty tensors
        if x.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=x.device)

        # Calculate error vector
        error_vector = y - x

        # Calculate error power (squared magnitude)
        error_power = torch.abs(error_vector) ** 2

        if self.normalize:
            # Calculate reference power
            reference_power = torch.abs(x) ** 2

            # Avoid division by zero
            reference_power = torch.clamp(reference_power, min=1e-12)

            # Normalize error power by reference power
            normalized_error = error_power / reference_power
        else:
            normalized_error = error_power

        # Calculate EVM based on mode
        if self.mode == "rms":
            # RMS EVM
            evm_squared = torch.mean(normalized_error)
            evm = torch.sqrt(evm_squared)
        elif self.mode == "peak":
            # Peak EVM
            evm_squared = torch.max(normalized_error)
            evm = torch.sqrt(evm_squared)
        elif self.mode == "percentile":
            # Percentile EVM
            evm_squared = torch.quantile(normalized_error.flatten(), self.percentile / 100.0)
            evm = torch.sqrt(evm_squared)

        # Convert to percentage
        evm_percent = evm * 100.0

        return evm_percent

    def calculate_per_symbol_evm(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate EVM for each symbol separately.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.

        Returns:
            Tensor: Per-symbol EVM values as percentages.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Handle empty tensors
        if x.numel() == 0:
            return torch.tensor([], dtype=torch.float32, device=x.device)

        # Calculate error vector
        error_vector = y - x

        # Calculate per-symbol error magnitude
        error_magnitude = torch.abs(error_vector)

        if self.normalize:
            # Calculate per-symbol reference magnitude
            reference_magnitude = torch.abs(x)
            reference_magnitude = torch.clamp(reference_magnitude, min=1e-12)

            # Normalize by reference magnitude
            per_symbol_evm = error_magnitude / reference_magnitude
        else:
            per_symbol_evm = error_magnitude

        # Convert to percentage
        per_symbol_evm_percent = per_symbol_evm * 100.0

        return per_symbol_evm_percent

    def calculate_statistics(self, x: Tensor, y: Tensor) -> dict:
        """Calculate comprehensive EVM statistics.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.

        Returns:
            dict: Dictionary containing various EVM statistics.
        """
        # Calculate per-symbol EVM
        per_symbol_evm = self.calculate_per_symbol_evm(x, y)

        # Calculate various statistics
        stats_dict = {
            "evm_rms": self.forward(x, y),
            "evm_mean": torch.mean(per_symbol_evm),
            "evm_std": torch.std(per_symbol_evm),
            "evm_min": torch.min(per_symbol_evm),
            "evm_max": torch.max(per_symbol_evm),
            "evm_median": torch.median(per_symbol_evm),
            "evm_95th": torch.quantile(per_symbol_evm.flatten(), 0.95),
            "evm_99th": torch.quantile(per_symbol_evm.flatten(), 0.99),
            "evm_per_symbol": per_symbol_evm,
        }

        return stats_dict


# Alias for backward compatibility
EVM = ErrorVectorMagnitude
