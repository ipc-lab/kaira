"""Error Vector Magnitude (EVM) metric.

EVM is a key performance indicator used in digital communication systems to quantify the difference
between the ideal transmitted signal and the received signal. It provides a comprehensive measure
of signal quality by considering both magnitude and phase errors.

This implementation supports:
- Multiple calculation modes (RMS, peak, percentile)
- Both normalized and non-normalized EVM
- Per-symbol and aggregate statistics
- Magnitude and phase error decomposition
- EVM threshold compliance checking
- Support for multi-dimensional signals (MIMO, multi-carrier)
"""

from typing import Any, Optional, Tuple

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

    Features:
    - Multiple calculation modes: 'rms', 'peak', 'percentile'
    - Normalized and non-normalized variants
    - Per-symbol and aggregate statistics
    - Magnitude and phase error decomposition
    - EVM threshold compliance checking
    - Support for multi-dimensional signals

    Attributes:
        normalize (bool): Whether to normalize by reference signal power (default: True).
        mode (str): EVM calculation mode ('rms', 'peak', or 'percentile').
        percentile (float): Percentile value when mode is 'percentile' (default: 95.0).
        threshold (Optional[float]): EVM threshold for compliance checking (in %).
    """

    is_differentiable = True
    higher_is_better = False

    def __init__(self, normalize: bool = True, mode: str = "rms", percentile: float = 95.0, threshold: Optional[float] = None, name: Optional[str] = None, *args: Any, **kwargs: Any):
        """Initialize the EVM metric.

        Args:
            normalize (bool): Whether to normalize by reference signal power (default: True).
            mode (str): EVM calculation mode ('rms', 'peak', or 'percentile').
            percentile (float): Percentile value when mode is 'percentile' (default: 95.0).
            threshold (Optional[float]): EVM threshold for compliance checking (in %).
            name (Optional[str]): Optional name for the metric.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(name=name or "EVM")
        self.normalize = normalize
        self.mode = mode.lower()
        self.percentile = percentile
        self.threshold = threshold

        # Validate parameters
        if self.mode not in ["rms", "peak", "percentile"]:
            raise ValueError(f"Mode must be 'rms', 'peak', or 'percentile', got '{mode}'")

        if not 0 < percentile <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

        if threshold is not None and threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")

        # Initialize state for torchmetrics-style interface
        self.add_state("sum_error_power", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_reference_power", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("evm_violations", default=torch.tensor(0), dist_reduce_fx="sum")

    def add_state(self, name: str, default: torch.Tensor, dist_reduce_fx: str = "sum"):
        """Add a state tensor to the module (torchmetrics compatibility)."""
        setattr(self, name, default.clone())

    def reset(self):
        """Reset the metric state."""
        self.sum_error_power = torch.tensor(0.0)
        self.sum_reference_power = torch.tensor(0.0)
        self.total = torch.tensor(0)
        self.evm_violations = torch.tensor(0)

    def update(self, x: Tensor, y: Tensor):
        """Update the metric state with new data.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Calculate error vector and power more efficiently
        error_vector = y - x
        error_power = torch.real(error_vector * torch.conj(error_vector))

        if self.normalize:
            reference_power = torch.real(x * torch.conj(x))
            # More stable clamping
            reference_power = torch.clamp(reference_power, min=1e-12)

            self.sum_error_power += torch.sum(error_power)
            self.sum_reference_power += torch.sum(reference_power)
        else:
            self.sum_error_power += torch.sum(error_power)
            self.sum_reference_power += torch.tensor(float(x.numel()))

        self.total += x.numel()

        # Track EVM violations if threshold is set
        if self.threshold is not None:
            current_evm = self.forward(x, y)
            if current_evm > self.threshold:
                self.evm_violations += 1

    def compute(self) -> Tensor:
        """Compute the final EVM value from accumulated state.

        Returns:
            Tensor: Error Vector Magnitude as a percentage.
        """
        if self.total == 0:
            return torch.tensor(0.0)

        if self.normalize:
            evm_squared = self.sum_error_power / self.sum_reference_power
        else:
            evm_squared = self.sum_error_power / self.total

        evm = torch.sqrt(evm_squared)
        return evm * 100.0

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

        # Calculate error vector more efficiently
        error_vector = y - x

        # Calculate error power using complex conjugate for better numerical stability
        error_power = torch.real(error_vector * torch.conj(error_vector))

        if self.normalize:
            # Calculate reference power
            reference_power = torch.real(x * torch.conj(x))

            # Handle zero reference signal case
            max_ref_power = torch.max(reference_power)
            if max_ref_power == 0:
                # All reference signals are zero, return normalized error power
                evm = torch.sqrt(torch.clamp(torch.mean(error_power), min=0.0)) * 100.0
                return evm

            # Improved numerical stability with relative tolerance
            eps = torch.finfo(reference_power.dtype).eps
            reference_power = torch.clamp(reference_power, min=eps * max_ref_power)

            # Normalize error power by reference power
            normalized_error = error_power / reference_power
        else:
            normalized_error = error_power

        # Calculate EVM based on mode with improved efficiency
        if self.mode == "rms":
            # RMS EVM
            evm_squared = torch.mean(normalized_error)
        elif self.mode == "peak":
            # Peak EVM
            evm_squared = torch.max(normalized_error)
        elif self.mode == "percentile":
            # Percentile EVM - flatten only once for better performance
            flattened_error = normalized_error.flatten()
            evm_squared = torch.quantile(flattened_error, self.percentile / 100.0)

        # Convert to percentage
        evm = torch.sqrt(torch.clamp(evm_squared, min=0.0)) * 100.0

        return evm

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

        # Calculate per-symbol error magnitude using complex conjugate
        error_magnitude = torch.sqrt(torch.real(error_vector * torch.conj(error_vector)))

        if self.normalize:
            # Calculate per-symbol reference magnitude
            reference_magnitude = torch.sqrt(torch.real(x * torch.conj(x)))

            # Improved numerical stability
            eps = torch.finfo(reference_magnitude.dtype).eps
            reference_magnitude = torch.clamp(reference_magnitude, min=eps)

            # Normalize by reference magnitude
            per_symbol_evm = error_magnitude / reference_magnitude
        else:
            per_symbol_evm = error_magnitude

        # Convert to percentage
        per_symbol_evm_percent = per_symbol_evm * 100.0

        return per_symbol_evm_percent

    def calculate_magnitude_phase_evm(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate separate magnitude and phase EVM components.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.

        Returns:
            tuple[Tensor, Tensor]: Magnitude EVM and Phase EVM in percentages.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Handle empty tensors
        if x.numel() == 0:
            zero_tensor = torch.tensor(0.0, dtype=torch.float32, device=x.device)
            return zero_tensor, zero_tensor

        # Extract magnitude and phase
        ref_magnitude = torch.abs(x)
        ref_phase = torch.angle(x)
        rec_magnitude = torch.abs(y)
        rec_phase = torch.angle(y)

        # Calculate magnitude error
        magnitude_error = torch.abs(rec_magnitude - ref_magnitude)

        # Calculate phase error (handle phase wrapping)
        phase_error = rec_phase - ref_phase
        phase_error = torch.atan2(torch.sin(phase_error), torch.cos(phase_error))  # Wrap to [-π, π]
        phase_error = torch.abs(phase_error)

        if self.normalize:
            # Normalize by reference values
            eps = torch.finfo(ref_magnitude.dtype).eps
            ref_magnitude_safe = torch.clamp(ref_magnitude, min=eps)

            magnitude_evm = magnitude_error / ref_magnitude_safe
            # Phase EVM normalized by unit circle (π radians = 180 degrees)
            phase_evm = phase_error / torch.pi
        else:
            magnitude_evm = magnitude_error
            phase_evm = phase_error / torch.pi  # Convert to normalized units

        # Apply aggregation mode
        if self.mode == "rms":
            magnitude_evm = torch.sqrt(torch.mean(magnitude_evm**2))
            phase_evm = torch.sqrt(torch.mean(phase_evm**2))
        elif self.mode == "peak":
            magnitude_evm = torch.max(magnitude_evm)
            phase_evm = torch.max(phase_evm)
        elif self.mode == "percentile":
            magnitude_evm = torch.quantile(magnitude_evm.flatten(), self.percentile / 100.0)
            phase_evm = torch.quantile(phase_evm.flatten(), self.percentile / 100.0)

        # Convert to percentage
        magnitude_evm_percent = magnitude_evm * 100.0
        phase_evm_percent = phase_evm * 100.0

        return magnitude_evm_percent, phase_evm_percent

    def check_compliance(self, x: Tensor, y: Tensor) -> dict:
        """Check EVM compliance against threshold.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.

        Returns:
            dict: Compliance report with pass/fail status and statistics.
        """
        if self.threshold is None:
            raise ValueError("No threshold set for compliance checking")

        evm_value = self.forward(x, y)
        per_symbol_evm = self.calculate_per_symbol_evm(x, y)

        # Calculate violation statistics
        violations = per_symbol_evm > self.threshold
        violation_count = torch.sum(violations).item()
        total_symbols = per_symbol_evm.numel()
        violation_rate = violation_count / total_symbols if total_symbols > 0 else 0.0

        # Calculate worst-case EVM
        worst_evm = torch.max(per_symbol_evm).item() if total_symbols > 0 else 0.0

        compliance_report = {
            "pass": evm_value.item() <= self.threshold,
            "evm_value": evm_value.item(),
            "threshold": self.threshold,
            "margin_db": 20 * torch.log10(torch.tensor(self.threshold / evm_value.item())).item(),
            "violation_count": violation_count,
            "total_symbols": total_symbols,
            "violation_rate": violation_rate,
            "worst_evm": worst_evm,
        }

        return compliance_report

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

        # Calculate magnitude and phase components
        magnitude_evm, phase_evm = self.calculate_magnitude_phase_evm(x, y)

        # Enhanced statistics
        stats_dict = {
            # Basic EVM statistics
            "evm_rms": self.forward(x, y),
            "evm_mean": torch.mean(per_symbol_evm),
            "evm_std": torch.std(per_symbol_evm),
            "evm_min": torch.min(per_symbol_evm),
            "evm_max": torch.max(per_symbol_evm),
            "evm_median": torch.median(per_symbol_evm),
            # Extended percentiles
            "evm_75th": torch.quantile(per_symbol_evm.flatten(), 0.75),
            "evm_90th": torch.quantile(per_symbol_evm.flatten(), 0.90),
            "evm_95th": torch.quantile(per_symbol_evm.flatten(), 0.95),
            "evm_99th": torch.quantile(per_symbol_evm.flatten(), 0.99),
            "evm_99_9th": torch.quantile(per_symbol_evm.flatten(), 0.999),
            # Magnitude and phase components
            "magnitude_evm": magnitude_evm,
            "phase_evm": phase_evm,
            # Advanced statistics
            "evm_variance": torch.var(per_symbol_evm),
            "evm_skewness": self._calculate_skewness(per_symbol_evm),
            "evm_kurtosis": self._calculate_kurtosis(per_symbol_evm),
            # Per-symbol data
            "evm_per_symbol": per_symbol_evm,
        }

        # Add compliance statistics if threshold is set
        if self.threshold is not None:
            compliance = self.check_compliance(x, y)
            stats_dict.update(
                {
                    "compliance_pass": compliance["pass"],
                    "violation_rate": compliance["violation_rate"],
                    "margin_db": compliance["margin_db"],
                }
            )

        return stats_dict

    def _calculate_skewness(self, data: Tensor) -> Tensor:
        """Calculate skewness of the data."""
        if data.numel() < 3:
            return torch.tensor(0.0, device=data.device)

        mean = torch.mean(data)
        std = torch.std(data, unbiased=False)

        if std == 0:
            return torch.tensor(0.0, device=data.device)

        normalized = (data - mean) / std
        skewness = torch.mean(normalized**3)

        return skewness

    def _calculate_kurtosis(self, data: Tensor) -> Tensor:
        """Calculate kurtosis of the data."""
        if data.numel() < 4:
            return torch.tensor(0.0, device=data.device)

        mean = torch.mean(data)
        std = torch.std(data, unbiased=False)

        if std == 0:
            return torch.tensor(0.0, device=data.device)

        normalized = (data - mean) / std
        kurtosis = torch.mean(normalized**4) - 3.0  # Subtract 3 for excess kurtosis

        return kurtosis

    def calculate_multi_dimensional_evm(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        """Calculate EVM along a specific dimension (useful for MIMO or multi-carrier systems).

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.
            dim (int): Dimension along which to calculate EVM (default: -1, last dimension).

        Returns:
            Tensor: EVM values along the specified dimension.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        # Calculate error vector
        error_vector = y - x
        error_power = torch.real(error_vector * torch.conj(error_vector))

        if self.normalize:
            reference_power = torch.real(x * torch.conj(x))
            eps = torch.finfo(reference_power.dtype).eps
            reference_power = torch.clamp(reference_power, min=eps)
            normalized_error = error_power / reference_power
        else:
            normalized_error = error_power

        # Calculate EVM along specified dimension
        if self.mode == "rms":
            evm_squared = torch.mean(normalized_error, dim=dim, keepdim=False)
        elif self.mode == "peak":
            evm_squared = torch.max(normalized_error, dim=dim, keepdim=False)[0]
        elif self.mode == "percentile":
            # Sort along dimension and take percentile
            sorted_vals = torch.sort(normalized_error, dim=dim)[0]
            idx = int((self.percentile / 100.0) * (sorted_vals.shape[dim] - 1))
            evm_squared = torch.index_select(sorted_vals, dim, torch.tensor(idx, device=sorted_vals.device))
            evm_squared = evm_squared.squeeze(dim)

        evm = torch.sqrt(torch.clamp(evm_squared, min=0.0)) * 100.0
        return evm

    def calculate_constellation_evm(self, constellation_points: Tensor, received_symbols: Tensor) -> dict:
        """Calculate EVM for each point in a constellation.

        Args:
            constellation_points (Tensor): Ideal constellation points.
            received_symbols (Tensor): Received symbols corresponding to constellation points.

        Returns:
            dict: Per-constellation-point EVM statistics.
        """
        unique_points = torch.unique(constellation_points)
        constellation_evm = {}

        for point in unique_points:
            # Find symbols corresponding to this constellation point
            mask = torch.isclose(constellation_points, point, rtol=1e-5)
            if torch.any(mask):
                ref_symbols = constellation_points[mask]
                rec_symbols = received_symbols[mask]

                # Calculate EVM for this constellation point
                point_evm = self.calculate_per_symbol_evm(ref_symbols, rec_symbols)

                constellation_evm[f"point_{point.item():.3f}"] = {
                    "mean_evm": torch.mean(point_evm),
                    "max_evm": torch.max(point_evm),
                    "std_evm": torch.std(point_evm),
                    "count": len(point_evm),
                }

        return constellation_evm

    def get_evm_vs_time(self, x: Tensor, y: Tensor, window_size: int = 100) -> Tensor:
        """Calculate EVM over sliding time windows.

        Args:
            x (Tensor): The transmitted/reference signal tensor.
            y (Tensor): The received signal tensor.
            window_size (int): Size of the sliding window.

        Returns:
            Tensor: EVM values for each time window.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: {x.shape} vs {y.shape}")

        if x.numel() < window_size:
            return self.forward(x, y).unsqueeze(0)

        # Flatten for time-series analysis
        x_flat = x.flatten()
        y_flat = y.flatten()

        num_windows = len(x_flat) - window_size + 1
        evm_time_series = torch.zeros(num_windows, device=x.device)

        for i in range(num_windows):
            x_window = x_flat[i : i + window_size]
            y_window = y_flat[i : i + window_size]
            evm_time_series[i] = self.forward(x_window, y_window)

        return evm_time_series

    def get_recommended_thresholds(self, modulation_scheme: str) -> dict:
        """Get recommended EVM thresholds for common modulation schemes.

        Args:
            modulation_scheme (str): Modulation scheme name.

        Returns:
            dict: Recommended thresholds for different applications.
        """
        # Standard EVM thresholds based on 3GPP and IEEE standards
        thresholds = {
            "bpsk": {"measurement": 8.0, "design": 5.0},
            "qpsk": {"measurement": 8.0, "design": 5.0},
            "16qam": {"measurement": 12.5, "design": 8.0},
            "64qam": {"measurement": 8.0, "design": 5.0},
            "256qam": {"measurement": 3.5, "design": 2.5},
            "1024qam": {"measurement": 1.5, "design": 1.0},
        }

        scheme = modulation_scheme.lower()
        if scheme in thresholds:
            return thresholds[scheme]
        else:
            return {"measurement": 10.0, "design": 6.0}  # Conservative defaults


# Alias for backward compatibility
EVM = ErrorVectorMagnitude
