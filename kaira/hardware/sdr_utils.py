"""SDR utilities and configuration for hardware integration."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class FrequencyBand(Enum):
    """Common frequency bands for SDR operations."""

    ISM_433 = 433e6  # 433 MHz ISM band
    ISM_915 = 915e6  # 915 MHz ISM band
    ISM_2400 = 2.4e9  # 2.4 GHz ISM band
    WIFI_2400 = 2.4e9  # WiFi 2.4 GHz
    WIFI_5000 = 5e9  # WiFi 5 GHz
    GPS_L1 = 1575.42e6  # GPS L1
    FM_BROADCAST = 100e6  # FM radio


@dataclass
class SDRConfig:
    """Configuration for SDR hardware operations."""

    # Frequency settings
    center_frequency: float  # Hz
    sample_rate: float  # Hz
    bandwidth: Optional[float] = None  # Hz, defaults to sample_rate

    # Hardware settings
    tx_gain: float = 0.0  # dB
    rx_gain: float = 0.0  # dB
    antenna: str = "TX/RX"

    # Signal processing
    buffer_size: int = 1024
    num_channels: int = 1

    # Device settings
    device_args: Optional[str] = None
    stream_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.bandwidth is None:
            self.bandwidth = self.sample_rate

        if self.center_frequency <= 0:
            raise ValueError("Center frequency must be positive")

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        if not (0 <= self.tx_gain <= 100):
            raise ValueError("TX gain must be between 0 and 100 dB")

        if not (0 <= self.rx_gain <= 100):
            raise ValueError("RX gain must be between 0 and 100 dB")


class HardwareError(Exception):
    """Exception raised for hardware-related errors."""

    pass


def validate_frequency_range(frequency: float, min_freq: float = 70e6, max_freq: float = 6e9) -> bool:
    """Validate frequency is within typical SDR range.

    Args:
        frequency: Frequency in Hz
        min_freq: Minimum allowed frequency in Hz
        max_freq: Maximum allowed frequency in Hz

    Returns:
        True if frequency is valid

    Raises:
        ValueError: If frequency is out of range
    """
    if not (min_freq <= frequency <= max_freq):
        raise ValueError(f"Frequency {frequency/1e6:.1f} MHz is outside valid range " f"[{min_freq/1e6:.1f}, {max_freq/1e6:.1f}] MHz")
    return True


def db_to_linear(db_value: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db_value / 10.0)


def linear_to_db(linear_value: float) -> float:
    """Convert linear scale to dB."""
    return 10 * np.log10(linear_value)


def normalize_complex_signal(signal: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """Normalize complex signal to target power.

    Args:
        signal: Complex signal array
        target_power: Target signal power

    Returns:
        Normalized signal
    """
    current_power = np.mean(np.abs(signal) ** 2)
    if current_power > 0:
        scale_factor = np.sqrt(target_power / current_power)
        return signal * scale_factor
    return signal
