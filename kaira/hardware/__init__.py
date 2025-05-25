"""Hardware integration module for Kaira.

This module provides interfaces for real-world hardware integration, particularly focusing on
Software Defined Radio (SDR) capabilities through GNU Radio.

Classes:
    GNURadioBridge: Main bridge class for integrating Kaira models with GNU Radio
    GNURadioTransmitter: Specialized transmitter using GNU Radio
    GNURadioReceiver: Specialized receiver using GNU Radio
    SDRConfig: Configuration dataclass for SDR hardware operations
    FrequencyBand: Common frequency bands for SDR operations
    HardwareError: Exception raised for hardware-related errors

Constants:
    GNURADIO_AVAILABLE: Boolean indicating if GNU Radio is available
"""

from typing import TYPE_CHECKING, Any

from .sdr_utils import FrequencyBand, HardwareError, SDRConfig

if TYPE_CHECKING:
    from .gnuradio_bridge import GNURadioBridge, GNURadioReceiver, GNURadioTransmitter

# Try to import GNU Radio components
try:
    from .gnuradio_bridge import (
        GNURADIO_AVAILABLE,
        GNURadioBridge,
        GNURadioReceiver,
        GNURadioTransmitter,
    )
except ImportError:
    GNURADIO_AVAILABLE = False

    # Create stub classes when GNU Radio is not available
    class _GNURadioStub:
        """Stub class when GNU Radio is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("GNU Radio is not available. Please install gnuradio>=3.9.0")

    # Use Any for type checking compatibility
    GNURadioBridge = _GNURadioStub  # type: ignore
    GNURadioTransmitter = _GNURadioStub  # type: ignore
    GNURadioReceiver = _GNURadioStub  # type: ignore


__all__ = [
    "GNURadioBridge",
    "GNURadioTransmitter",
    "GNURadioReceiver",
    "SDRConfig",
    "FrequencyBand",
    "HardwareError",
    "GNURADIO_AVAILABLE",
]
