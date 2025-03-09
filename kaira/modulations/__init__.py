"""Digital modulation schemes for wireless communications.

This package provides implementations of common digital modulation and demodulation techniques used
in modern communication systems, including PSK, QAM, PAM, and differential modulation schemes.
"""

# Base classes
from .base import BaseDemodulator, BaseModulator

# Differential schemes
from .dpsk import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)

# Identity schemes (for testing/debugging)
from .identity import IdentityDemodulator, IdentityModulator
from .oqpsk import OQPSKDemodulator, OQPSKModulator

# PAM schemes
from .pam import PAMDemodulator, PAMModulator

# Special PSK variants
from .pi4qpsk import Pi4QPSKDemodulator, Pi4QPSKModulator

# PSK schemes
from .psk import (
    BPSKDemodulator,
    BPSKModulator,
    PSKDemodulator,
    PSKModulator,
    QPSKDemodulator,
    QPSKModulator,
)

# QAM schemes
from .qam import QAMDemodulator, QAMModulator

# Utility functions
from .utils import (
    binary_array_to_gray,
    binary_to_gray,
    calculate_spectral_efficiency,
    calculate_theoretical_ber,
    gray_array_to_binary,
    gray_to_binary,
    plot_constellation,
)

from .registry import ModulationRegistry

__all__ = [
    # Base classes
    "BaseModulator",
    "BaseDemodulator",
    # QAM schemes
    "QAMModulator",
    "QAMDemodulator",
    # PSK schemes
    "BPSKModulator",
    "BPSKDemodulator",
    "QPSKModulator",
    "QPSKDemodulator",
    "PSKModulator",
    "PSKDemodulator",
    # PAM schemes
    "PAMModulator",
    "PAMDemodulator",
    # Special PSK variants
    "Pi4QPSKModulator",
    "Pi4QPSKDemodulator",
    "OQPSKModulator",
    "OQPSKDemodulator",
    # Differential schemes
    "DPSKModulator",
    "DPSKDemodulator",
    "DBPSKModulator",
    "DBPSKDemodulator",
    "DQPSKModulator",
    "DQPSKDemodulator",
    # Identity schemes
    "IdentityModulator",
    "IdentityDemodulator",
    # Utility functions
    "binary_to_gray",
    "gray_to_binary",
    "binary_array_to_gray",
    "gray_array_to_binary",
    "plot_constellation",
    "calculate_theoretical_ber",
    "calculate_spectral_efficiency",
    # Registry
    "ModulationRegistry",
]
