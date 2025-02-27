"""Digital modulation schemes for wireless communication systems.

This module provides implementations of common digital modulation techniques used in wireless
communications, with a focus on PyTorch-based implementations that support GPU acceleration.
"""

# Version information
__version__ = "0.1.0"

# Import base classes
from .base import BaseDemodulator, BaseModulator
from .dpsk import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)

# Import primary public API
from .factory import (
    Modem,
    available_schemes,
    create_demodulator,
    create_modem,
    create_modulator,
    get_modulation_info,
)
from .identity import IdentityDemodulator, IdentityModulator
from .oqpsk import OQPSKDemodulator, OQPSKModulator
from .pam import PAMDemodulator, PAMModulator
from .pi4qpsk import Pi4QPSKDemodulator, Pi4QPSKModulator

# Import modulation classes for direct access
from .psk import (
    BPSKDemodulator,
    BPSKModulator,
    PSKDemodulator,
    PSKModulator,
    QPSKDemodulator,
    QPSKModulator,
)
from .qam import QAMDemodulator, QAMModulator

# Import utilities
from .utils import binary_to_gray, gray_to_binary, plot_constellation

# Define the public API
__all__ = [
    # API
    "available_schemes",
    "create_demodulator",
    "create_modulator",
    "create_modem",
    "get_modulation_info",
    "Modem",
    # Base classes
    "BaseModulator",
    "BaseDemodulator",
    # PSK family
    "BPSKModulator",
    "BPSKDemodulator",
    "QPSKModulator",
    "QPSKDemodulator",
    "PSKModulator",
    "PSKDemodulator",
    # QAM family
    "QAMModulator",
    "QAMDemodulator",
    # PAM family
    "PAMModulator",
    "PAMDemodulator",
    # Differential schemes
    "DPSKModulator",
    "DPSKDemodulator",
    "DBPSKModulator",
    "DBPSKDemodulator",
    "DQPSKModulator",
    "DQPSKDemodulator",
    # Offset schemes
    "OQPSKModulator",
    "OQPSKDemodulator",
    "Pi4QPSKModulator",
    "Pi4QPSKDemodulator",
    # Identity scheme
    "IdentityModulator",
    "IdentityDemodulator",
    # Utilities
    "plot_constellation",
    # Coding utilities
    "binary_to_gray",
    "gray_to_binary",
]
