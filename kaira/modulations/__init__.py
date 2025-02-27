"""Digital modulation schemes for wireless communication systems.

This module provides implementations of common digital modulation techniques used in wireless
communications, with a focus on PyTorch-based implementations that support GPU acceleration.
"""

# Version information
__version__ = "0.1.0"

# Import primary public API
from .api import (
    Modem,
    available_schemes,
    create_demodulator,
    create_modem,
    create_modulator,
    get_modulation_info,
)

# Import benchmarking tools
from .benchmark import (
    benchmark_modulation_schemes,
    compare_modulation_schemes,
    measure_ber,
    plot_ber_curve,
)

# Import channel models
from .channels.awgn import AWGNChannel
from .schemes.differential import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)
from .schemes.identity import IdentityDemodulator, IdentityModulator
from .schemes.offset import (
    OQPSKDemodulator,
    OQPSKModulator,
    Pi4QPSKDemodulator,
    Pi4QPSKModulator,
)
from .schemes.pam import PAMDemodulator, PAMModulator

# Import modulation classes for direct access
from .schemes.psk import (
    BPSKDemodulator,
    BPSKModulator,
    PSKDemodulator,
    PSKModulator,
    QPSKDemodulator,
    QPSKModulator,
)
from .schemes.qam import QAMDemodulator, QAMModulator
from .utils.coding import binary_to_gray, gray_to_binary
from .utils.metrics import calculate_spectral_efficiency, calculate_theoretical_ber

# Import utilities
from .utils.visualization import ConstellationVisualizer, plot_constellation

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
    # Visualization
    "ConstellationVisualizer",
    "plot_constellation",
    # Coding utilities
    "binary_to_gray",
    "gray_to_binary",
    # Metrics
    "calculate_theoretical_ber",
    "calculate_spectral_efficiency",
    # Channel models
    "AWGNChannel",
    # Benchmarking
    "measure_ber",
    "compare_modulation_schemes",
    "plot_ber_curve",
    "benchmark_modulation_schemes",
]
