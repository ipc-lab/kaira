"""Modulation schemes for wireless communication systems.

This module provides implementations of common digital modulation techniques used in wireless
communications, including PSK and QAM variants.
"""

from .benchmark import (
    awgn_channel,
    benchmark_modulation_schemes,
    compare_modulation_schemes,
    measure_ber,
    measure_throughput,
    plot_ber_curve,
)
from .constellation_viz import ConstellationVisualizer
from .dpsk import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)
from .oqpsk import OQPSKDemodulator, OQPSKModulator
from .pam import PAMDemodulator, PAMModulator
from .pi4qpsk import Pi4QPSKDemodulator, Pi4QPSKModulator
from .psk import (
    BPSKDemodulator,
    BPSKModulator,
    PSKDemodulator,
    PSKModulator,
    QPSKDemodulator,
    QPSKModulator,
)
from .qam import QAMDemodulator, QAMModulator
from .utils import (
    binary_to_gray,
    calculate_theoretical_ber,
    gray_to_binary,
    plot_constellation,
)

__all__ = [
    "BPSKModulator",
    "BPSKDemodulator",
    "QPSKModulator",
    "QPSKDemodulator",
    "PSKModulator",
    "PSKDemodulator",
    "QAMModulator",
    "QAMDemodulator",
    "PAMModulator",
    "PAMDemodulator",
    "OQPSKModulator",
    "OQPSKDemodulator",
    "Pi4QPSKModulator",
    "Pi4QPSKDemodulator",
    "DPSKModulator",
    "DPSKDemodulator",
    "DBPSKModulator",
    "DBPSKDemodulator",
    "DQPSKModulator",
    "DQPSKDemodulator",
    "plot_constellation",
    "binary_to_gray",
    "gray_to_binary",
    "calculate_theoretical_ber",
    "awgn_channel",
    "measure_ber",
    "plot_ber_curve",
    "compare_modulation_schemes",
    "measure_throughput",
    "benchmark_modulation_schemes",
    "ConstellationVisualizer",
]
