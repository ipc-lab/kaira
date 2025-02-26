"""Modulation schemes for wireless communication systems.

This module provides implementations of common digital modulation techniques
used in wireless communications, including PSK and QAM variants.
"""

from .psk import BPSKModulator, BPSKDemodulator, QPSKModulator, QPSKDemodulator, PSKModulator, PSKDemodulator
from .qam import QAMModulator, QAMDemodulator
from .pam import PAMModulator, PAMDemodulator
from .oqpsk import OQPSKModulator, OQPSKDemodulator
from .pi4qpsk import Pi4QPSKModulator, Pi4QPSKDemodulator
from .dpsk import DPSKModulator, DPSKDemodulator, DBPSKModulator, DBPSKDemodulator, DQPSKModulator, DQPSKDemodulator
from .utils import plot_constellation, binary_to_gray, gray_to_binary, calculate_theoretical_ber
from .benchmark import awgn_channel, measure_ber, plot_ber_curve, compare_modulation_schemes, measure_throughput, benchmark_modulation_schemes
from .constellation_viz import ConstellationVisualizer

__all__ = [
    "BPSKModulator", "BPSKDemodulator",
    "QPSKModulator", "QPSKDemodulator",
    "PSKModulator", "PSKDemodulator",
    "QAMModulator", "QAMDemodulator",
    "PAMModulator", "PAMDemodulator",
    "OQPSKModulator", "OQPSKDemodulator",
    "Pi4QPSKModulator", "Pi4QPSKDemodulator",
    "DPSKModulator", "DPSKDemodulator",
    "DBPSKModulator", "DBPSKDemodulator",
    "DQPSKModulator", "DQPSKDemodulator",
    "plot_constellation", "binary_to_gray", "gray_to_binary", "calculate_theoretical_ber",
    "awgn_channel", "measure_ber", "plot_ber_curve", "compare_modulation_schemes",
    "measure_throughput", "benchmark_modulation_schemes",
    "ConstellationVisualizer"
]
