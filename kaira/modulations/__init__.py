"""Modulation schemes for wireless communication systems.

This module provides implementations of common digital modulation techniques
used in wireless communications, including PSK and QAM variants.
"""

from .base import Modulator, Demodulator
from .psk import BPSKModulator, BPSKDemodulator, QPSKModulator, QPSKDemodulator, PSKModulator, PSKDemodulator
from .qam import QAMModulator, QAMDemodulator
from .pam import PAMModulator, PAMDemodulator
from .oqpsk import OQPSKModulator, OQPSKDemodulator

__all__ = [
    "Modulator", "Demodulator",
    "BPSKModulator", "BPSKDemodulator",
    "QPSKModulator", "QPSKDemodulator",
    "PSKModulator", "PSKDemodulator",
    "QAMModulator", "QAMDemodulator",
    "PAMModulator", "PAMDemodulator",
    "OQPSKModulator", "OQPSKDemodulator",
]
