"""Forward Error Correction (FEC) package for Kaira.

This package provides implementations of various forward error correction techniques, enabling
robust data transmission over noisy channels. FEC codes add redundant information to transmitted
data, allowing receivers to detect and correct errors without retransmission.

Key components:
- algebra: Mathematical foundations for finite fields and binary polynomials
- encoders: Various channel encoding schemes (block codes, algebraic codes, etc.)
- decoders: Implementations of corresponding decoding algorithms
- utils: Utility functions for binary operations and code manipulation

Common FEC codes implemented:
- Block codes: Linear block codes, systematic codes, single parity check
- Cyclic codes: BCH, Reed-Solomon, Golay codes
- Other codes: Hamming, repetition codes, Reed-Muller

These implementations can be used in communication system simulations, coded modulation
schemes, and for educational purposes in information theory and coding :cite:`lin2004error,moon2005error`.
"""

from . import algebra, decoders, encoders, utils

__all__ = ["algebra", "encoders", "decoders", "utils"]
