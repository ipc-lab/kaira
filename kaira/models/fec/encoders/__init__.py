"""Forward Error Correction encoders for Kaira.

This module provides various encoder implementations for forward error correction, including:
- Block codes: Fundamental error correction codes that operate on fixed-size blocks
- Linear block codes: Codes with linear algebraic structure allowing matrix operations
- LDPC codes: Low-Density Parity-Check codes with sparse parity-check matrices
- Cyclic codes: Special class of linear codes with cyclic shift properties
- BCH codes: Powerful algebraic codes with precise error-correction capabilities
- Reed-Solomon codes: Widely-used subset of BCH codes for burst error correction
- Hamming codes: Simple single-error-correcting codes with efficient implementation
- Repetition codes: Basic codes that repeat each bit multiple times
- Golay codes: Perfect codes with specific error correction properties
- Single parity-check codes: Simple error detection through parity bit addition

These encoders can be used to add redundancy to data for enabling error detection and correction
in communication systems, storage devices, and other applications requiring reliable data
transmission over noisy channels :cite:`lin2004error,moon2005error`.
"""

from .base import BaseBlockCodeEncoder
from .bch_code import BCHCodeEncoder
from .cyclic_code import CyclicCodeEncoder
from .golay_code import GolayCodeEncoder
from .hamming_code import HammingCodeEncoder
from .ldpc_code import LDPCCodeEncoder
from .linear_block_code import LinearBlockCodeEncoder
from .reed_solomon_code import ReedSolomonCodeEncoder
from .repetition_code import RepetitionCodeEncoder
from .single_parity_check_code import SingleParityCheckCodeEncoder
from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder

__all__ = ["BaseBlockCodeEncoder", "LinearBlockCodeEncoder", "LDPCCodeEncoder", "SystematicLinearBlockCodeEncoder", "HammingCodeEncoder", "RepetitionCodeEncoder", "CyclicCodeEncoder", "BCHCodeEncoder", "GolayCodeEncoder", "ReedSolomonCodeEncoder", "SingleParityCheckCodeEncoder"]
