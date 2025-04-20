"""Forward Error Correction encoders for Kaira.

This module provides various encoder implementations for forward error correction, including:
- Block codes
- Linear block codes
- Cyclic codes
- BCH codes
- Reed-Solomon codes
- Hamming codes
- Repetition codes
- Golay codes
- Single parity-check codes

These encoders can be used to add redundancy to data to enable error detection and correction.
"""

from .bch_code import BCHCodeEncoder
from .block_code import BlockCodeEncoder
from .cyclic_code import CyclicCodeEncoder
from .golay_code import GolayCodeEncoder
from .hamming_code import HammingCodeEncoder
from .linear_block_code import LinearBlockCodeEncoder
from .reed_solomon_code import ReedSolomonCodeEncoder
from .repetition_code import MajorityVoteDecoder, RepetitionCodeEncoder
from .single_parity_check_code import SingleParityCheckCodeEncoder
from .systematic_linear_block_code import SystematicLinearBlockCodeEncoder

__all__ = ["BlockCodeEncoder", "LinearBlockCodeEncoder", "SystematicLinearBlockCodeEncoder", "HammingCodeEncoder", "RepetitionCodeEncoder", "MajorityVoteDecoder", "CyclicCodeEncoder", "BCHCodeEncoder", "GolayCodeEncoder", "ReedSolomonCodeEncoder", "SingleParityCheckCodeEncoder"]
