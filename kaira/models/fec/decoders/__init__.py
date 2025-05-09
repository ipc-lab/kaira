"""Forward Error Correction (FEC) decoders for Kaira.

This module provides various decoder implementations for forward error correction codes.
The decoders in this module are designed to work seamlessly with the corresponding encoders
from the `kaira.models.fec.encoders` module.

Decoders
--------
- BlockDecoder: Base class for all block code decoders
- SyndromeLookupDecoder: Decoder using syndrome lookup tables for efficient error correction
- BerlekampMasseyDecoder: Implementation of Berlekamp-Massey algorithm for decoding BCH and Reed-Solomon codes
- ReedMullerDecoder: Implementation of Reed-Muller decoding algorithm for Reed-Muller codes
- WagnerSoftDecisionDecoder: Implementation of Wagner's soft-decision decoder for single-parity check codes
- BruteForceMLDecoder: Maximum likelihood decoder that searches through all possible codewords
- BeliefPropagationDecoder: Implementation of belief propagation algorithm for decoding LDPC codes

These decoders can be used to recover original messages from possibly corrupted codewords
that have been transmitted over noisy channels. Each decoder has specific strengths and
is optimized for particular types of codes or error patterns.

Examples
--------
>>> from kaira.models.fec.encoders import BCHCodeEncoder
>>> from kaira.models.fec.decoders import BerlekampMasseyDecoder
>>> encoder = BCHCodeEncoder(15, 7)
>>> decoder = BerlekampMasseyDecoder(encoder)
>>> # Example decoding
>>> received = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
>>> decoded = decoder(received)
"""

from .base import BaseBlockDecoder
from .belief_propagation import BeliefPropagationDecoder
from .berlekamp_massey import BerlekampMasseyDecoder
from .brute_force_ml import BruteForceMLDecoder
from .reed_muller_decoder import ReedMullerDecoder
from .syndrome_lookup import SyndromeLookupDecoder
from .wagner_soft_decision_decoder import WagnerSoftDecisionDecoder

__all__ = [
    "BaseBlockDecoder",
    "SyndromeLookupDecoder",
    "BerlekampMasseyDecoder",
    "ReedMullerDecoder",
    "WagnerSoftDecisionDecoder",
    "BruteForceMLDecoder",
    "BeliefPropagationDecoder",
]
