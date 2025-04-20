"""Forward Error Correction (FEC) package for Kaira.

This package provides implementations of various forward error correction techniques, particularly
focusing on binary error correction codes like block codes, cyclic codes, and more advanced
algebraic codes like BCH codes.
"""

from . import algebra, encoders

__all__ = ["algebra", "encoders"]
