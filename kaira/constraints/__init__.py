"""Constraints module for Kaira.

This module contains various constraints that can be applied to transmitted signals,
organized into the following categories:
- Base constraint definitions (BaseConstraint)
- Power constraints (TotalPowerConstraint, PAPRConstraint, etc.)
- Antenna constraints (PerAntennaPowerConstraint)
- Signal constraints (PeakAmplitudeConstraint, SpectralMaskConstraint)
- Constraint composition (CompositeConstraint)
- Factory functions for creating common constraint combinations
- Testing/validation utilities
"""

# Utility functions
from . import utils

# Antenna constraints
from .antenna import PerAntennaPowerConstraint

# Base constraints
from .base import BaseConstraint
from .composite import CompositeConstraint

# Power constraints
from .power import (
    AveragePowerConstraint,
    ComplexAveragePowerConstraint,
    ComplexTotalPowerConstraint,
    PAPRConstraint,
    TotalPowerConstraint,
)

# Signal constraints
from .signal import PeakAmplitudeConstraint, SpectralMaskConstraint

__all__ = [
    "BaseConstraint",
    "CompositeConstraint",
    "TotalPowerConstraint",
    "AveragePowerConstraint",
    "ComplexTotalPowerConstraint",
    "ComplexAveragePowerConstraint",
    "PAPRConstraint",
    "PerAntennaPowerConstraint",
    "PeakAmplitudeConstraint",
    "SpectralMaskConstraint",
    "utils",
]
