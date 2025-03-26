"""Constraints module for Kaira.

This module contains various constraints that can be applied to transmitted signals in
wireless communication systems. These constraints ensure signals meet practical requirements
such as power limitations, hardware capabilities, and regulatory specifications.

Available constraint categories:
- Base constraint definitions: Abstract base classes for all constraints
- Power constraints: Control total power, average power, and PAPR
- Antenna constraints: Manage power distribution across multiple antennas
- Signal constraints: Handle amplitude limitations and spectral properties
- Constraint composition: Combine multiple constraints sequentially

The module also provides factory functions for creating common constraint combinations
and utilities for testing and validating constraint effectiveness.

Example:
    >>> from kaira.constraints import TotalPowerConstraint, PAPRConstraint
    >>> from kaira.constraints.utils import combine_constraints
    >>>
    >>> # Create individual constraints
    >>> power_constr = TotalPowerConstraint(total_power=1.0)
    >>> papr_constr = PAPRConstraint(max_papr=4.0)
    >>>
    >>> # Combine constraints into a single operation
    >>> combined = combine_constraints([power_constr, papr_constr])
    >>>
    >>> # Apply to a signal
    >>> constrained_signal = combined(input_signal)
"""

# Utility functions
from . import utils

# Antenna constraints
from .antenna import PerAntennaPowerConstraint

# Base constraint
from .base import BaseConstraint

# Composite constraint
from .composite import CompositeConstraint

# Basic functional constraints
from .identity import IdentityConstraint
from .lambda_constraint import LambdaConstraint

# Power constraints
from .power import AveragePowerConstraint, PAPRConstraint, TotalPowerConstraint
from .registry import ConstraintRegistry

# Signal constraints
from .signal import PeakAmplitudeConstraint, SpectralMaskConstraint

__all__ = [
    # Base classes
    "BaseConstraint",
    "CompositeConstraint",
    # Basic functional constraints
    "IdentityConstraint",
    "LambdaConstraint",
    # Power constraints
    "TotalPowerConstraint",
    "AveragePowerConstraint",
    "PAPRConstraint",
    # Antenna constraints
    "PerAntennaPowerConstraint",
    # Signal constraints
    "PeakAmplitudeConstraint",
    "SpectralMaskConstraint",
    # Utility functions
    "utils",
    # Registry
    "ConstraintRegistry",
]
