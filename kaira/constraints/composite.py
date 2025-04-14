"""Composite constraint implementation for combining multiple constraints.

This module provides the CompositeConstraint class, which allows multiple constraints to be applied
sequentially as a single unified constraint. This enables modular constraint creation and
composition for complex signal requirements.
"""

from typing import Sequence

import torch
from torch import nn

from .base import BaseConstraint


class CompositeConstraint(BaseConstraint):
    """Applies multiple constraints in sequence.

    This constraint combines multiple independent constraints and applies them
    in sequence to the input tensor. This allows for more complex constraint
    compositions like applying both power and spectral constraints together.

    Attributes:
        constraints (nn.ModuleList): List of constraint modules to apply in sequence

    Example:
        >>> power_constraint = TotalPowerConstraint(1.0)
        >>> papr_constraint = PAPRConstraint(4.0)
        >>> combined = CompositeConstraint([power_constraint, papr_constraint])
        >>> constrained_signal = combined(input_signal)

    Note:
        When a composite constraint is applied, each component constraint is applied
        in the order they were provided. This ordering can significantly affect the
        final result, as constraints may interact with each other.
    """

    def __init__(self, constraints: Sequence[BaseConstraint] | nn.ModuleList, *args, **kwargs) -> None:
        """Initialize the composite constraint with a list of constraints.

        Args:
            constraints (Sequence[BaseConstraint] | nn.ModuleList): List of constraint modules to apply in sequence
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            TypeError: If any element in constraints is not a BaseConstraint
        """
        super().__init__(*args, **kwargs)  # Call parent constructor

        # Validate that all constraints are BaseConstraint instances
        for constraint in constraints:
            if not isinstance(constraint, BaseConstraint):
                raise TypeError(f"Expected BaseConstraint, got {type(constraint).__name__}")

        self.constraints = constraints if isinstance(constraints, torch.nn.ModuleList) else torch.nn.ModuleList(constraints)

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Add a new constraint to the composite.

        Args:
            constraint (BaseConstraint): New constraint to add to the sequence
        """
        if not isinstance(constraint, BaseConstraint):
            raise TypeError(f"Expected BaseConstraint, got {type(constraint).__name__}")
        self.constraints.append(constraint)

    def forward(self, x, *args, **kwargs):
        """Apply the composite constraint to the input signal.

        Args:
            x: Input signal to constrain
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Constrained signal after applying all component constraints
        """
        for step in self.constraints:
            x = step(x, *args, **kwargs)

        return x
