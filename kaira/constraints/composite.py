"""Composite constraint implementation for combining multiple constraints.

This module provides the CompositeConstraint class, which allows multiple constraints to be applied
sequentially as a single unified constraint. This enables modular constraint creation and
composition for complex signal requirements.
"""

from typing import Sequence

from .base import BaseConstraint


class CompositeConstraint(BaseConstraint):
    """Applies multiple signal constraints in sequence as a single unified constraint.

    This class combines multiple BaseConstraint objects into a single constraint that applies
    each component constraint sequentially. It inherits from both BaseConstraint to provide constraint functionality.

    The composite pattern allows complex constraint combinations to be treated as a
    single constraint object, enabling modular constraint creation and reuse.

    Attributes:
        constraints (list): List of BaseConstraint objects to apply in sequence

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

    def __init__(self, constraints: Sequence[BaseConstraint]) -> None:
        """Initialize a composite constraint with a list of component constraints.

        Args:
            constraints (Sequence[BaseConstraint]): List of constraint objects to apply in sequence

        Raises:
            ValueError: If constraints list is empty
        """
        if not constraints:
            raise ValueError("CompositeConstraint requires at least one constraint")

        # Convert to List[Callable] for SequentialModel compatibility
        super().__init__(constraints)

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Add a new constraint to the composite.

        Args:
            constraint (BaseConstraint): New constraint to add to the sequence
        """
        if not isinstance(constraint, BaseConstraint):
            raise TypeError(f"Expected BaseConstraint, got {type(constraint).__name__}")

        self.add_step(constraint)

    def forward(self, x):
        """Apply the composite constraint to the input signal.

        Args:
            x: Input signal to constrain

        Returns:
            Constrained signal after applying all component constraints
        """
        for step in self.steps:
            x = step(x)

        return x
