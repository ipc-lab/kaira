"""Base constraint definitions for signal processing.

This module defines the abstract base classes for all constraint implementations in the Kaira
constraints system. The BaseConstraint class provides the foundation for creating constraints that
can be applied to signals in a PyTorch-compatible way.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from kaira.pipelines.sequential import SequentialPipeline


class BaseConstraint(nn.Module, ABC):
    """Abstract foundation for implementing signal constraints in PyTorch-compatible format.

    This is an abstract base class for defining constraints on transmitted signals. Subclasses
    should implement the forward method to apply the specific constraint logic.

    All constraints inherit from both nn.Module and ABC (Abstract Base Class) to ensure they are
    PyTorch-compatible and require implementation of key methods.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the constraint.

        Applies the constraint to the input tensor and returns the constrained output.
        This method must be implemented by all subclasses.

        Args:
            x (torch.Tensor): The input signal tensor

        Returns:
            torch.Tensor: The output signal tensor after applying the constraint
        """
        pass

    @staticmethod
    def get_dimensions(x: torch.Tensor, exclude_batch: bool = True) -> Tuple[int, ...]:
        """Helper method to get all dimensions except batch for calculating norms/means.

        Utility function to generate dimension indices for reduction operations like
        mean or norm. Typically used to calculate signal properties across all dimensions
        except the batch dimension.

        Args:
            x (torch.Tensor): Input tensor
            exclude_batch (bool, optional): Whether to exclude the batch dimension
                (first dimension). Defaults to True.

        Returns:
            Tuple[int, ...]: Dimensions to use for reduction operations (e.g., mean, norm)

        Example:
            >>> x = torch.randn(32, 4, 128)  # [batch, antennas, time]
            >>> dims = BaseConstraint.get_dimensions(x)
            >>> # dims will be (1, 2) for summing across antennas and time
        """
        start_dim = 1 if exclude_batch else 0
        return tuple(range(start_dim, len(x.shape)))


class CompositeConstraint(BaseConstraint, SequentialPipeline):
    """Applies multiple signal constraints in sequence as a single unified constraint.

    This class combines multiple BaseConstraint objects into a single constraint that applies
    each component constraint sequentially. It inherits from both BaseConstraint and
    SequentialPipeline to provide constraint functionality with sequential processing
    capabilities.

    The composite pattern allows complex constraint combinations to be treated as a
    single constraint object, enabling modular constraint creation and reuse.

    Attributes:
        constraints (list): List of BaseConstraint objects to apply in sequence

    Example:
        >>> power_constraint = TotalPowerConstraint(1.0)
        >>> papr_constraint = PAPRConstraint(4.0)
        >>> combined = CompositeConstraint([power_constraint, papr_constraint])
        >>> # Or using the utility function:
        >>> # combined = combine_constraints([power_constraint, papr_constraint])
        >>> constrained_signal = combined(input_signal)

    Note:
        When a composite constraint is applied, each component constraint is applied
        in the order they were provided. This ordering can significantly affect the
        final result, as constraints may interact with each other.
    """

    pass
