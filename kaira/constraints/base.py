"""Base constraint definitions for signal processing.

This module defines the abstract base classes for all constraint implementations in the Kaira
constraints system. The BaseConstraint class provides the foundation for creating constraints that
can be applied to signals in a PyTorch-compatible way.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


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
