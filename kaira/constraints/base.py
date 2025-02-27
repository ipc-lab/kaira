from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional

import torch
from torch import nn
from kaira.pipelines.sequential import SequentialPipeline

class BaseConstraint(nn.Module, ABC):
    """Base Constraint Module.

    This is an abstract base class for defining constraints on the transmitted signal. Subclasses
    should implement the forward method to apply the constraint.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the constraint.

        Args:
            x (torch.Tensor): The input signal.

        Returns:
            torch.Tensor: The output signal after applying the constraint.
        """
        pass
    
    @staticmethod
    def get_dimensions(x: torch.Tensor, exclude_batch: bool = True) -> Tuple[int, ...]:
        """Helper method to get all dimensions except batch for calculating norms/means.
        
        Args:
            x (torch.Tensor): Input tensor
            exclude_batch (bool): Whether to exclude the batch dimension (first dimension)
            
        Returns:
            Tuple[int, ...]: Dimensions to use for reduction operations
        """
        start_dim = 1 if exclude_batch else 0
        return tuple(range(start_dim, len(x.shape)))
    
    def __repr__(self) -> str:
        """Return a string representation of the constraint."""
        return f"{self.__class__.__name__}()"


class CompositeConstraint(SequentialPipeline):
    pass