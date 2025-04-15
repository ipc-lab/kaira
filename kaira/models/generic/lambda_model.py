"""Defines a model based on a user-provided lambda function."""

from typing import Any, Callable, Optional

import torch

from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model()
class LambdaModel(BaseModel):
    """Lambda Model.

    This model applies a user-provided function to the input tensor. It's useful
    for quickly implementing custom transformations without creating a new model class.

    Example:
        >>> # Apply a simple scaling function
        >>> model = LambdaModel(lambda x: 2.0 * x)
        >>> x = torch.ones(5, 10)
        >>> output = model(x)
        >>> assert torch.all(output == 2.0)
    """

    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor], name: Optional[str] = None):
        """Initialize the Lambda model.

        Args:
            func (Callable[[torch.Tensor], torch.Tensor]): Function to apply to input tensors.
                Should take a torch.Tensor as input and return a torch.Tensor.
            name (Optional[str], optional): Name for the model. If None, uses the function's name.
                Defaults to None.
        """
        super().__init__()
        self.func = func
        self.name = name or func.__name__

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply the lambda function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor
            *args: Additional positional arguments passed to the function
            **kwargs: Additional keyword arguments passed to the function

        Returns:
            torch.Tensor: Result of applying the lambda function to the input
        """
        return self.func(x, *args, **kwargs)

    def __repr__(self) -> str:
        """Get string representation of the model.

        Returns:
            str: Description including model name and function
        """
        # Check if the function is a lambda by examining its name and code
        if self.func.__name__ == "<lambda>" or (hasattr(self.func, "__code__") and self.func.__code__.co_name == "<lambda>"):
            func_repr = "<lambda>"
        else:
            func_repr = str(self.func)

        return f"{self.__class__.__name__}(name={self.name}, func={func_repr})"
