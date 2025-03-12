from typing import Callable, Optional, Any

from kaira.models.registry import ModelRegistry
import torch

from kaira.models.base import BaseModel


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
        """Initialize the LambdaModel.
        
        Args:
            func (callable): A function to be applied to the input tensor.
            name (Optional[str]): A name for the lambda function, used in __repr__.
                                 If None, uses the function's __name__ attribute.
        """
        super().__init__()
        self.func = func
        self.name = name if name is not None else getattr(func, "__name__", "anonymous_function")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The result of applying the provided function to the input tensor.
        """
        return self.func(x)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(func={self.name})"
