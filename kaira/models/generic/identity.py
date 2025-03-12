from kaira.models.registry import ModelRegistry
import torch

from kaira.models.base import BaseModel


@ModelRegistry.register_model()
class IdentityModel(BaseModel):
    """Identity Model.
    
    This model returns the input tensor without any modifications. It can be used
    as a baseline model or as a placeholder in model pipelines.
    
    Example:
        >>> model = IdentityModel()
        >>> x = torch.randn(5, 10)
        >>> output = model(x)
        >>> assert torch.allclose(x, output)
    """
    
    def __init__(self):
        """Initialize the IdentityModel."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The input tensor (unchanged).
        """
        return x
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}()"
