"""Base model definitions for deep learning architectures.

This module provides the foundation for all model implementations in the Kaira framework. The
BaseModel class implements common functionality and enforces a consistent interface across
different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List  # Added Callable

from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the Kaira framework.

    This class extends PyTorch's nn.Module and adds framework-specific functionality. All models
    should inherit from this class to ensure compatibility with the framework's training,
    evaluation, and inference pipelines.

    The class provides a consistent interface for model implementation while allowing flexibility
    in architecture design. It enforces proper initialization and forward pass implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the model.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the forward pass computation.

        This method should be implemented by all subclasses to define how input data
        is processed through the model to produce output.

        Args:
            *args: Variable positional arguments for flexible input handling
            **kwargs: Variable keyword arguments for optional parameters

        Returns:
            Any: Model output, type depends on specific implementation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ConfigurableModel(BaseModel):
    """Model that supports dynamically adding and removing steps.

    This class extends the basic model functionality with methods to add, remove, and manage model
    steps during runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the configurable model."""
        super().__init__(*args, **kwargs)
        self.steps: List[Callable] = []  # Added initialization here, changed type to List[Callable]

    def add_step(self, step: Callable) -> "ConfigurableModel":  # Changed step type to Callable
        """Add a processing step to the model.

        Args:
            step: A callable that will be added to the processing pipeline.
                Must accept and return tensor-like objects.

        Returns:
            Self for method chaining
        """
        if not callable(step):  # Added check
            raise TypeError("Step must be callable")
        self.steps.append(step)
        return self

    def remove_step(self, index: int) -> "ConfigurableModel":
        """Remove a processing step from the model.

        Args:
            index: The index of the step to remove

        Returns:
            Self for method chaining

        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        self.steps.pop(index)
        return self

    def forward(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Process input through all steps sequentially.

        Args:
            input_data (Any): The input to process
            *args (Any): Positional arguments passed to each step
            **kwargs (Any): Additional keyword arguments passed to each step

        Returns:
            The result after applying all steps
        """
        result = input_data
        for step in self.steps:
            result = step(result, *args, **kwargs)
        return result
