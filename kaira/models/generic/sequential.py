"""Defines a sequential model container."""

from typing import Any, Callable, Optional, Sequence

from ..base import ConfigurableModel
from ..registry import ModelRegistry


@ModelRegistry.register_model()
class SequentialModel(ConfigurableModel):
    """A model that processes steps sequentially.

    Each step receives the output of the previous step as its input.
    """

    def __init__(self, steps: Optional[Sequence[Callable]] = None, *args: Any, **kwargs: Any):
        """Initialize the sequential model.

        Args:
            steps: Optional initial list of processing steps
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        if steps:
            self.steps = list(steps)

    def add_step(self, step: Callable):
        """Add a processing step to the model.

        Args:
            step: A callable function or object that processes input data

        Returns:
            The model instance for method chaining
        """
        if not callable(step):
            raise TypeError("Step must be callable")
        return super().add_step(step)

    def forward(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the model sequentially on the input data.

        Args:
            x: The initial data to process
            *args: Additional positional arguments passed to each step.
            **kwargs: Additional keyword arguments passed to each step.

        Returns:
            The final result after passing through all steps
        """
        result = x
        for step in self.steps:
            result = step(result, *args, **kwargs)  # Pass *args and **kwargs to each step
        return result
