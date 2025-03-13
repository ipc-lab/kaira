from typing import Any, Callable, Optional, Sequence

from ..base import ConfigurableModel
from ..registry import ModelRegistry


@ModelRegistry.register_model()
class SequentialModel(ConfigurableModel):
    """A model that processes steps sequentially.

    Each step receives the output of the previous step as its input.
    """

    def __init__(self, steps: Optional[Sequence[Callable]] = None):
        """Initialize the sequential model.

        Args:
            steps: Optional initial list of processing steps
        """
        super().__init__()
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

    def forward(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the model sequentially on the input data.

        Args:
            input_data: The initial data to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The final result after passing through all steps
        """
        result = input_data
        for step in self.steps:
            result = step(result)
        return result
