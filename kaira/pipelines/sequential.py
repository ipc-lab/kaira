from typing import Any, Callable, Optional, Sequence

from .base import ConfigurablePipeline
from .registry import PipelineRegistry


@PipelineRegistry.register_pipeline()
class SequentialPipeline(ConfigurablePipeline):
    """A pipeline that processes steps sequentially.

    Each step receives the output of the previous step as its input.
    """

    def __init__(self, steps: Optional[Sequence[Callable]] = None):
        """Initialize the sequential pipeline.

        Args:
            steps: Optional initial list of processing steps
        """
        super().__init__()
        if steps:
            self.steps = list(steps)

    def add_step(self, step: Callable):
        """Add a processing step to the pipeline.

        Args:
            step: A callable function or object that processes input data

        Returns:
            The pipeline instance for method chaining
        """
        if not callable(step):
            raise TypeError("Step must be callable")
        return super().add_step(step)

    def forward(self, input_data: Any) -> Any:
        """Execute the pipeline sequentially on the input data.

        Args:
            input_data: The initial data to process

        Returns:
            The final result after passing through all steps
        """
        result = input_data
        for step in self.steps:
            result = step(result)
        return result
