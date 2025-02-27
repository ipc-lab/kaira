from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import ConfigurablePipeline


class ParallelPipeline(ConfigurablePipeline):
    """A pipeline that processes steps in parallel.

    All steps receive the same input data and process independently.
    """

    def __init__(
        self, max_workers: Optional[int] = None, steps: Optional[List[Tuple[str, Callable]]] = None
    ):
        """Initialize the parallel pipeline.

        Args:
            max_workers: Maximum number of worker threads (None uses default ThreadPoolExecutor behavior)
            steps: Optional initial list of named processing steps as (name, step) tuples
        """
        super().__init__()
        self.max_workers = max_workers
        if steps:
            self.steps = list(steps)

    def add_step(self, step: Callable, name: Optional[str] = None):
        """Add a processing step to the pipeline with an optional name.

        Args:
            step: A callable function or object that processes input data
            name: Optional name for the step (auto-generated if None)

        Returns:
            The pipeline instance for method chaining

        Raises:
            TypeError: If step is not callable
        """
        if not callable(step):
            raise TypeError("Step must be callable")

        if name is None:
            name = f"step_{len(self.steps)}"
        self.steps.append((name, step))
        return self

    def remove_step(self, index: int):
        """Remove a processing step from the pipeline.

        Args:
            index: The index of the step to remove

        Returns:
            The pipeline instance for method chaining

        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        self.steps.pop(index)
        return self

    def forward(self, input_data: Any) -> Dict[str, Any]:
        """Execute the pipeline in parallel on the input data.

        Args:
            input_data: The data to process

        Returns:
            Dictionary mapping step names to their respective outputs
        """
        if not self.steps:
            return {}

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_step = {executor.submit(step, input_data): name for name, step in self.steps}

            for future in as_completed(future_to_step):
                step_name = future_to_step[future]
                try:
                    results[step_name] = future.result()
                except Exception as exc:
                    results[step_name] = f"Error: {exc}"

        return results
