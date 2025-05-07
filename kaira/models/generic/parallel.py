"""Defines a model that applies multiple modules in parallel to the input."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Tuple

from ..base import ConfigurableModel
from ..registry import ModelRegistry


@ModelRegistry.register_model()
class ParallelModel(ConfigurableModel):
    """A model that processes steps in parallel.

    All steps receive the same input data and process independently.
    """

    step_configs: List[Tuple[str, Callable]]  # Renamed from 'steps'

    def __init__(self, max_workers: Optional[int] = None, steps: Optional[List[Tuple[str, Callable]]] = None, branches: Optional[List[Callable]] = None, aggregator: Optional[Callable] = None):
        """Initialize the parallel model.

        Args:
            max_workers: Maximum number of worker threads (None uses default ThreadPoolExecutor behavior)
            steps: Optional initial list of named processing steps as (name, step) tuples
            branches: Alternative way to specify processing steps as a list of callables
            aggregator: Optional function to aggregate results (if None, returns dict of outputs)
        """
        super().__init__()
        self.max_workers = max_workers
        self.aggregator = aggregator
        self._step_counter = 0  # Counter for auto-naming steps

        # Initialize step_configs list
        if steps:
            self.step_configs = steps  # Use new attribute name
        else:
            self.step_configs = []  # Use new attribute name

        # Add branches if provided
        if branches:
            for i, branch in enumerate(branches):
                self.add_step(branch, f"branch_{i}")

    def add_step(self, step: Callable, name: Optional[str] = None):
        """Add a processing step to the model with an optional name.

        Args:
            step: A callable function or object that processes input data
            name: Optional name for the step (auto-generated if None)

        Returns:
            The model instance for method chaining

        Raises:
            TypeError: If step is not callable
        """
        if not callable(step):
            raise TypeError("Step must be callable")

        if name is None:
            name = f"step_{self._step_counter}"
            self._step_counter += 1
        self.step_configs.append((name, step))  # Use new attribute name
        return self

    def remove_step(self, index: int):
        """Remove a processing step from the model.

        Args:
            index: The index of the step to remove

        Returns:
            The model instance for method chaining

        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.step_configs):  # Use new attribute name
            raise IndexError(f"Step index {index} out of range (0-{len(self.step_configs)-1})")  # Use new attribute name
        self.step_configs.pop(index)  # Use new attribute name
        return self

    def forward(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the model in parallel on the input data.

        Args:
            input_data: The data to process
            *args: Additional positional arguments passed to each step.
            **kwargs: Additional keyword arguments passed to each step.

        Returns:
            Dictionary mapping step names to their respective outputs
            or aggregated results if an aggregator is provided
        """
        if not self.step_configs:  # Use new attribute name
            return {}

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Pass *args and **kwargs to each step submitted to the executor
            future_to_step = {executor.submit(step_func, input_data, *args, **kwargs): name for name, step_func in self.step_configs}  # Use new attribute name and step_func

            for future in as_completed(future_to_step):
                step_name = future_to_step[future]
                try:
                    results[step_name] = future.result()
                except Exception as exc:
                    results[step_name] = f"Error: {exc}"

        # Apply aggregator if provided
        if self.aggregator:
            # Convert dictionary of results to a list of values for the aggregator
            return self.aggregator(list(results.values()))
        return results
