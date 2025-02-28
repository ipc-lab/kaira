from typing import Any, Callable, Dict, Optional

from .base import BasePipeline


class BranchingPipeline(BasePipeline):
    """A pipeline that routes input to different branches based on conditions.

    Each branch has a condition that determines if it should process the input. Only the first
    matching branch (or default) will be executed.
    """

    def __init__(self):
        """Initialize the branching pipeline."""
        super().__init__()
        self.branches = {}  # Dict of name -> (condition, pipeline)
        self.default_branch = None

    def add_branch(self, condition: Callable[[Any], bool], pipeline: BasePipeline, name: Optional[str] = None):
        """Add a conditional branch with associated pipeline.

        Args:
            condition: A callable that evaluates if this branch should be taken
            pipeline: The pipeline to execute for this branch
            name: Optional name for the branch (auto-generated if None)

        Returns:
            The pipeline instance for method chaining

        Raises:
            TypeError: If condition is not callable or pipeline is not a BasePipeline instance
        """
        if not callable(condition):
            raise TypeError("Branch condition must be callable")
        if not isinstance(pipeline, BasePipeline):
            raise TypeError("Branch pipeline must be a BasePipeline instance")

        if name is None:
            name = f"branch_{len(self.branches)}"
        self.branches[name] = (condition, pipeline)
        return self

    def set_default_branch(self, pipeline: BasePipeline):
        """Set the default branch for when no conditions match.

        Args:
            pipeline: The pipeline to execute as default

        Returns:
            The pipeline instance for method chaining

        Raises:
            TypeError: If pipeline is not a BasePipeline instance
        """
        if not isinstance(pipeline, BasePipeline):
            raise TypeError("Default branch pipeline must be a BasePipeline instance")
        self.default_branch = pipeline
        return self

    def remove_branch(self, name: str):
        """Remove a branch by name.

        Args:
            name: The name of the branch to remove

        Returns:
            The pipeline instance for method chaining

        Raises:
            KeyError: If the branch name doesn't exist
        """
        if name not in self.branches:
            raise KeyError(f"Branch '{name}' not found")
        del self.branches[name]
        return self

    def forward(self, input_data: Any) -> Dict[str, Any]:
        """Execute the appropriate branch based on the conditions.

        Args:
            input_data: The data to process

        Returns:
            Dictionary with a single key-value pair containing the name of the
            executed branch and its result
        """
        results = {}

        # Check each branch condition
        for name, (condition, pipeline) in self.branches.items():
            if condition(input_data):
                results[name] = pipeline.forward(input_data)
                return results

        # Use default branch if no condition matched
        if self.default_branch:
            results["default"] = self.default_branch.forward(input_data)

        return results
