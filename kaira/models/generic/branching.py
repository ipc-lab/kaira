from typing import Any, Callable, Dict, Optional

from ..base import BaseModel
from ..registry import ModelRegistry


@ModelRegistry.register_model()
class BranchingModel(BaseModel):
    """A model that routes input to different branches based on conditions.

    Each branch has a condition that determines if it should process the input. Only the first
    matching branch (or default) will be executed.
    """

    def __init__(self):
        """Initialize the branching model."""
        super().__init__()
        self.branches = {}  # Dict of name -> (condition, model)
        self.default_branch = None

    def add_branch(self, condition: Callable[[Any], bool], model: BaseModel, name: Optional[str] = None):
        """Add a conditional branch with associated model.

        Args:
            condition: A callable that evaluates if this branch should be taken
            model: The model to execute for this branch
            name: Optional name for the branch (auto-generated if None)

        Returns:
            The model instance for method chaining

        Raises:
            TypeError: If condition is not callable or model is not a BaseModel instance
        """
        if not callable(condition):
            raise TypeError("Branch condition must be callable")
        if not isinstance(model, BaseModel):
            raise TypeError("Branch model must be a BaseModel instance")

        if name is None:
            name = f"branch_{len(self.branches)}"
        self.branches[name] = (condition, model)
        return self

    def set_default_branch(self, model: BaseModel):
        """Set the default branch for when no conditions match.

        Args:
            model: The model to execute as default

        Returns:
            The model instance for method chaining

        Raises:
            TypeError: If model is not a BaseModel instance
        """
        if not isinstance(model, BaseModel):
            raise TypeError("Default branch model must be a BaseModel instance")
        self.default_branch = model
        return self

    def remove_branch(self, name: str):
        """Remove a branch by name.

        Args:
            name: The name of the branch to remove

        Returns:
            The model instance for method chaining

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
        for name, (condition, model) in self.branches.items():
            if condition(input_data):
                results[name] = model.forward(input_data)
                return results

        # Use default branch if no condition matched
        if self.default_branch:
            results["default"] = self.default_branch.forward(input_data)

        return results
