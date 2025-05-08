"""Branching model for conditional processing paths.

This module provides the BranchingModel class which enables dynamic routing of inputs through
different processing paths based on runtime conditions.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from ..base import BaseModel
from ..registry import ModelRegistry
from .identity import IdentityModel


@ModelRegistry.register_model()
class BranchingModel(BaseModel):
    """Model that routes inputs through different paths based on conditions.

    This model enables conditional processing by maintaining a collection of branches,
    where each branch consists of:
    - A condition function that determines if the branch should be taken
    - A model that processes the input when the branch is taken

    The model also supports a default branch that is taken when no other branch
    conditions are met.

    Key features:
    - Dynamic branch selection based on input or state
    - Multiple independent processing paths
    - Optional default path for unmatched conditions
    - Branch conditions can be any callable returning a boolean
    - Branch models can be any BaseModel instance

    Example:
        >>> model = BranchingModel()
        >>> # Add branch for small inputs
        >>> model.add_branch("small",
        ...                  condition=lambda x: x.shape[-1] < 64,
        ...                  model=small_processor)
        >>> # Add branch for large inputs
        >>> model.add_branch("large",
        ...                  condition=lambda x: x.shape[-1] >= 64,
        ...                  model=large_processor)
        >>> # Process input - automatically selects appropriate branch
        >>> output = model(input_tensor)
    """

    def __init__(self, condition: Optional[Callable[[Any], bool]] = None, true_branch: Optional[BaseModel] = None, false_branch: Optional[BaseModel] = None):
        """Initialize a branching model.

        Args:
            condition: Optional condition function for simple true/false branching
            true_branch: Model to use when condition is True
            false_branch: Model to use when condition is False
        """
        super().__init__()
        self.branches: Dict[str, Tuple[Callable[[Any], bool], BaseModel]] = {}
        self.default_branch: Optional[BaseModel] = None

        # Set up simple binary branching if condition is provided
        if condition is not None:
            # Use identity models if branches not specified
            true_model = true_branch if true_branch is not None else IdentityModel()
            false_model = false_branch if false_branch is not None else IdentityModel()

            # Add branches
            self.add_branch("true_branch", condition=condition, model=true_model)
            self.set_default_branch(false_model)

    def add_branch(self, name: str, condition: Callable[[Any], bool], model: BaseModel) -> None:
        """Add a new conditional branch.

        Args:
            name: Unique identifier for the branch
            condition: Function that determines if branch should be taken.
                Should take same input as model and return bool.
            model: Model to use when branch is taken

        Raises:
            ValueError: If branch name already exists
        """
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")
        self.branches[name] = (condition, model)

    def set_default_branch(self, model: BaseModel) -> None:
        """Set the default branch model.

        The default branch is used when no other branch conditions are met.

        Args:
            model: Model to use as default branch
        """
        self.default_branch = model

    def remove_branch(self, name: str) -> None:
        """Remove a branch by name.

        Args:
            name: Name of branch to remove

        Raises:
            KeyError: If branch doesn't exist
        """
        if name not in self.branches:
            raise KeyError(f"Branch '{name}' not found")
        del self.branches[name]

    def forward(self, x: Any, return_branch: bool = False, *args: Any, **kwargs: Any) -> Any:
        """Process input through the appropriate branch.

        Evaluates branch conditions in registration order and processes input
        through the first matching branch. If no branches match and a default
        branch exists, processes through default branch.

        Args:
            x: Input to process
            return_branch: If True, returns tuple of (output, branch_name)
            *args: Additional positional arguments passed to branch models
            **kwargs: Additional keyword arguments passed to branch models

        Returns:
            - If return_branch=False: Output from selected branch
            - If return_branch=True: Tuple of (output, branch_name)

        Raises:
            RuntimeError: If no matching branch and no default branch
        """
        # Check each branch condition
        for name, (condition, model) in self.branches.items():
            # Convert PyTorch tensor to Python bool if needed
            condition_result = condition(x)
            if hasattr(condition_result, "item"):
                condition_result = bool(condition_result.item())
            else:
                condition_result = bool(condition_result)

            if condition_result:
                output = model(x, *args, **kwargs)
                return (output, name) if return_branch else output

        # Use default branch if no conditions matched
        if self.default_branch is not None:
            output = self.default_branch(x, *args, **kwargs)
            return (output, "default") if return_branch else output

        raise RuntimeError("No matching branch conditions and no default branch set")

    def get_branch(self, name: str) -> Tuple[Callable[[Any], bool], BaseModel]:
        """Get a branch's condition and model.

        Args:
            name: Name of branch to retrieve

        Returns:
            Tuple of (condition_function, model)

        Raises:
            KeyError: If branch doesn't exist
        """
        if name not in self.branches:
            raise KeyError(f"Branch '{name}' not found")

        # Return a wrapped condition function that converts PyTorch tensors to Python bools
        original_condition, model = self.branches[name]

        def wrapped_condition(x):
            result = original_condition(x)
            if hasattr(result, "item"):
                return bool(result.item())
            return bool(result)

        return wrapped_condition, model
