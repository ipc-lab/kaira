"""Constraint registry for Kaira."""

from typing import Callable, Dict, Optional, Type

from .base import BaseConstraint


class ConstraintRegistry:
    """A registry for constraints in Kaira.

    This class provides a centralized registry for all constraints, making it easier to instantiate
    them by name with appropriate parameters.
    """

    _constraints: Dict[str, Type[BaseConstraint]] = {}

    @classmethod
    def register(cls, name: str, constraint_class: Type[BaseConstraint]) -> None:
        """Register a new constraint in the registry.

        Args:
            name (str): The name to register the constraint under.
            constraint_class (Type[BaseConstraint]): The constraint class to register.
        """
        cls._constraints[name] = constraint_class

    @classmethod
    def register_constraint(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a constraint class in the registry.

        Args:
            name (Optional[str], optional): The name to register the constraint under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the constraint class.
        """

        def decorator(constraint_class):
            constraint_name = name if name is not None else constraint_class.__name__.lower()
            cls.register(constraint_name, constraint_class)
            return constraint_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseConstraint]:
        """Get a constraint class by name.

        Args:
            name (str): The name of the constraint to get.

        Returns:
            Type[BaseConstraint]: The constraint class.

        Raises:
            KeyError: If the constraint is not registered.
        """
        if name not in cls._constraints:
            raise KeyError(f"Constraint '{name}' not found in registry. Available constraints: {list(cls._constraints.keys())}")
        return cls._constraints[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseConstraint:
        """Create a constraint instance by name.

        Args:
            name (str): The name of the constraint to create.
            **kwargs: Additional arguments to pass to the constraint constructor.

        Returns:
            BaseConstraint: The instantiated constraint.
        """
        constraint_class = cls.get(name)
        return constraint_class(**kwargs)

    @classmethod
    def list_constraints(cls) -> list:
        """List all available constraints in the registry.

        Returns:
            list: A list of constraint names.
        """
        return list(cls._constraints.keys())
