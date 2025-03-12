"""Loss registry for Kaira."""

from typing import Callable, Dict, Optional, Type

from .base import BaseLoss


class LossRegistry:
    """A registry for loss functions in Kaira.

    This class provides a centralized registry for all loss functions, making it easier to
    instantiate them by name with appropriate parameters.
    """

    _losses: Dict[str, Type[BaseLoss]] = {}

    @classmethod
    def register(cls, name: str, loss_class: Type[BaseLoss]) -> None:
        """Register a new loss in the registry.

        Args:
            name (str): The name to register the loss under.
            loss_class (Type[BaseLoss]): The loss class to register.
        """
        cls._losses[name] = loss_class

    @classmethod
    def register_loss(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a loss class in the registry.

        Args:
            name (Optional[str], optional): The name to register the loss under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the loss class.
        """

        def decorator(loss_class):
            loss_name = name if name is not None else loss_class.__name__.lower()
            cls.register(loss_name, loss_class)
            return loss_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseLoss]:
        """Get a loss class by name.

        Args:
            name (str): The name of the loss to get.

        Returns:
            Type[BaseLoss]: The loss class.

        Raises:
            KeyError: If the loss is not registered.
        """
        if name not in cls._losses:
            raise KeyError(f"Loss '{name}' not found in registry. Available losses: {list(cls._losses.keys())}")
        return cls._losses[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLoss:
        """Create a loss instance by name.

        Args:
            name (str): The name of the loss to create.
            **kwargs: Additional arguments to pass to the loss constructor.

        Returns:
            BaseLoss: The instantiated loss.
        """
        loss_class = cls.get(name)
        return loss_class(**kwargs)

    @classmethod
    def list_losses(cls) -> list:
        """List all available losses in the registry.

        Returns:
            list: A list of loss names.
        """
        return list(cls._losses.keys())
