"""Model registry for Kaira."""

from typing import Callable, Dict, Type

from .base import BaseModel


class ModelRegistry:
    """A registry for models in Kaira.

    This class provides a centralized registry for all models, making it easier to instantiate them
    by name with appropriate parameters.
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model in the registry.

        Args:
            name (str): The name to register the model under.
            model_class (Type[BaseModel]): The model class to register.
        """
        cls._models[name] = model_class

    @classmethod
    def register_model(cls, name: str = None) -> Callable:
        """Decorator to register a model class in the registry.

        Args:
            name (str, optional): The name to register the model under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the model class.
        """

        def decorator(model_class):
            model_name = name if name is not None else model_class.__name__.lower()
            cls.register(model_name, model_class)
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name.

        Args:
            name (str): The name of the model to get.

        Returns:
            Type[BaseModel]: The model class.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in cls._models:
            raise KeyError(
                f"Model '{name}' not found in registry. Available models: {list(cls._models.keys())}"
            )
        return cls._models[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name.

        Args:
            name (str): The name of the model to create.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            BaseModel: The instantiated model.
        """
        model_class = cls.get(name)
        return model_class(**kwargs)

    @classmethod
    def list_models(cls) -> list:
        """List all available models in the registry.

        Returns:
            list: A list of model names.
        """
        return list(cls._models.keys())
