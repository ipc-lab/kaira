"""Model registry for Kaira.

This module provides a central registry for all models in the Kaira framework. It enables dynamic
registration and creation of models by name, making it easier to configure and instantiate models
at runtime.
"""

from typing import Callable, Dict, Optional, Type

from .base import BaseModel


class ModelRegistry:
    """A registry for models in Kaira.

    This class provides a centralized registry system that maintains a mapping between
    model names and their corresponding classes. It supports:
    - Registration of new model classes (both directly and via decorator)
    - Creation of model instances by name
    - Lookup of registered model classes
    - Listing of available models

    The registry pattern enables dynamic discovery and instantiation of models, which
    is particularly useful for configurable pipelines and experiments where model
    architectures need to be selected at runtime.

    Example:
        >>> @ModelRegistry.register_model()
        >>> class MyCustomModel(BaseModel):
        ...     def __init__(self, hidden_size=64):
        ...         super().__init__()
        ...         self.hidden_size = hidden_size
        ...
        >>> # Create instance from registry
        >>> model = ModelRegistry.create("mycustommodel", hidden_size=128)
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class in the registry.

        Args:
            name (str): The name under which to register the model class.
                This name will be used to look up and create instances of the model.
            model_class (Type[BaseModel]): The model class to register. Must be a
                subclass of BaseModel.

        Raises:
            TypeError: If model_class is not a subclass of BaseModel
            ValueError: If a model with the given name is already registered
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model class {model_class.__name__} must inherit from BaseModel")
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered")
        cls._models[name] = model_class

    @classmethod
    def register_model(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a model class in the registry.

        This decorator provides a convenient way to register model classes at definition
        time. If no name is provided, the lowercase version of the class name is used
        as the registration key.

        Args:
            name (Optional[str]): Optional custom name for the model. If not provided,
                the lowercase class name will be used as the registration key.

        Returns:
            Callable: A decorator function that registers the model class

        Example:
            >>> @ModelRegistry.register_model()  # Uses class name as key
            >>> class MyModel(BaseModel):
            ...     # implementation
            ...
            >>> @ModelRegistry.register_model("better_name")  # Uses custom name
            >>> class GenericNameThatNeedsBetterRegistryKey(BaseModel):
            ...     # implementation
        """

        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            model_name = name if name is not None else model_class.__name__.lower()
            cls.register(model_name, model_class)
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name.

        Args:
            name (str): The registered name of the model class to retrieve

        Returns:
            Type[BaseModel]: The requested model class

        Raises:
            KeyError: If no model is registered under the given name
        """
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found in registry. " f"Available models: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def get_model_cls(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name.

        This is an alias for the get() method for backwards compatibility.

        Args:
            name (str): The registered name of the model class to retrieve

        Returns:
            Type[BaseModel]: The requested model class

        Raises:
            KeyError: If no model is registered under the given name
        """
        return cls.get(name)

    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        """Get information about a registered model.

        This method provides detailed information about a model registered in the registry,
        including its name, class name, and constructor signature.

        Args:
            name (str): The registered name of the model to get information about

        Returns:
            Dict: A dictionary containing information about the model:
                - name: The registered name of the model
                - class: The class name of the model
                - signature: The constructor signature of the model

        Raises:
            ValueError: If no model is registered under the given name
        """
        import inspect

        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry")

        model_class = cls._models[name]

        return {"name": name, "class": model_class.__name__, "signature": str(inspect.signature(model_class.__init__))}

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name.

        This method provides a convenient way to instantiate models from their
        registered names, passing any necessary configuration parameters.

        Args:
            name (str): The registered name of the model to create
            **kwargs: Configuration parameters to pass to the model constructor

        Returns:
            BaseModel: An instantiated model object

        Raises:
            KeyError: If no model is registered under the given name
            TypeError: If the model constructor receives invalid arguments
        """
        model_class = cls.get(name)
        try:
            return model_class(**kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to create model '{name}': {str(e)}")

    @classmethod
    def list_models(cls) -> list:
        """List all available models in the registry.

        Returns:
            list: A list of registered model names that can be used with create()
        """
        return list(cls._models.keys())
