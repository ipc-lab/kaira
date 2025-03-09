"""Pipeline registry for Kaira."""

from typing import Callable, Dict, Optional, Type

from .base import BasePipeline


class PipelineRegistry:
    """A registry for pipelines in Kaira.

    This class provides a centralized registry for all pipelines, making it easier to instantiate them
    by name with appropriate parameters.
    """

    _pipelines: Dict[str, Type[BasePipeline]] = {}

    @classmethod
    def register(cls, name: str, pipeline_class: Type[BasePipeline]) -> None:
        """Register a new pipeline in the registry.

        Args:
            name (str): The name to register the pipeline under.
            pipeline_class (Type[BasePipeline]): The pipeline class to register.
        """
        cls._pipelines[name] = pipeline_class

    @classmethod
    def register_pipeline(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a pipeline class in the registry.

        Args:
            name (Optional[str], optional): The name to register the pipeline under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the pipeline class.
        """

        def decorator(pipeline_class):
            pipeline_name = name if name is not None else pipeline_class.__name__.lower()
            cls.register(pipeline_name, pipeline_class)
            return pipeline_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BasePipeline]:
        """Get a pipeline class by name.

        Args:
            name (str): The name of the pipeline to get.

        Returns:
            Type[BasePipeline]: The pipeline class.

        Raises:
            KeyError: If the pipeline is not registered.
        """
        if name not in cls._pipelines:
            raise KeyError(f"Pipeline '{name}' not found in registry. Available pipelines: {list(cls._pipelines.keys())}")
        return cls._pipelines[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BasePipeline:
        """Create a pipeline instance by name.

        Args:
            name (str): The name of the pipeline to create.
            **kwargs: Additional arguments to pass to the pipeline constructor.

        Returns:
            BasePipeline: The instantiated pipeline.
        """
        pipeline_class = cls.get(name)
        return pipeline_class(**kwargs)

    @classmethod
    def list_pipelines(cls) -> list:
        """List all available pipelines in the registry.

        Returns:
            list: A list of pipeline names.
        """
        return list(cls._pipelines.keys())
