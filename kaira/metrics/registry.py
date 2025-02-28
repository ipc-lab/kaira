"""Registry for metrics and metric factories.

This module provides functionality for registering, discovering and creating metrics.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import BaseMetric

# Registries for metrics and factories
_METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}
_FACTORY_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: Optional[str] = None):
    """Decorator to register a metric class.

    Args:
        name (Optional[str]): Optional name for the metric. If not provided, the class name will be used.

    Returns:
        Callable: Decorator function
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        metric_name = name or cls.__name__.lower()
        if metric_name in _METRIC_REGISTRY:
            raise ValueError(f"Metric with name '{metric_name}' already registered")
        _METRIC_REGISTRY[metric_name] = cls
        return cls

    return decorator


def register_factory(name: str):
    """Decorator to register a factory function.

    Args:
        name (str): Name of the factory

    Returns:
        Callable: Decorator function
    """

    def decorator(func: Callable) -> Callable:
        if name in _FACTORY_REGISTRY:
            raise ValueError(f"Factory with name '{name}' already registered")
        _FACTORY_REGISTRY[name] = func
        return func

    return decorator


def create_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Create a metric instance from registry.

    Args:
        name (str): Name of the metric to create
        **kwargs: Arguments to pass to the metric constructor

    Returns:
        BaseMetric: Instantiated metric

    Raises:
        KeyError: If the metric name is not registered
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Metric '{name}' not found in registry. Available metrics: {list(_METRIC_REGISTRY.keys())}"
        )
    return _METRIC_REGISTRY[name](**kwargs)


def create_from_factory(name: str, **kwargs: Any) -> Any:
    """Create metrics using a registered factory.

    Args:
        name (str): Name of the factory to use
        **kwargs: Arguments to pass to the factory function

    Returns:
        Any: Result of the factory function

    Raises:
        KeyError: If the factory name is not registered
    """
    if name not in _FACTORY_REGISTRY:
        raise KeyError(
            f"Factory '{name}' not found in registry. Available factories: {list(_FACTORY_REGISTRY.keys())}"
        )
    return _FACTORY_REGISTRY[name](**kwargs)


def list_metrics() -> List[str]:
    """List all registered metrics.

    Returns:
        List[str]: Names of all registered metrics
    """
    return list(_METRIC_REGISTRY.keys())


def list_factories() -> List[str]:
    """List all registered factories.

    Returns:
        List[str]: Names of all registered factories
    """
    return list(_FACTORY_REGISTRY.keys())


def get_metric_info(name: str) -> Dict[str, Any]:
    """Get information about a registered metric.

    Args:
        name (str): Name of the metric

    Returns:
        Dict[str, Any]: Information about the metric

    Raises:
        KeyError: If the metric name is not registered
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not found in registry")

    metric_class = _METRIC_REGISTRY[name]
    signature = inspect.signature(metric_class.__init__)
    params = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in list(signature.parameters.items())[1:]  # Skip 'self'
    }

    return {
        "name": name,
        "class": metric_class.__name__,
        "module": metric_class.__module__,
        "docstring": inspect.getdoc(metric_class),
        "parameters": params,
    }
