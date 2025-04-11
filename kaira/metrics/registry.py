"""Metrics registry and factory module.

This module provides three main functionalities:
1. Registration of metrics and discovery of available metrics
2. Creation of metric instances from registered classes with proper parameters
3. Convenient factory functions for creating common metric collections and combinations

The registry pattern enables dynamic discovery of metrics and simplifies the creation
of configurable evaluation pipelines that can select metrics at runtime.

Examples:
    Register a custom metric:

    >>> from kaira.metrics.registry import MetricRegistry
    >>> from kaira.metrics.base import BaseMetric
    >>>
    >>> @MetricRegistry.register_metric()  # Uses class name as the registration key
    >>> class MyCustomMetric(BaseMetric):
    ...     def __init__(self, param1=1.0):
    ...         super().__init__(name="my_custom_metric")
    ...         self.param1 = param1
    ...
    ...     def forward(self, x, y):
    ...         # Implement metric computation
    ...         return result

    Register with custom name:

    >>> @MetricRegistry.register_metric("awesome_metric")
    >>> class ComplicatedMetricWithLongName(BaseMetric):
    ...     # implementation

    Create a metric from registry:

    >>> from kaira.metrics.registry import MetricRegistry
    >>>
    >>> # Create instance using registered name
    >>> metric = MetricRegistry.create("mycustommetric", param1=2.0)

    Use factory functions for common metric suites:

    >>> from kaira.metrics.registry import MetricRegistry
    >>>
    >>> # Create standard image metrics
    >>> metrics_dict = MetricRegistry.create_image_quality_metrics(data_range=2.0)
    >>>
    >>> # Create weighted combination
    >>> weights = {"psnr": 0.6, "ssim": 0.4}
    >>> combined = MetricRegistry.create_composite_metric(metrics_dict, weights)
"""

import inspect
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import torch

from .base import BaseMetric
from .composite import CompositeMetric


class MetricRegistry:
    """A registry for metrics in Kaira.

    This class provides a centralized registry for all metrics, making it easier to instantiate
    them by name with appropriate parameters.
    """

    _metrics: Dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, name: str, metric_class: Type[BaseMetric]) -> None:
        """Register a new metric in the registry.

        Args:
            name (str): The name to register the metric under.
            metric_class (Type[BaseMetric]): The metric class to register.
        """
        if name in cls._metrics:
            raise ValueError(f"Metric with name '{name}' already registered")
        cls._metrics[name] = metric_class

    @classmethod
    def register_metric(cls, name: Optional[str] = None) -> Callable[[Type[BaseMetric]], Type[BaseMetric]]:
        """Decorator to register a metric class in the global registry.

        This makes the metric discoverable and instantiable through the registry system.
        Each registered metric must inherit from BaseMetric to ensure compatibility.

        Args:
            name (Optional[str]): Optional custom name for the metric. If not provided,
                the lowercase class name will be used as the registration key.
                Using custom names is helpful for shorter keys or when the class name
                is not descriptive enough.

        Returns:
            Callable: Decorator function that registers the metric class

        Example:
            >>> @MetricRegistry.register_metric()  # Uses class name as key
            >>> class MyMetric(BaseMetric):
            ...     # implementation
            ...
            >>> @MetricRegistry.register_metric("better_name")  # Uses custom name as key
            >>> class GenericNameThatNeedsBetterRegistryKey(BaseMetric):
            ...     # implementation
        """

        def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
            metric_name = name or cls.__name__.lower()
            MetricRegistry.register(metric_name, cls)
            return cls

        return decorator

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> BaseMetric:
        """Create a metric instance from the registry with the specified parameters.

        This function instantiates a registered metric class with the provided parameters,
        allowing for flexible creation of metrics at runtime based on configuration.

        Args:
            name (str): Name of the metric to create (case-sensitive registry key)
            *args: Positional arguments to pass to the metric constructor
            **kwargs: Keyword arguments to pass to the metric constructor. These should match
                the parameters expected by the metric's __init__ method.

        Returns:
            BaseMetric: Instantiated metric object ready for use

        Raises:
            KeyError: If the metric name is not found in the registry
            TypeError: If the provided args/kwargs don't match the metric's expected parameters

        Example:
            >>> # Create a PSNR metric with custom parameters
            >>> psnr = MetricRegistry.create("psnr", data_range=255.0)
            >>>
            >>> # Create a custom registered metric with positional arguments
            >>> my_metric = MetricRegistry.create("mycustommetric", 10, param2="value")
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found in registry. Available metrics: {list(cls._metrics.keys())}")
        return cls._metrics[name](*args, **kwargs)

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metrics available for creation.

        This function returns the names of all metrics that have been registered
        and can be instantiated using the create_metric function.

        Returns:
            List[str]: Names (registry keys) of all registered metrics

        Example:
            >>> available_metrics = MetricRegistry.list_metrics()
            >>> print(f"Available metrics: {available_metrics}")
            >>>
            >>> # Check if a specific metric is available
            >>> if "lpips" in MetricRegistry.list_metrics():
            ...     metric = MetricRegistry.create("lpips")
        """
        return list(cls._metrics.keys())

    @classmethod
    def get_metric_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed information about a registered metric.

        This function provides introspection capabilities to examine a metric's
        parameters, documentation, and other metadata without instantiating it.
        Useful for dynamic UI generation or parameter validation.

        Args:
            name (str): Name of the metric to inspect

        Returns:
            Dict[str, Any]: Dictionary containing:
                - name: Registry key of the metric
                - class: Original class name
                - module: Module where the class is defined
                - docstring: Documentation string
                - parameters: Dictionary of parameter names and default values

        Raises:
            KeyError: If the metric name is not found in the registry

        Example:
            >>> # Get information about the PSNR metric
            >>> psnr_info = MetricRegistry.get_metric_info("psnr")
            >>> print(f"PSNR parameters: {psnr_info['parameters']}")
            >>> print(f"Documentation: {psnr_info['docstring']}")
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' not found in registry")

        metric_class = cls._metrics[name]
        signature = inspect.signature(metric_class.__init__)
        params = {k: v.default if v.default is not inspect.Parameter.empty else None for k, v in list(signature.parameters.items())[1:]}  # Skip 'self'

        return {
            "name": name,
            "class": metric_class.__name__,
            "module": metric_class.__module__,
            "docstring": inspect.getdoc(metric_class),
            "parameters": params,
        }

    @classmethod
    def create_image_quality_metrics(cls, data_range: float = 1.0, lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex", device: Optional[torch.device] = None) -> Dict[str, BaseMetric]:
        """Create a standard suite of image quality assessment metrics.

        This factory function creates a collection of commonly used image quality metrics
        with consistent parameters, making it easy to evaluate images across multiple metrics.

        The returned metrics include:
        - PSNR (Peak Signal-to-Noise Ratio): A pixel-level fidelity metric
        - SSIM (Structural Similarity Index): A perceptual metric focusing on structure
        - MS-SSIM (Multi-Scale SSIM): A multi-scale version of SSIM
        - LPIPS (Learned Perceptual Image Patch Similarity): A learned perceptual metric

        Args:
            data_range (float): The data range of the images. Use 1.0 for normalized images
                in range [0,1] or 255.0 for uint8 images in range [0,255].
            lpips_net_type (Literal['vgg', 'alex', 'squeeze']): The backbone network for LPIPS. Options are:
                - 'alex': AlexNet (faster, less accurate)
                - 'vgg': VGG network (slower, more accurate)
                - 'squeeze': SqueezeNet (fastest, least accurate)
            device (Optional[torch.device]): Device to place the metrics on.
                If None, metrics will be on the default device (typically CPU).

        Returns:
            Dict[str, BaseMetric]: Dictionary mapping metric names to initialized metrics.
                All metrics follow the BaseMetric interface and can be called directly
                with input tensors.

        Example:
            >>> import torch
            >>>
            >>> # Create metrics for normalized images [0,1]
            >>> metrics = MetricRegistry.create_image_quality_metrics(data_range=1.0, device=torch.device('cuda'))
            >>>
            >>> # Generate some test images
            >>> pred = torch.rand(1, 3, 256, 256).cuda()  # Batch of random RGB images
            >>> target = torch.rand(1, 3, 256, 256).cuda()
            >>>
            >>> # Compute metrics individually
            >>> psnr_value = metrics['psnr'](pred, target)
            >>> ssim_value = metrics['ssim'](pred, target)
            >>>
            >>> # Or create a composite metric
            >>> composite = MetricRegistry.create_composite_metric(metrics, weights={'psnr': 0.5, 'ssim': 0.5})
            >>> score = composite(pred, target)
        """
        from .image import LPIPS, PSNR, SSIM, MultiScaleSSIM

        metrics = {
            "psnr": PSNR(data_range=data_range),
            "ssim": SSIM(data_range=data_range),
            "ms_ssim": MultiScaleSSIM(data_range=data_range),
            "lpips": LPIPS(net_type=lpips_net_type),
        }

        if device is not None:
            for metric in metrics.values():
                metric.to(device)

        return metrics

    @classmethod
    def create_composite_metric(cls, metrics: Dict[str, BaseMetric], weights: Optional[Dict[str, float]] = None) -> BaseMetric:
        """Create a composite metric that combines multiple metrics with weights.

        This factory function creates a CompositeMetric instance that applies multiple
        metrics to the same inputs and combines their results according to specified weights.

        This is useful for:
        - Creating custom evaluation criteria that balance multiple aspects
        - Combining complementary metrics (e.g., pixel accuracy and perceptual quality)
        - Building task-specific evaluation metrics that focus on relevant properties

        Args:
            metrics (Dict[str, BaseMetric]): Dictionary mapping metric names to metric objects.
                All provided metrics should follow the BaseMetric interface.
            weights (Optional[Dict[str, float]]): Optional dictionary mapping metric names to
                their relative weights. If None, metrics will be equally weighted.

                Use negative weights for metrics where lower values are better (like LPIPS)
                when combining with metrics where higher values are better (like PSNR/SSIM).

        Returns:
            BaseMetric: A composite metric that combines the provided metrics according
                to the specified weights. This metric follows the BaseMetric interface
                and can be used like any other metric.

        Example:
            >>> from kaira.metrics import PSNR, SSIM
            >>> from kaira.metrics.registry import MetricRegistry
            >>>
            >>> # Create individual metrics
            >>> psnr = PSNR(data_range=1.0)
            >>> ssim = SSIM(data_range=1.0)
            >>> lpips = LPIPS(net_type='alex')  # Lower values are better
            >>>
            >>> # Create a balanced composite metric (higher values = better)
            >>> metrics = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
            >>> weights = {'psnr': 0.4, 'ssim': 0.4, 'lpips': -0.2}  # Negative weight for LPIPS
            >>>
            >>> balanced_metric = MetricRegistry.create_composite_metric(metrics, weights)
        """
        return CompositeMetric(metrics, weights)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics from the registry.

        This is primarily useful for testing and reinitialization scenarios.
        """
        cls._metrics.clear()

    @classmethod
    def available_metrics(cls) -> List[str]:
        """Get a list of all available metrics in the registry. This is an alias for list_metrics()
        for backward compatibility.

        Returns:
            List[str]: List of registered metric names
        """
        return cls.list_metrics()
