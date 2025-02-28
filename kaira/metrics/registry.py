"""Metrics registry and factory module.

This module provides three main functionalities:
1. Registration of metrics and discovery of available metrics
2. Creation of metric instances from registered classes with proper parameters
3. Convenient factory functions for creating common metric collections and combinations

The registry pattern enables dynamic discovery of metrics and simplifies the creation
of configurable evaluation pipelines that can select metrics at runtime.

Examples:
    Register a custom metric:
    ```python
    from kaira.metrics.registry import register_metric
    from kaira.metrics.base import BaseMetric

    @register_metric()  # Uses class name as the registration key
    class MyCustomMetric(BaseMetric):
        def __init__(self, param1=1.0):
            super().__init__(name="my_custom_metric")
            self.param1 = param1

        def forward(self, x, y):
            # Implement metric computation
            return result
    ```

    Register with custom name:
    ```python
    @register_metric("awesome_metric")
    class ComplicatedMetricWithLongName(BaseMetric):
        # implementation
    ```

    Create a metric from registry:
    ```python
    from kaira.metrics.registry import create_metric

    # Create instance using registered name
    metric = create_metric("mycustommetric", param1=2.0)
    ```

    Use factory functions for common metric suites:
    ```python
    from kaira.metrics.registry import create_image_quality_metrics, create_composite_metric

    # Create standard image metrics
    metrics_dict = create_image_quality_metrics(data_range=2.0)

    # Create weighted combination
    weights = {"psnr": 0.6, "ssim": 0.4}
    combined = create_composite_metric(metrics_dict, weights)
    ```
"""

import inspect
from typing import Any, Dict, List, Literal, Optional, Type

import torch

from .base import BaseMetric
from .composite import CompositeMetric
from .image import LPIPS, PSNR, SSIM, MultiScaleSSIM

# Registries for metrics and factories
_METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register_metric(name: Optional[str] = None):
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
        ```python
        @register_metric()  # Uses class name as key
        class MyMetric(BaseMetric):
            # implementation

        @register_metric("better_name")  # Uses custom name as key
        class GenericNameThatNeedsBetterRegistryKey(BaseMetric):
            # implementation
        ```
    """

    def decorator(cls: Type[BaseMetric]) -> Type[BaseMetric]:
        metric_name = name or cls.__name__.lower()
        if metric_name in _METRIC_REGISTRY:
            raise ValueError(f"Metric with name '{metric_name}' already registered")
        _METRIC_REGISTRY[metric_name] = cls
        return cls

    return decorator


def create_metric(name: str, **kwargs: Any) -> BaseMetric:
    """Create a metric instance from the registry with the specified parameters.

    This function instantiates a registered metric class with the provided parameters,
    allowing for flexible creation of metrics at runtime based on configuration.

    Args:
        name (str): Name of the metric to create (case-sensitive registry key)
        **kwargs: Arguments to pass to the metric constructor. These should match
            the parameters expected by the metric's __init__ method.

    Returns:
        BaseMetric: Instantiated metric object ready for use

    Raises:
        KeyError: If the metric name is not found in the registry
        TypeError: If the provided kwargs don't match the metric's expected parameters

    Example:
        ```python
        # Create a PSNR metric with custom parameters
        psnr = create_metric("psnr", data_range=255.0)

        # Create a custom registered metric
        my_metric = create_metric("mycustommetric", param1=10, param2="value")
        ```
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not found in registry. Available metrics: {list(_METRIC_REGISTRY.keys())}")
    return _METRIC_REGISTRY[name](**kwargs)


def list_metrics() -> List[str]:
    """List all registered metrics available for creation.

    This function returns the names of all metrics that have been registered
    and can be instantiated using the create_metric function.

    Returns:
        List[str]: Names (registry keys) of all registered metrics

    Example:
        ```python
        available_metrics = list_metrics()
        print(f"Available metrics: {available_metrics}")

        # Check if a specific metric is available
        if "lpips" in list_metrics():
            metric = create_metric("lpips")
        ```
    """
    return list(_METRIC_REGISTRY.keys())


def get_metric_info(name: str) -> Dict[str, Any]:
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
        ```python
        # Get information about the PSNR metric
        psnr_info = get_metric_info("psnr")
        print(f"PSNR parameters: {psnr_info['parameters']}")
        print(f"Documentation: {psnr_info['docstring']}")
        ```
    """
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not found in registry")

    metric_class = _METRIC_REGISTRY[name]
    signature = inspect.signature(metric_class.__init__)
    params = {k: v.default if v.default is not inspect.Parameter.empty else None for k, v in list(signature.parameters.items())[1:]}  # Skip 'self'

    return {
        "name": name,
        "class": metric_class.__name__,
        "module": metric_class.__module__,
        "docstring": inspect.getdoc(metric_class),
        "parameters": params,
    }


def create_image_quality_metrics(data_range: float = 1.0, lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex", device: Optional[torch.device] = None) -> Dict[str, BaseMetric]:
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
        ```python
        import torch

        # Create metrics for normalized images [0,1]
        metrics = create_image_quality_metrics(data_range=1.0, device=torch.device('cuda'))

        # Generate some test images
        pred = torch.rand(1, 3, 256, 256).cuda()  # Batch of random RGB images
        target = torch.rand(1, 3, 256, 256).cuda()

        # Compute metrics individually
        psnr_value = metrics['psnr'](pred, target)
        ssim_value = metrics['ssim'](pred, target)

        # Or create a composite metric
        composite = create_composite_metric(metrics, weights={'psnr': 0.5, 'ssim': 0.5})
        score = composite(pred, target)
        ```
    """
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


def create_composite_metric(metrics: Dict[str, BaseMetric], weights: Optional[Dict[str, float]] = None) -> BaseMetric:
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
        ```python
        from kaira.metrics import PSNR, SSIM
        from kaira.metrics.registry import create_composite_metric

        # Create individual metrics
        psnr = PSNR(data_range=1.0)
        ssim = SSIM(data_range=1.0)
        lpips = LPIPS(net_type='alex')  # Lower values are better

        # Create a balanced composite metric (higher values = better)
        metrics = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
        weights = {'psnr': 0.4, 'ssim': 0.4, 'lpips': -0.2}  # Negative weight for LPIPS

        balanced_metric = create_composite_metric(metrics, weights)
        ```
    """
    return CompositeMetric(metrics, weights)
