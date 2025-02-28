"""Factory functions for creating metrics with common configurations."""

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from .base import BaseMetric
from .image import LPIPS, PSNR, SSIM, MultiScaleSSIM
from .registry import register_factory


@register_factory("image_quality_suite")
def create_image_quality_metrics(
    data_range: float = 1.0, lpips_net_type: str = "alex", device: Optional[torch.device] = None
) -> Dict[str, BaseMetric]:
    """Create a suite of standard image quality metrics.

    Args:
        data_range (float): Data range for the metrics
        lpips_net_type (str): Network type for LPIPS
        device (Optional[torch.device]): Device to place the metrics on

    Returns:
        Dict[str, BaseMetric]: Dictionary of metrics
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


@register_factory("composite_metric")
def create_composite_metric(
    metrics: Dict[str, BaseMetric], weights: Optional[Dict[str, float]] = None
) -> BaseMetric:
    """Create a composite metric that combines multiple metrics.

    Args:
        metrics (Dict[str, BaseMetric]): Dictionary of metrics
        weights (Optional[Dict[str, float]]): Weights for each metric

    Returns:
        BaseMetric: Composite metric
    """
    return CompositeMetric(metrics, weights)


class CompositeMetric(BaseMetric):
    """A metric that combines multiple metrics with optional weighting."""

    def __init__(self, metrics: Dict[str, BaseMetric], weights: Optional[Dict[str, float]] = None):
        """Initialize composite metric.

        Args:
            metrics (Dict[str, BaseMetric]): Metrics to combine
            weights (Optional[Dict[str, float]]): Weights for each metric
        """
        super().__init__(name="CompositeMetric")
        self.metrics = nn.ModuleDict(metrics)
        self.weights = weights or {name: 1.0 for name in metrics}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of metrics.

        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor

        Returns:
            torch.Tensor: Combined metric value
        """
        result = 0.0
        for name, metric in self.metrics.items():
            if name in self.weights:
                metric_value = metric(x, y)
                if isinstance(metric_value, tuple):
                    metric_value = metric_value[0]  # Take mean if tuple of (mean, std)
                result = result + self.weights[name] * metric_value
        return result

    def compute_individual(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all individual metrics.

        Args:
            x (torch.Tensor): First input tensor
            y (torch.Tensor): Second input tensor

        Returns:
            Dict[str, torch.Tensor]: Individual metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(x, y)
        return results
