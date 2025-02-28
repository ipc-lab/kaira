"""Composite metrics module for combining multiple evaluation metrics.

This module provides functionality to create composite metrics that combine
multiple individual metrics with customizable weights. This is particularly
useful for cases where evaluation quality is better represented by a blend
of different metrics rather than a single measurement.

The composite approach addresses several common challenges in evaluation:
- Different metrics capture different aspects of similarity/quality
- Some applications require balancing perceptual quality with pixel accuracy
- Custom evaluation schemes may need to emphasize certain properties over others
"""

from typing import Dict, Optional

import torch
from torch import nn

from .base import BaseMetric


class CompositeMetric(BaseMetric):
    """A metric that combines multiple metrics with optional weighting.

    This class allows for the creation of custom evaluation metrics by combining
    multiple individual metrics with specified weights. It's useful when a single
    metric doesn't capture all the desired qualities of a comparison, such as
    combining perceptual and statistical image similarity measures.

    The composite approach can balance the trade-offs between different metrics.
    For example, PSNR tends to favor smoothness, while perceptual metrics may
    favor visual sharpness. By combining them, you can create more balanced
    evaluation criteria.

    Note:
        When combining metrics where some are "higher is better" and others are
        "lower is better", you may need to invert certain metrics (e.g., by using
        negative weights or transforming the metric beforehand).

    Example:
        >>> from kaira.metrics import PSNR, SSIM, LPIPS
        >>> from kaira.metrics.composite import CompositeMetric
        >>>
        >>> # Create individual metrics
        >>> psnr = PSNR()
        >>> ssim = SSIM()
        >>> lpips = LPIPS()
        >>>
        >>> # Create a composite metric with custom weights
        >>> # Note: LPIPS is "lower is better" while PSNR and SSIM are "higher is better"
        >>> metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
        >>> weights = {"psnr": 0.3, "ssim": 0.3, "lpips": -0.4}  # Negative weight for LPIPS
        >>> composite = CompositeMetric(metrics=metrics, weights=weights)
        >>>
        >>> # Evaluate images
        >>> score = composite(prediction, target)
        >>> individual_scores = composite.compute_individual(prediction, target)
    """

    def __init__(self, metrics: Dict[str, BaseMetric], weights: Optional[Dict[str, float]] = None):
        """Initialize composite metric with component metrics and their weights.

        Args:
            metrics (Dict[str, BaseMetric]): Dictionary mapping metric names to metric objects.
                Each metric should be a subclass of BaseMetric.
            weights (Optional[Dict[str, float]]): Dictionary mapping metric names to their
                relative importance. If None, equal weights are assigned to all metrics.
                Weights are automatically normalized to sum to 1.0.

                Use negative weights for metrics where lower values indicate better quality
                (e.g., LPIPS, MSE) when combining with metrics where higher values indicate
                better quality (e.g., PSNR, SSIM).
        """
        super().__init__(name="CompositeMetric")
        self.metrics = nn.ModuleDict(metrics)
        self.weights = weights or {name: 1.0 for name in metrics}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the weighted combination of all component metrics.

        Evaluates each metric on the input tensors and combines them according
        to the normalized weights specified during initialization.

        Note:
            If a metric returns a tuple (e.g., containing mean and std), only the
            first element (typically the mean) is used in the weighted combination.
            For more control, access individual metrics through compute_individual().

        Args:
            x (torch.Tensor): First input tensor, typically the prediction or generated output
            y (torch.Tensor): Second input tensor, typically the target or ground truth

        Returns:
            torch.Tensor: Weighted sum of all metric values as a single scalar tensor.
                The interpretation of this value depends on the constituent metrics and weights.
                With appropriate weighting, higher values typically indicate better results.
        """
        result = torch.tensor(0.0, device=x.device)
        for name, metric in self.metrics.items():
            if name in self.weights:
                metric_value = metric(x, y)
                if isinstance(metric_value, tuple):
                    metric_value = metric_value[0]  # Take mean if tuple of (mean, std)
                result = result + self.weights[name] * metric_value
        return result

    def compute_individual(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all individual metrics separately without combining them.

        Unlike the forward method which returns a weighted combination, this method
        returns the raw value for each individual metric. This is useful for:
        - Debugging the contribution of individual metrics
        - Creating custom visualizations or reports
        - Applying post-processing to individual metrics before combining them
        - Evaluating metrics with different criteria that cannot be combined directly

        Args:
            x (torch.Tensor): First input tensor, typically the prediction or generated output
            y (torch.Tensor): Second input tensor, typically the target or ground truth

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping metric names to their computed values.
                May contain tuple values (e.g., mean and std) for metrics that return multiple values.
                The interpretation of values (higher/lower is better) depends on the specific metric.
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(x, y)
        return results
