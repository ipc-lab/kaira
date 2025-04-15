"""Structural Similarity Index Measure (SSIM) metrics.

SSIM is a perceptual metric that quantifies image quality degradation caused by
processing such as data compression or by losses in data transmission :cite:`wang2004image`.
MS-SSIM extends this concept to multiple scales :cite:`wang2003multiscale`.
"""

# Need to import inspect
import inspect
from typing import Any, Optional, Tuple

import torch
import torchmetrics
from pytorch_msssim import ms_ssim
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("ssim")
class StructuralSimilarityIndexMeasure(BaseMetric):
    """Structural Similarity Index Measure (SSIM) Module.

    SSIM measures the perceptual difference between two similar images. Values range from 0 to 1,
    where 1 means perfect similarity. The metric considers luminance, contrast, and structure
    to better match human visual perception :cite:`wang2004image` :cite:`brunet2011mathematical`.
    """

    def __init__(self, data_range: float = 1.0, kernel_size: int = 11, sigma: float = 1.5, reduction: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        """Initialize the SSIM module.

        Args:
            data_range (float): Range of the input data (typically 1.0 or 255)
            kernel_size (int): Size of the Gaussian kernel
            sigma (float): Standard deviation of the Gaussian kernel
            reduction (Optional[str]): Reduction method. The underlying torchmetrics implementation
                requires reduction=None, so this parameter controls post-processing reduction.
            *args: Variable length argument list passed to the base class and torchmetrics.
            **kwargs: Arbitrary keyword arguments passed to the base class and torchmetrics.
        """
        # Remove name="SSIM" as BaseMetric handles it
        super().__init__(*args, **kwargs)  # Pass args and kwargs
        self.reduction = reduction
        # Always use reduction=None in the underlying implementation
        # Pass only relevant kwargs to torchmetrics
        torchmetrics_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(torchmetrics.image.StructuralSimilarityIndexMeasure.__init__).parameters}
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=data_range, kernel_size=kernel_size, sigma=sigma, reduction=None, **torchmetrics_kwargs)

    # Rename preds to x and targets to y to match BaseMetric
    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate SSIM between predicted and target images.

        Args:
            x (Tensor): Predicted images
            y (Tensor): Target images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            Tensor: SSIM values for each sample or reduced according to reduction parameter
        """
        # Note: *args and **kwargs are not directly used by self.ssim call here
        # but are included for interface consistency.
        values = self.ssim(x, y)

        # Apply reduction if specified
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        else:
            # Ensure the tensor has at least one dimension when not reduced
            if values.dim() == 0:
                return values.unsqueeze(0)
            return values

    # Rename preds to x and targets to y to match BaseMetric
    def compute_with_stats(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """Compute SSIM with mean and standard deviation.

        Args:
            x (Tensor): Predicted images
            y (Tensor): Target images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of SSIM values
        """
        # Note: *args and **kwargs are not directly used here
        # but are included for interface consistency.
        values = self.forward(x, y)  # Use self.forward to handle reduction
        # Handle single value case to avoid NaN in std calculation
        if values.numel() <= 1:
            return values.mean(), torch.tensor(0.0)
        return values.mean(), values.std()


@MetricRegistry.register_metric("ms_ssim")
class MultiScaleSSIM(BaseMetric):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM) Module.

    This module calculates the MS-SSIM between two images. MS-SSIM is an extension of the SSIM
    metric that considers multiple scales to better capture perceptual similarity
    :cite:`wang2003multiscale`. It has been shown to correlate better with human perception
    than single-scale methods :cite:`wang2004image`.
    """

    def __init__(self, kernel_size: int = 11, data_range: float = 1.0, reduction: Optional[str] = None, weights: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any) -> None:
        """Initialize the MultiScaleSSIM module.

        Args:
            kernel_size (int): The size of the Gaussian kernel
            data_range (float): The range of the input data (typically 1.0 or 255)
            reduction (Optional[str]): Reduction method ('mean', 'sum', or None)
            weights (Optional[torch.Tensor]): Weights for different scales. Default is equal weighting.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        # Remove name="MS-SSIM" as BaseMetric handles it
        super().__init__(*args, **kwargs)  # Pass args and kwargs
        self.kernel_size = kernel_size
        self.data_range = data_range
        self.reduction = reduction
        self.weights = weights
        self.register_buffer("sum_values", torch.tensor(0.0))
        self.register_buffer("sum_sq", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))

    # Rename preds to x and targets to y to match BaseMetric
    def forward(self, x: torch.Tensor, y: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Calculate MS-SSIM between predicted and target images.

        Args:
            x (torch.Tensor): Predicted images
            y (torch.Tensor): Target images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            torch.Tensor: MS-SSIM values for each sample, or reduced according to reduction parameter
        """
        # Note: *args and **kwargs are not directly used by ms_ssim call here
        # but are included for interface consistency.
        values = ms_ssim(
            x,
            y,
            data_range=self.data_range,
            size_average=False,
            win_size=self.kernel_size,
            weights=self.weights,
        )

        # Apply reduction if specified
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        else:
            return values.unsqueeze(0) if values.dim() == 0 else values

    def update(self, preds: torch.Tensor, targets: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Update internal state with batch of samples.

        Args:
            preds (torch.Tensor): Predicted images
            targets (torch.Tensor): Target images
            *args: Variable length argument list passed to forward.
            **kwargs: Arbitrary keyword arguments passed to forward.
        """
        values = self.forward(preds, targets, *args, **kwargs)  # Pass args/kwargs
        if values.numel() == 0:
            return  # Avoid updating with empty values
        self.sum_values += values.sum()
        self.sum_sq += (values**2).sum()
        self.count += values.numel()

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute accumulated MS-SSIM statistics.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation
        """
        mean = self.sum_values / self.count
        std = torch.sqrt((self.sum_sq / self.count) - mean**2)
        return mean, std

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.sum_values.zero_()
        self.sum_sq.zero_()
        self.count.zero_()


# Alias for backward compatibility
SSIM = StructuralSimilarityIndexMeasure
