"""Peak Signal-to-Noise Ratio (PSNR) metric.

PSNR is a widely used objective quality metric for image and video processing
:cite:`hore2010image` :cite:`huynh2008scope`. Despite its limitations in perceptual
correlation, it remains one of the most common benchmarks for image quality assessment.
"""

import inspect
from typing import Any, Optional, Tuple

import torch
import torchmetrics.image
from torch import Tensor

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("psnr")
class PeakSignalNoiseRatio(BaseMetric):
    """Peak Signal-to-Noise Ratio (PSNR) Module.

    PSNR measures the ratio between the maximum possible power of a signal and the power of
    corrupting noise that affects the quality of its representation. Higher values indicate better
    quality :cite:`hore2010image`. While PSNR doesn't perfectly correlate with human perception,
    it is widely used for its simplicity and clear physical meaning :cite:`wang2009mean`.
    """

    def __init__(self, data_range: float = 1.0, reduction: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        """Initialize the PeakSignalNoiseRatio module.

        Args:
            data_range (float): The range of the input data (typically 1.0 or 255)
            reduction (Optional[str]): Reduction method. The underlying torchmetrics implementation
                requires reduction=None, so this parameter controls post-processing reduction.
            *args: Variable length argument list passed to the base class and torchmetrics.
            **kwargs: Arbitrary keyword arguments passed to the base class and torchmetrics.
        """
        # Remove name="PSNR" as BaseMetric handles it
        super().__init__(*args, **kwargs)  # Pass args and kwargs
        self.reduction = reduction
        if "dim" not in kwargs:
            kwargs["dim"] = [1, 2, 3]
        # Always use reduction=None in the underlying implementation
        # Pass only relevant kwargs to torchmetrics
        torchmetrics_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(torchmetrics.image.PeakSignalNoiseRatio.__init__).parameters}
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range, reduction=None, **torchmetrics_kwargs)

    # Rename preds to x and targets to y to match BaseMetric
    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate PSNR between predicted and target images.

        Args:
            x (Tensor): Predicted images
            y (Tensor): Target images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).
        Returns:
            Tensor: PSNR values for each sample or reduced according to reduction parameter
        """
        # Note: *args and **kwargs are not directly used by self.psnr call here
        # but are included for interface consistency.
        values = self.psnr(x, y)

        # Apply reduction if specified
        if self.reduction == "mean":
            return values.mean()
        elif self.reduction == "sum":
            return values.sum()
        else:
            return values

    # Rename preds to x and targets to y to match BaseMetric
    def compute_with_stats(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """Compute PSNR with mean and standard deviation.

        Args:
            x (Tensor): Predicted images
            y (Tensor): Target images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).
        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of PSNR values
        """
        # Note: *args and **kwargs are not directly used by self.psnr call here
        # but are included for interface consistency.
        values = self.psnr(x, y)
        # Handle single value case to avoid NaN in std calculation
        if values.numel() <= 1:
            return values.mean(), torch.tensor(0.0)

        return values.mean(), values.std()


PSNR = PeakSignalNoiseRatio
