"""Peak Signal-to-Noise Ratio (PSNR) metric."""

from typing import Tuple, Any

import torch
from torch import Tensor
import torchmetrics.image

from ..base import BaseMetric
from ..registry import register_metric


@register_metric("psnr")
class PeakSignalNoiseRatio(BaseMetric):
    """Peak Signal-to-Noise Ratio (PSNR) Module.
    
    PSNR measures the ratio between the maximum possible power of a signal
    and the power of corrupting noise that affects the quality of its representation.
    Higher values indicate better quality.
    """

    def __init__(self, data_range: float = 1.0, **kwargs: Any) -> None:
        """Initialize the PeakSignalNoiseRatio module.
        
        Args:
            data_range (float): The range of the input data (typically 1.0 or 255)
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="PSNR")
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(
            data_range=data_range, 
            reduction=None,
            dim=[1, 2, 3],
            **kwargs
        )

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Calculate PSNR between predicted and target images.
        
        Args:
            preds (Tensor): Predicted images
            targets (Tensor): Target images
            
        Returns:
            Tensor: PSNR values for each sample
        """
        return self.psnr(preds, targets)
    
    def compute_with_stats(self, preds: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute PSNR with mean and standard deviation.
        
        Args:
            preds (Tensor): Predicted images
            targets (Tensor): Target images
            
        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of PSNR values
        """
        values = self.forward(preds, targets)
        return values.mean(), values.std()


# Alias for backward compatibility
PSNR = PeakSignalNoiseRatio
