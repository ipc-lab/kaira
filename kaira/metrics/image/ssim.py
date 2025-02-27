"""Structural Similarity Index Measure (SSIM) metrics."""

from typing import Any, Tuple

import torch
from torch import Tensor
import torchmetrics
from pytorch_msssim import ms_ssim

from ..base import BaseMetric
from ..registry import register_metric


@register_metric("ssim")
class StructuralSimilarityIndexMeasure(BaseMetric):
    """Structural Similarity Index Measure (SSIM) Module.
    
    SSIM measures the perceptual difference between two similar images.
    Values range from 0 to 1, where 1 means perfect similarity.
    """

    def __init__(
        self, 
        data_range: float = 1.0, 
        kernel_size: int = 11, 
        sigma: float = 1.5, 
        **kwargs: Any
    ) -> None:
        """Initialize the SSIM module.
        
        Args:
            data_range (float): Range of the input data (typically 1.0 or 255)
            kernel_size (int): Size of the Gaussian kernel
            sigma (float): Standard deviation of the Gaussian kernel
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="SSIM")
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=kernel_size,
            sigma=sigma,
            reduction=None,
            **kwargs
        )

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Calculate SSIM between predicted and target images.
        
        Args:
            preds (Tensor): Predicted images
            targets (Tensor): Target images
            
        Returns:
            Tensor: SSIM values for each sample
        """
        return self.ssim(preds, targets)
    
    def compute_with_stats(self, preds: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute SSIM with mean and standard deviation.
        
        Args:
            preds (Tensor): Predicted images
            targets (Tensor): Target images
            
        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of SSIM values
        """
        values = self.forward(preds, targets)
        return values.mean(), values.std()


@register_metric("ms_ssim")
class MultiScaleSSIM(BaseMetric):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM) Module.

    This module calculates the MS-SSIM between two images. MS-SSIM is an extension of the SSIM
    metric that considers multiple scales to better capture perceptual similarity.
    """

    def __init__(self, kernel_size: int = 11, data_range: float = 1.0, **kwargs: Any) -> None:
        """Initialize the MultiScaleSSIM module.

        Args:
            kernel_size (int): The size of the Gaussian kernel
            data_range (float): The range of the input data (typically 1.0 or 255)
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="MS-SSIM")
        self.kernel_size = kernel_size
        self.data_range = data_range
        self.register_buffer("sum_values", torch.tensor(0.0))
        self.register_buffer("sum_sq", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate MS-SSIM between predicted and target images.
        
        Args:
            preds (torch.Tensor): Predicted images
            targets (torch.Tensor): Target images
            
        Returns:
            torch.Tensor: MS-SSIM values for each sample
        """
        return ms_ssim(
            preds, 
            targets, 
            data_range=self.data_range, 
            size_average=False, 
            win_size=self.kernel_size
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update internal state with batch of samples.
        
        Args:
            preds (torch.Tensor): Predicted images
            targets (torch.Tensor): Target images
        """
        values = self.forward(preds, targets)
        self.sum_values += values.sum()
        self.sum_sq += (values ** 2).sum()
        self.count += values.numel()

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute accumulated MS-SSIM statistics.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation
        """
        mean = self.sum_values / self.count
        std = torch.sqrt((self.sum_sq / self.count) - mean ** 2)
        return mean, std

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.sum_values.zero_()
        self.sum_sq.zero_()
        self.count.zero_()


# Alias for backward compatibility
SSIM = StructuralSimilarityIndexMeasure
