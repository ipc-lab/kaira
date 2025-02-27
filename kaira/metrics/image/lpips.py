"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

from typing import Literal, Any, Tuple

import torch
from torch import Tensor
import torchmetrics
from torchmetrics.functional.image.lpips import _lpips_compute, _lpips_update

from ..base import BaseMetric
from ..registry import register_metric


@register_metric("lpips")
class LearnedPerceptualImagePatchSimilarity(BaseMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) Module.
    
    LPIPS measures the perceptual similarity between images using deep features.
    Lower values indicate greater perceptual similarity.
    """

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        normalize: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the LPIPS module.
        
        Args:
            net_type (str): The backbone network to use ('vgg', 'alex', or 'squeeze')
            normalize (bool): Whether to normalize the input images
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="LPIPS")
        self.net_type = net_type
        self.normalize = normalize
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type, 
            normalize=normalize, 
            reduction=None, 
            **kwargs
        )
        
        self.register_buffer("sum_scores", torch.tensor(0.0))
        self.register_buffer("sum_sq", torch.tensor(0.0))
        self.register_buffer("total", torch.tensor(0))

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Calculate LPIPS between two images.
        
        Args:
            img1 (Tensor): First batch of images
            img2 (Tensor): Second batch of images
            
        Returns:
            Tensor: LPIPS values for each sample
        """
        return self.lpips(img1, img2)

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update the internal state with a batch of samples.
        
        Args:
            img1 (Tensor): First batch of images
            img2 (Tensor): Second batch of images
        """
        loss, total = _lpips_update(img1, img2, net=self.lpips.net, normalize=self.normalize)
        self.sum_scores += loss.sum()
        self.total += total
        self.sum_sq += (loss**2).sum()

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute the accumulated LPIPS statistics.
        
        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of LPIPS values
        """
        mean = _lpips_compute(self.sum_scores, self.total, "mean")
        std = torch.sqrt((self.sum_sq / self.total) - mean**2)
        return mean, std
    
    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.sum_scores.zero_()
        self.sum_sq.zero_()
        self.total.zero_()


# Alias for backward compatibility
LPIPS = LearnedPerceptualImagePatchSimilarity
