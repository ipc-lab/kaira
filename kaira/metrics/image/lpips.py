"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

LPIPS is a learned perceptual metric that leverages deep features and better correlates
with human perception than traditional metrics :cite:`zhang2018unreasonable`.
"""

from typing import Any, Literal, Tuple

import torch
import torchmetrics
from torch import Tensor
from torchmetrics.functional.image.lpips import _lpips_compute, _lpips_update

from ..base import BaseMetric
from ..registry import MetricRegistry


@MetricRegistry.register_metric("lpips")
class LearnedPerceptualImagePatchSimilarity(BaseMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) Module.

    LPIPS measures the perceptual similarity between images using deep features. Lower values
    indicate greater perceptual similarity. Unlike traditional metrics like PSNR and SSIM,
    LPIPS uses human perceptual judgments to calibrate a deep feature-based metric
    :cite:`zhang2018unreasonable`.
    """

    def __init__(self, net_type: Literal["vgg", "alex", "squeeze"] = "alex", normalize: bool = False, **kwargs: Any) -> None:
        """Initialize the LPIPS module.

        Args:
            net_type (str): The backbone network to use ('vgg', 'alex', or 'squeeze')
            normalize (bool): Whether to normalize the input images to [-1,1] range. If True, the input images
                should be in the range [0,1]. If False, the input images should be in the range [-1,1].
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="LPIPS")
        self.net_type = net_type
        self.normalize = normalize
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type, normalize=normalize, **kwargs)

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
        # Normalize inputs to the expected range if they exceed bounds
        if not self.normalize:
            # When normalize=False, inputs should be in range [-1, 1]
            img1 = torch.clamp(img1, -1.0, 1.0)
            img2 = torch.clamp(img2, -1.0, 1.0)
        else:
            # When normalize=True, inputs should be in range [0, 1]
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)

        result = self.lpips(img1, img2)
        return result.unsqueeze(0) if result.dim() == 0 else result

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update the internal state with a batch of samples.

        Args:
            img1 (Tensor): First batch of images
            img2 (Tensor): Second batch of images
        """
        # Normalize inputs to the expected range if they exceed bounds
        if not self.normalize:
            # When normalize=False, inputs should be in range [-1, 1]
            img1 = torch.clamp(img1, -1.0, 1.0)
            img2 = torch.clamp(img2, -1.0, 1.0)
        else:
            # When normalize=True, inputs should be in range [0, 1]
            img1 = torch.clamp(img1, 0.0, 1.0)
            img2 = torch.clamp(img2, 0.0, 1.0)

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
