"""Learned Perceptual Image Patch Similarity (LPIPS) metric.

LPIPS is a learned perceptual metric that leverages deep features and better correlates
with human perception than traditional metrics :cite:`zhang2018unreasonable`.
"""

# Need to import inspect
import inspect
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

    def __init__(self, net_type: Literal["vgg", "alex", "squeeze"] = "alex", normalize: bool = False, *args: Any, **kwargs: Any) -> None:
        """Initialize the LPIPS module.

        Args:
            net_type (str): The backbone network to use ('vgg', 'alex', or 'squeeze')
            normalize (bool): Whether to normalize the input images to [-1,1] range. If True, the input images
                should be in the range [0,1]. If False, the input images should be in the range [-1,1].
            *args: Variable length argument list passed to the base class and torchmetrics.
            **kwargs: Arbitrary keyword arguments passed to the base class and torchmetrics.
        """
        # Remove name="LPIPS" as BaseMetric handles it
        super().__init__(*args, **kwargs)  # Pass args and kwargs
        self.net_type = net_type
        self.normalize = normalize
        # Pass only relevant kwargs to torchmetrics
        torchmetrics_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity.__init__).parameters}
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type, normalize=normalize, **torchmetrics_kwargs)

        self.register_buffer("sum_scores", torch.tensor(0.0))
        self.register_buffer("sum_sq", torch.tensor(0.0))
        self.register_buffer("total", torch.tensor(0))

    # Rename img1 to x and img2 to y to match BaseMetric
    def forward(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate LPIPS between two images.

        Args:
            x (Tensor): First batch of images
            y (Tensor): Second batch of images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).

        Returns:
            Tensor: LPIPS values for each sample
        """
        # Note: *args and **kwargs are not directly used by self.lpips call here
        # but are included for interface consistency.
        result = self.lpips(x, y)
        return result.unsqueeze(0) if result.dim() == 0 else result

    # Rename img1 to x and img2 to y to match BaseMetric
    def update(self, x: Tensor, y: Tensor, *args: Any, **kwargs: Any) -> None:
        """Update the internal state with a batch of samples.

        Args:
            x (Tensor): First batch of images
            y (Tensor): Second batch of images
            *args: Variable length argument list (currently unused).
            **kwargs: Arbitrary keyword arguments (currently unused).
        """
        # Note: *args and **kwargs are not directly used by _lpips_update call here
        # but are included for interface consistency.
        loss, total = _lpips_update(x, y, net=self.lpips.net, normalize=self.normalize)
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
