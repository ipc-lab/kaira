"""Metrics module for Kaira.

This module contains various metrics for evaluating the performance of communication systems, such
as image quality assessment metrics.
"""

from typing import Any, Literal

import torch
import torchmetrics
from pytorch_msssim import ms_ssim
from torch import Tensor
from torchmetrics import MeanMetric
from torchmetrics.functional.image.lpips import _lpips_compute, _lpips_update
from torchmetrics.image.inception import InceptionScore, Tuple


# A metric class that computes the multi-scale structural similarity index measure (SSIM) between two images.
class MultiScaleSSIM(MeanMetric):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM) Module.

    This module calculates the MS-SSIM between two images. MS-SSIM is an extension of the SSIM
    metric that considers multiple scales to better capture perceptual similarity.
    """

    def __init__(self, kernel_size=11, data_range=1.0, **kwargs: Any) -> None:
        """Initialize the MultiScaleSSIM module.

        Args:
            kernel_size (int): The size of the Gaussian kernel.
            data_range (float): The range of the input data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__("warn", **kwargs)

        self.kernel_size = kernel_size
        self.data_range = data_range

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update the internal state of the metric with new data.

        Args:
            preds (torch.Tensor): The predicted images.
            targets (torch.Tensor): The target images.

        Returns:
            torch.Tensor: The updated metric value.
        """
        value = ms_ssim(
            preds, targets, data_range=1.0, size_average=False, win_size=self.kernel_size
        )

        return super().update(value, 1)


class LearnedPerceptualImagePatchSimilarity(
    torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity
):
    """Learned Perceptual Image Patch Similarity (LPIPS) Module."""

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        normalize: bool = False,
        **kwargs: Any
    ) -> None:
        """Initialize the LearnedPerceptualImagePatchSimilarity module.

        Args:
            net_type (str): The type of network to use.
            normalize (bool): Whether to normalize the input images.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(net_type, normalize=normalize, reduction="mean", **kwargs)

        self.add_state("sum_sq", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update the internal state of the metric with new data.

        Args:
            img1 (Tensor): The first image.
            img2 (Tensor): The second image.
        """
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)

        self.sum_scores += loss.sum()
        self.total += total

        self.sum_sq += (loss**2).sum()

    def compute(self) -> Tensor:
        """Compute the final LPIPS value.

        Returns:
            Tensor: The LPIPS value.
        """
        mean = _lpips_compute(self.sum_scores, self.total, "mean")
        std = torch.sqrt((self.sum_sq / self.total) - mean**2)

        return mean, std


class PeakSignalNoiseRatio(torchmetrics.image.PeakSignalNoiseRatio):
    """Peak Signal-to-Noise Ratio (PSNR) Module."""

    def __init__(self, *args, **kwargs):
        """Initialize the PeakSignalNoiseRatio module."""
        super().__init__(*args, reduction=None, dim=[1, 2, 3], **kwargs)

    def compute(self):
        """Compute the final PSNR value.

        Returns:
            Tensor: The PSNR value.
        """
        res_per_sample = super().compute()
        return res_per_sample.mean(), res_per_sample.std()


# A metric class that computes the structural similarity index measure (SSIM) between two images.
class StructuralSimilarityIndexMeasure(torchmetrics.image.StructuralSimilarityIndexMeasure):
    """Structural Similarity Index Measure (SSIM) Module."""

    def __init__(self, *args, **kwargs):
        """Initialize the StructuralSimilarityIndexMeasure module."""
        super().__init__(*args, reduction=None, **kwargs)

    def compute(self):
        """Compute the final SSIM value.

        Returns:
            Tensor: The SSIM value.
        """
        res_per_sample = super().compute()

        return res_per_sample.mean(), res_per_sample.std()
