"""Peak Signal-to-Noise Ratio (PSNR) metric.

PSNR is a widely used objective quality metric for image and video processing
:cite:`hore2010image` :cite:`huynh2008scope`. Despite its limitations in perceptual
correlation, it remains one of the most common benchmarks for image quality assessment.
"""

from typing import Any, Tuple

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

    def __init__(self, data_range: float = 1.0, **kwargs: Any) -> None:
        """Initialize the PeakSignalNoiseRatio module.

        Args:
            data_range (float): The range of the input data (typically 1.0 or 255)
            **kwargs: Additional keyword arguments
        """
        super().__init__(name="PSNR")
        self.psnr = torchmetrics.image.PeakSignalNoiseRatio(data_range=data_range, reduction=None, dim=[1, 2, 3], **kwargs)

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


PSNR = PeakSignalNoiseRatio
