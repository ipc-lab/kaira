"""Metrics module for Kaira.

This module contains various metrics for evaluating the performance of communication systems.
"""

from .base import BaseMetric
from .composite import CompositeMetric
from .image import (
    LPIPS,
    PSNR,
    SSIM,
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleSSIM,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from .signal import BER, BLER, SNR, BitErrorRate, BlockErrorRate, SignalToNoiseRatio
from .utils import compute_multiple_metrics, format_metric_results

__all__ = [
    # Base classes
    "BaseMetric",
    "CompositeMetric",
    # Image metrics
    "PeakSignalNoiseRatio",
    "PSNR",
    "StructuralSimilarityIndexMeasure",
    "SSIM",
    "MultiScaleSSIM",
    "LearnedPerceptualImagePatchSimilarity",
    "LPIPS",
    # Signal metrics
    "SignalToNoiseRatio",
    "SNR",
    "BitErrorRate",
    "BER",
    "BlockErrorRate",
    "BLER",
    # Registry
    "MetricRegistry",
    # Utils
    "compute_multiple_metrics",
    "format_metric_results",
]
