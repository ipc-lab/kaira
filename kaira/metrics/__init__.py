"""Metrics module for Kaira.

This module contains various metrics for evaluating the performance of communication systems.
"""

from .base import BaseMetric
from .factories import (
    CompositeMetric,
    create_composite_metric,
    create_image_quality_metrics,
)
from .image import (
    LPIPS,
    PSNR,
    SSIM,
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleSSIM,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from .registry import create_metric, list_metrics, register_metric
from .signal import BER, BLER, SNR, BitErrorRate, BlockErrorRate, SignalToNoiseRatio
from .utils import compute_multiple_metrics, format_metric_results

__all__ = [
    # Base classes
    "BaseMetric",
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
    "register_metric",
    "create_metric",
    "list_metrics",
    # Factories
    "create_image_quality_metrics",
    "create_composite_metric",
    "CompositeMetric",
    # Utils
    "compute_multiple_metrics",
    "format_metric_results",
]
