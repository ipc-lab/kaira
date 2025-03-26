"""Metrics module for Kaira.

This module contains various metrics for evaluating the performance of communication systems.
"""

from . import utils
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
from .registry import MetricRegistry
from .signal import (
    BER,
    BLER,
    FER,
    SER,
    SNR,
    BitErrorRate,
    BlockErrorRate,
    FrameErrorRate,
    SignalToNoiseRatio,
    SymbolErrorRate,
)

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
    "FrameErrorRate",
    "FER",
    "SymbolErrorRate",
    "SER",
    # Registry
    "MetricRegistry",
    # Utils
    "utils",
]
