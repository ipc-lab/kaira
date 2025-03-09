"""Kaira Losses Package.

This package provides various loss functions for different modalities.
"""

from . import adversarial, audio, image, multimodal, text
from .base import BaseLoss
from .composite import CompositeLoss
from .registry import LossRegistry

__all__ = ["image", "audio", "text", "multimodal", "adversarial", "BaseLoss", "CompositeLoss", "LossRegistry"]
