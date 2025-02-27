"""Kaira Losses Package.

This package provides various loss functions for different modalities.
"""

from . import adversarial, audio, image, multimodal, text

__all__ = ["image", "audio", "text", "multimodal", "adversarial"]
