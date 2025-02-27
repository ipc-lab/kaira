"""Kaira Losses Package.

This package provides various loss functions for different modalities.
"""

from . import image, audio, text, multimodal, adversarial

__all__ = [
    "image",
    "audio",
    "text",
    "multimodal",
    "adversarial"
]