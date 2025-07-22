"""Kaira Losses Package.

This package provides various loss functions for different modalities.
"""

from . import image
from .base import BaseLoss
from .composite import CompositeLoss
from .registry import LossRegistry

__all__ = ["image", "BaseLoss", "CompositeLoss", "LossRegistry"]
