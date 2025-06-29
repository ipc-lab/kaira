"""Data utilities for Kaira.

This module provides simple and efficient dataset classes for communication systems and information
theory experiments. All datasets are memory-efficient and generate data on-demand.
"""

from .datasets import (
    BinaryDataset,
    CorrelatedDataset,
    FunctionDataset,
    GaussianDataset,
    UniformDataset,
)
from .sample_data import ImageDataset

__all__ = [
    "BinaryDataset",
    "UniformDataset",
    "GaussianDataset",
    "CorrelatedDataset",
    "FunctionDataset",
    "ImageDataset",
]
