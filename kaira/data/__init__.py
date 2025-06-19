"""Data utilities for Kaira using HuggingFace datasets.

This module provides simplified HuggingFace datasets for generating data commonly used in
communication systems and information theory experiments.
"""

from .datasets import (
    BinaryTensorDataset,
    UniformTensorDataset,
    WynerZivCorrelationDataset,
)
from .sample_data import SampleImagesDataset, TorchVisionDataset, download_image

__all__ = [
    "BinaryTensorDataset",
    "UniformTensorDataset",
    "WynerZivCorrelationDataset",
    "SampleImagesDataset",
    "TorchVisionDataset",
    "download_image",
]
