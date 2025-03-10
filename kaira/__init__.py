"""Kaira is an open-source simulation toolkit for wireless communications built on PyTorch.

It provides a modular, user-friendly platform for developing, testing, and benchmarking advanced
wireless transmission algorithms—including deep learning-based approaches such as deep joint
source-channel coding (DeepJSCC). Designed to accelerate research and innovation, Kaira integrates
seamlessly with existing PyTorch projects, supporting rapid prototyping of novel communication
strategies.
"""
from . import channels, constraints, metrics, models, modulations
from .version import __version__

__all__ = [
    "__version__",
    "channels",
    "constraints",
    "metrics",
    "models",
    "modulations",
]
