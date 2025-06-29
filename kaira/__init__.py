"""Kaira is an open-source simulation toolkit for wireless communications built on PyTorch.

It provides a modular, user-friendly platform for developing, testing, and benchmarking advanced
communications algorithms—including deep learning-based approaches such as deep joint source-
channel coding (DeepJSCC). Designed to accelerate research and innovation, Kaira integrates
seamlessly with existing PyTorch projects, supporting rapid prototyping of novel communication
strategies.
"""

import os

# Import configs from top-level configs directory
import sys

from . import (
    channels,
    constraints,
    data,
    losses,
    metrics,
    models,
    modulations,
    training,
    utils,
)
from .version import __version__

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

__all__ = [
    "__version__",
    "channels",
    "constraints",
    "data",
    "losses",
    "metrics",
    "models",
    "modulations",
    "training",
    "utils",
]
