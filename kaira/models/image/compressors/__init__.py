"""Image compressor models, including standard and neural network-based methods."""

from .bpg import BPGCompressor
from .neural import NeuralCompressor

__all__ = ["BPGCompressor", "NeuralCompressor"]
