"""Image compressor models, including standard and neural network-based methods."""

from .base import BaseImageCompressor
from .bpg import BPGCompressor
from .jpeg import JPEGCompressor
from .jpeg2000 import JPEG2000Compressor
from .jpegxl import JPEGXLCompressor
from .neural import NeuralCompressor
from .png import PNGCompressor
from .webp import WebPCompressor

__all__ = [
    "BaseImageCompressor",
    "BPGCompressor",
    "JPEG2000Compressor",
    "JPEGCompressor",
    "JPEGXLCompressor",
    "NeuralCompressor",
    "PNGCompressor",
    "WebPCompressor",
]
