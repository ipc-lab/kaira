"""
This module contains the LambdaChannel class, which applies a given function to the input signal.
"""

import torch
from .base import BaseChannel

class LambdaChannel(BaseChannel):
    """Channel that applies a given function to the input signal.

    This channel applies a given function to the input signal. The function should take a single
    tensor as input and return a tensor of the same shape.

    Args:
        fn (callable): The function to apply to the input signal.
    """

    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)