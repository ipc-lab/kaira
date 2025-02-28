"""Base Loss module for Kaira.

This module contains the base loss class that all Kaira loss functions derive from.
"""

from abc import ABC

import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Base class for all Kaira loss functions.

    This abstract class defines the interface that all loss functions in Kaira must implement. It
    inherits from nn.Module to ensure compatibility with PyTorch's training pipeline.
    """

    pass
