"""Losses module for Kaira.

This module contains various loss functions for training communication systems, including MSE loss,
LPIPS loss, and SSIM loss.
"""

import torch
import torch.nn as nn

from kaira.metrics import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
)


class MSELoss(nn.Module):
    """Mean Squared Error (MSE) Loss Module.

    This module calculates the MSE loss between the input and the target.
    """

    def __init__(self):
        """Initialize the MSELoss module."""
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MSELoss module.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The MSE loss between the input and the target.
        """
        return self.mse(x, target)


class CombinedLoss(nn.Module):
    """Combined Loss Module.

    This module combines multiple loss functions into a single loss function.
    """

    def __init__(self, losses: nn.ModuleList, weights: list[float]):
        """Initialize the CombinedLoss module.

        Args:
            losses (nn.ModuleList): A list of loss functions to combine.
            weights (list[float]): A list of weights for each loss function.
        """
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CombinedLoss module.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The combined loss between the input and the target.
        """
        loss = 0
        for i, l in enumerate(self.losses):
            loss += self.weights[i] * l(x, target)
        return loss


class MSELPIPSLoss(nn.Module):
    """MSELPIPSLoss Module."""

    def __init__(self):
        """Initialize the MSELPIPSLoss module."""
        super().__init__()
        raise NotImplementedError


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) Loss Module.

    This module calculates the LPIPS loss between the input and the target.
    """

    def __init__(self):
        """Initialize the LPIPSLoss module."""
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LPIPSLoss module.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The LPIPS loss between the input and the target.
        """
        return self.lpips(x, target)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) Loss Module.

    This module calculates the SSIM loss between the input and the target.
    """

    def __init__(self):
        """Initialize the SSIMLoss module."""
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SSIMLoss module.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The SSIM loss between the input and the target.
        """
        return 1 - self.ssim(x, target)


class MSSSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM) Loss Module.

    This module calculates the MS-SSIM loss between the input and the target.
    """

    def __init__(self):
        """Initialize the MSSSIMLoss module."""
        super().__init__()
        self.ms_ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MSSSIMLoss module.

        Args:
            x (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The MS-SSIM loss between the input and the target.
        """
        return 1 - self.ms_ssim(x, target)
