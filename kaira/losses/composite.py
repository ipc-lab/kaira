"""Composite loss module for combining multiple loss functions.

This module provides functionality to create composite losses that combine
multiple individual loss functions with customizable weights. This is particularly
useful for cases where training requires optimizing multiple objectives
simultaneously, such as reconstruction loss combined with adversarial loss
or perceptual loss.

The composite approach addresses several common challenges in training:
- Different losses capture different aspects of the desired output
- Some applications require balancing multiple objectives
- Custom training schemes may need to emphasize certain properties over others
"""

from typing import Dict, Optional

import torch
from torch import nn

from .base import BaseLoss


class CompositeLoss(BaseLoss):
    """A loss that combines multiple loss functions with optional weighting.

    This class allows for the creation of custom loss functions by combining
    multiple individual losses with specified weights. It's useful when training
    requires optimizing multiple objectives simultaneously, such as combining
    pixel-wise reconstruction loss with perceptual or adversarial losses.

    The composite approach can balance the trade-offs between different loss terms.
    For example, L1 loss promotes pixel accuracy, while perceptual loss promotes
    visual quality. By combining them, you can achieve outputs that satisfy
    multiple criteria.

    Example:
        >>> from kaira.losses import L1Loss, SSIMLoss, PerceptualLoss
        >>> from kaira.losses.composite import CompositeLoss
        >>>
        >>> # Create individual losses
        >>> l1_loss = L1Loss()
        >>> ssim_loss = SSIMLoss()
        >>> perceptual_loss = PerceptualLoss()
        >>>
        >>> # Create a composite loss with custom weights
        >>> losses = {"l1": l1_loss, "ssim": ssim_loss, "perceptual": perceptual_loss}
        >>> weights = {"l1": 1.0, "ssim": 0.5, "perceptual": 0.1}
        >>> composite_loss = CompositeLoss(losses=losses, weights=weights)
        >>>
        >>> # Train a model with the composite loss
        >>> output = model(input_data)
        >>> loss = composite_loss(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(self, losses: Dict[str, BaseLoss], weights: Optional[Dict[str, float]] = None):
        """Initialize composite loss with component losses and their weights.

        Args:
            losses (Dict[str, BaseLoss]): Dictionary mapping loss names to loss objects.
                Each loss should be a subclass of BaseLoss.
            weights (Optional[Dict[str, float]]): Dictionary mapping loss names to their
                relative importance. If None, equal weights are assigned to all losses.
                Weights are automatically normalized to sum to 1.0.

        Raises:
            ValueError: If weights dictionary contains keys not present in losses dictionary.
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)

        # Validate weights
        if weights is not None:
            for name in weights:
                if name not in losses:
                    raise ValueError(f"Weight key '{name}' not found in losses dictionary")

        self.weights = weights or {name: 1.0 for name in losses}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the weighted combination of all component losses.

        Evaluates each loss on the input tensors and combines them according
        to the normalized weights specified during initialization.

        Args:
            x (torch.Tensor): First input tensor, typically the prediction or generated output
            target (torch.Tensor): Second input tensor, typically the target or ground truth

        Returns:
            torch.Tensor: Weighted sum of all loss values as a single scalar tensor.
        """
        result = torch.tensor(0.0, device=x.device)
        for name, loss in self.losses.items():
            if name in self.weights:
                loss_value = loss(x, target)
                if isinstance(loss_value, tuple):
                    loss_value = loss_value[0]  # Take first value if tuple
                result = result + self.weights[name] * loss_value
        return result

    def get_individual_losses(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all individual losses separately without combining them.

        This method is an alias for compute_individual for backward compatibility.

        Args:
            x (torch.Tensor): First input tensor, typically the prediction or generated output
            target (torch.Tensor): Second input tensor, typically the target or ground truth

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping loss names to their computed values.
        """
        return self.compute_individual(x, target)

    def compute_individual(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all individual losses separately without combining them.

        This method is useful for debugging and monitoring individual loss components
        during training.

        Args:
            x (torch.Tensor): First input tensor, typically the prediction or generated output
            target (torch.Tensor): Second input tensor, typically the target or ground truth

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping loss names to their computed values.
        """
        results = {}
        for name, loss in self.losses.items():
            results[name] = loss(x, target)
        return results

    def add_loss(self, name: str, loss, weight: float = 1.0):
        """Add a new loss to the composite loss.

        Args:
            name (str): Name for the loss
            loss (BaseLoss): Loss module to add
            weight (float): Weight for the new loss (will be preserved exactly as provided)
        Returns:
            None: Updates the loss and weight dictionaries in-place
        Raises:
            ValueError: If a loss with the given name already exists
        """

        # Check if loss name already exists
        if name in self.losses:
            raise ValueError(f"Loss '{name}' already exists in the composite loss")

        # Add loss to ModuleDict
        self.losses[name] = loss

        # Preserve the exact weight for the new loss, adjust existing weights proportionally
        remaining_weight = 1.0 - weight
        current_total = sum(self.weights.values())

        # Adjust existing weights to maintain the sum of 1.0
        if current_total > 0:  # Avoid division by zero
            for k in self.weights:
                self.weights[k] = self.weights[k] * remaining_weight / current_total

        # Set the weight for the new loss
        self.weights[name] = weight

    def __str__(self) -> str:
        """Get a string representation of the composite loss with weights.

        Returns:
            str: String representation of the composite loss including components and weights
        """
        base_str = f"{self.__class__.__name__}(\n"
        base_str += "  (losses): ModuleDict(\n"

        for name, loss in self.losses.items():
            weight = self.weights.get(name, 0.0)
            base_str += f"    ({name}): {loss.__class__.__name__}  # weight={weight:.3f}\n"

        base_str += "  )\n)"
        return base_str
