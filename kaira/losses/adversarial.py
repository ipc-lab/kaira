"""Adversarial Losses module for Kaira.

This module contains various adversarial loss functions for GAN-based training.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss
from .registry import LossRegistry


@LossRegistry.register_loss()
class VanillaGANLoss(BaseLoss):
    """Vanilla GAN Loss Module.

    This module implements the original GAN loss from Goodfellow et al. 2014.
    """

    def __init__(self, reduction="mean"):
        """Initialize the VanillaGANLoss module.

        Args:
            reduction (str): Reduction method ('mean', 'sum', or 'none'). Default is 'mean'.
        """
        super().__init__()
        self.reduction = reduction

    def forward_discriminator(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Forward pass for discriminator.

        Args:
            real_logits (torch.Tensor): Discriminator outputs for real data.
            fake_logits (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits), reduction=self.reduction)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits), reduction=self.reduction)
        return real_loss + fake_loss

    def forward_generator(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Forward pass for generator.

        Args:
            fake_logits (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Generator loss.
        """
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits), reduction=self.reduction)

    def forward(self, discriminator_pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """Forward pass through the VanillaGANLoss module.

        Args:
            discriminator_pred (torch.Tensor): Discriminator outputs.
            is_real (bool): Whether predictions are for real data.

        Returns:
            torch.Tensor: The GAN loss.
        """
        target = torch.ones_like(discriminator_pred) if is_real else torch.zeros_like(discriminator_pred)
        return F.binary_cross_entropy_with_logits(discriminator_pred, target, reduction=self.reduction)


@LossRegistry.register_loss()
class LSGANLoss(BaseLoss):
    """Least Squares GAN Loss Module.

    This module implements the LSGAN loss from Mao et al. 2017.
    """

    def __init__(self, reduction="mean"):
        """Initialize the LSGANLoss module.

        Args:
            reduction (str): Reduction method ('mean', 'sum', or 'none'). Default is 'mean'.
        """
        super().__init__()
        self.reduction = reduction

    def forward_discriminator(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for discriminator.

        Args:
            real_pred (torch.Tensor): Discriminator outputs for real data.
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        real_loss = torch.mean((real_pred - 1) ** 2)
        fake_loss = torch.mean(fake_pred**2)
        return (real_loss + fake_loss) * 0.5

    def forward_generator(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for generator.

        Args:
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Generator loss.
        """
        return torch.mean((fake_pred - 1) ** 2)

    def forward(self, pred: torch.Tensor, is_real: bool, for_discriminator: bool = True) -> torch.Tensor:
        """Forward pass through the LSGANLoss module.

        Args:
            pred (torch.Tensor): Discriminator outputs.
            is_real (bool): Whether predictions are for real data.
            for_discriminator (bool): Whether calculating loss for discriminator. Default is True.

        Returns:
            torch.Tensor: The LSGAN loss.
        """
        if for_discriminator:
            if is_real:
                return torch.mean((pred - 1) ** 2)
            else:
                return torch.mean(pred**2)
        else:  # for generator
            return torch.mean((pred - 1) ** 2)


@LossRegistry.register_loss()
class WassersteinGANLoss(BaseLoss):
    """Wasserstein GAN Loss Module.

    This module implements the WGAN loss from Arjovsky et al. 2017.
    """

    def __init__(self):
        """Initialize the WassersteinGANLoss module."""
        super().__init__()

    def forward_discriminator(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for discriminator.

        Args:
            real_pred (torch.Tensor): Discriminator outputs for real data.
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        return -(torch.mean(real_pred) - torch.mean(fake_pred))

    def forward_generator(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for generator.

        Args:
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Generator loss.
        """
        return -torch.mean(fake_pred)

    def forward(self, pred: torch.Tensor, is_real: bool, for_discriminator: bool = True) -> torch.Tensor:
        """Forward pass through the WassersteinGANLoss module.

        Args:
            pred (torch.Tensor): Discriminator outputs.
            is_real (bool): Whether predictions are for real data.
            for_discriminator (bool): Whether calculating loss for discriminator. Default is True.

        Returns:
            torch.Tensor: The Wasserstein loss.
        """
        if for_discriminator:
            if is_real:
                return -torch.mean(pred)
            else:
                return torch.mean(pred)
        else:  # for generator
            return -torch.mean(pred)


@LossRegistry.register_loss()
class HingeLoss(BaseLoss):
    """Hinge Loss Module for GANs.

    This module implements the hinge loss commonly used in spectral normalization GAN.
    """

    def __init__(self):
        """Initialize the HingeLoss module."""
        super().__init__()

    def forward_discriminator(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for discriminator.

        Args:
            real_pred (torch.Tensor): Discriminator outputs for real data.
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Discriminator loss.
        """
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss

    def forward_generator(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass for generator.

        Args:
            fake_pred (torch.Tensor): Discriminator outputs for fake data.

        Returns:
            torch.Tensor: Generator loss.
        """
        return -fake_pred.mean()

    def forward(self, pred: torch.Tensor, is_real: bool, for_discriminator: bool = True) -> torch.Tensor:
        """Forward pass through the HingeLoss module.

        Args:
            pred (torch.Tensor): Discriminator outputs.
            is_real (bool): Whether predictions are for real data.
            for_discriminator (bool): Whether calculating loss for discriminator. Default is True.

        Returns:
            torch.Tensor: The hinge loss.
        """
        if for_discriminator:
            if is_real:
                return F.relu(1.0 - pred).mean()
            else:
                return F.relu(1.0 + pred).mean()
        else:  # for generator
            return -pred.mean()


@LossRegistry.register_loss()
class FeatureMatchingLoss(BaseLoss):
    """Feature Matching Loss Module for GANs.

    This module implements the feature matching loss for improved GAN training.
    """

    def __init__(self):
        """Initialize the FeatureMatchingLoss module."""
        super().__init__()

    def forward(self, real_features: list, fake_features: list) -> torch.Tensor:
        """Forward pass through the FeatureMatchingLoss module.

        Args:
            real_features (list): List of discriminator features for real data.
            fake_features (list): List of discriminator features for fake data.

        Returns:
            torch.Tensor: The feature matching loss.
        """
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat.mean(0), real_feat.detach().mean(0))

        return loss


@LossRegistry.register_loss()
class R1GradientPenalty(BaseLoss):
    """R1 Gradient Penalty Module for GANs.

    This module implements the R1 gradient penalty for GAN training.
    """

    def __init__(self, gamma=10.0):
        """Initialize the R1GradientPenalty module.

        Args:
            gamma (float): Weight for the gradient penalty. Default is 10.0.
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, real_data: torch.Tensor, real_outputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the R1GradientPenalty module.

        Args:
            real_data (torch.Tensor): Real input data.
            real_outputs (torch.Tensor): Discriminator outputs for real data.

        Returns:
            torch.Tensor: The R1 gradient penalty.
        """
        # Check if real_data requires gradients
        if not real_data.requires_grad:
            # If not, issue a warning and return zero penalty
            import warnings

            warnings.warn("The real_data tensor does not require gradients. The grad will be treated as zero.")
            return torch.tensor(0.0, device=real_data.device)

        # Create gradient graph
        grad_real = torch.autograd.grad(outputs=real_outputs.sum(), inputs=real_data, create_graph=True, retain_graph=True, allow_unused=True)[0]  # Allow unused gradients

        # If gradient is None, return zero penalty
        if grad_real is None:
            return torch.tensor(0.0, device=real_data.device)

        # Flatten the gradients
        grad_real = grad_real.view(grad_real.size(0), -1)

        # Calculate gradient penalty
        grad_penalty = (grad_real.norm(2, dim=1) ** 2).mean()

        return self.gamma * 0.5 * grad_penalty


__all__ = ["VanillaGANLoss", "LSGANLoss", "WassersteinGANLoss", "HingeLoss", "FeatureMatchingLoss", "R1GradientPenalty"]
