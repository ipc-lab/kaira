"""Constraints module for Kaira.

This module contains various constraints that can be applied to the transmitted signals. These
constraints help in shaping the transmitted signal to meet certain requirements such as power
limitations or spectral characteristics.
"""

import torch

from kaira.core import BaseConstraint

__all__ = [
    "TotalPowerConstraint",
    "AveragePowerConstraint",
    "ComplexTotalPowerConstraint",
    "ComplexAveragePowerConstraint",
]


class TotalPowerConstraint(BaseConstraint):
    """Total Power Constraint Module.

    This module applies a constraint on the total power of the input tensor. It ensures that the
    total power does not exceed a specified limit.
    """

    def __init__(self, total_power: float) -> None:
        """Initialize the TotalPowerConstraint module.

        Args:
            total_power (float): The maximum allowed total power.
        """
        super().__init__()
        self.total_power = total_power
        self.total_power_factor = torch.sqrt(torch.tensor(self.total_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the total power constraint to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The scaled tensor to meet the power constraint.
        """
        x_norm = torch.norm(x, dim=tuple(range(1, len(x.shape))), keepdim=True)
        x = x * self.total_power_factor / x_norm
        return x


class AveragePowerConstraint(BaseConstraint):
    """Average Power Constraint Module.

    This module applies a constraint on the average power of the input tensor. It ensures that the
    average power does not exceed a specified limit.
    """

    def __init__(self, average_power: float) -> None:
        """Initialize the AveragePowerConstraint module.

        Args:
            average_power (float): The maximum allowed average power.
        """
        super().__init__()
        self.average_power = average_power
        self.power_avg_factor = torch.sqrt(torch.tensor(self.average_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the average power constraint to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The scaled tensor to meet the power constraint.
        """
        x_norm = torch.norm(x, dim=tuple(range(1, len(x.shape))), keepdim=True)
        avg_power_sqrt = self.power_avg_factor * torch.sqrt(
            torch.prod(torch.tensor(x.shape[1:]), 0)
        )
        x = x * self.power_avg_factor * avg_power_sqrt / x_norm
        return x


class ComplexTotalPowerConstraint(TotalPowerConstraint):
    """Complex Total Power Constraint Module."""

    def __init__(self, total_power: float) -> None:
        """Initialize the ComplexTotalPowerConstraint module."""
        super().__init__(total_power * torch.sqrt(0.5))


class ComplexAveragePowerConstraint(AveragePowerConstraint):
    """Complex Average Power Constraint Module."""

    def __init__(self, average_power: float) -> None:
        """Initialize the ComplexAveragePowerConstraint module."""
        super().__init__(average_power * torch.sqrt(0.5))
