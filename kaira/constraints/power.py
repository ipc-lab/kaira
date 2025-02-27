"""Power constraints for transmitted signals.

This module contains constraint implementations that enforce power limitations on signals.
"""

import torch
from .base import BaseConstraint


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
        dims = self.get_dimensions(x)
        x_norm = torch.norm(x, dim=dims, keepdim=True)
        x = x * self.total_power_factor / (x_norm + 1e-8)
        return x
        
    def __repr__(self) -> str:
        """Return a string representation of the constraint."""
        return f"{self.__class__.__name__}(total_power={self.total_power})"


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
        dims = self.get_dimensions(x)
        x_norm = torch.norm(x, dim=dims, keepdim=True)
        avg_power_sqrt = self.power_avg_factor * torch.sqrt(
            torch.prod(torch.tensor(x.shape[1:]), 0)
        )
        x = x * self.power_avg_factor * avg_power_sqrt / (x_norm + 1e-8)
        return x
        
    def __repr__(self) -> str:
        """Return a string representation of the constraint."""
        return f"{self.__class__.__name__}(average_power={self.average_power})"


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


class PAPRConstraint(BaseConstraint):
    """Peak-to-Average Power Ratio (PAPR) Constraint.
    
    Limits the peak-to-average power ratio of the signal, which is critical
    in OFDM and multicarrier systems to reduce nonlinear distortions.
    """
    
    def __init__(self, max_papr: float = 3.0) -> None:
        """Initialize the PAPR constraint.
        
        Args:
            max_papr (float): Maximum allowed peak-to-average power ratio.
        """
        super().__init__()
        self.max_papr = max_papr
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PAPR constraint to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: Signal with constrained PAPR.
        """
        dims = self.get_dimensions(x)
        
        # Calculate average power
        avg_power = torch.mean(torch.abs(x)**2, dim=dims, keepdim=True)
        
        # Find peak power
        peak_power = torch.max(torch.abs(x)**2, dim=dims, keepdim=True)[0]
        
        # Calculate current PAPR
        current_papr = peak_power / (avg_power + 1e-8)
        
        # Apply soft clipping where PAPR exceeds threshold
        excess_indices = current_papr > self.max_papr
        
        if torch.any(excess_indices):
            correction = torch.sqrt(avg_power * self.max_papr / peak_power)
            scaling = torch.where(excess_indices, correction, torch.ones_like(correction))
            return x * scaling
            
        return x
        
    def __repr__(self) -> str:
        """Return a string representation of the constraint."""
        return f"{self.__class__.__name__}(max_papr={self.max_papr})"
