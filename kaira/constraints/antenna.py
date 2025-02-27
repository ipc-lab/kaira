"""Antenna-specific constraints for communication systems.

This module provides constraints that apply to multi-antenna systems such as MIMO,
including per-antenna power distribution and other antenna-specific limitations.
"""

import torch
from .base import BaseConstraint


class PerAntennaPowerConstraint(BaseConstraint):
    """Per-Antenna Power Constraint.
    
    Ensures each antenna in a MIMO system adheres to its power budget.
    """
    
    def __init__(self, power_budget: torch.Tensor = None, uniform_power: float = None) -> None:
        """Initialize the per-antenna power constraint.
        
        Args:
            power_budget (torch.Tensor, optional): Power budget for each antenna.
            uniform_power (float, optional): Uniform power across all antennas.
        """
        super().__init__()
        assert (power_budget is not None) or (uniform_power is not None), "Either power_budget or uniform_power must be provided"
        self.power_budget = power_budget
        self.uniform_power = uniform_power
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-antenna power constraint.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_antennas, ...].
            
        Returns:
            torch.Tensor: Power-constrained signal.
        """
        # Calculate current power per antenna (all dimensions except batch and antenna)
        spatial_dims = tuple(range(2, len(x.shape)))
        antenna_power = torch.mean(torch.abs(x)**2, dim=spatial_dims, keepdim=True)
        
        # Determine target power
        if self.power_budget is not None:
            target_power = self.power_budget.view(1, -1, *([1] * (len(x.shape) - 2)))
        else:  # Use uniform power
            target_power = self.uniform_power * torch.ones_like(antenna_power)
        
        # Scale to meet power constraints
        scaling_factor = torch.sqrt(target_power / (antenna_power + 1e-8))
        
        return x * scaling_factor
    