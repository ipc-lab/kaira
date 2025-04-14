"""Antenna-specific constraints for communication systems.

This module provides constraints that apply to multi-antenna systems such as MIMO, including per-
antenna power distribution and other antenna-specific limitations. These constraints are essential
for ensuring proper operation of multi-antenna transmitters and compliance with hardware
specifications :cite:`paulraj2003introduction` :cite:`spencer2004introduction`.
"""

from typing import Optional

import torch

from .base import BaseConstraint
from .registry import ConstraintRegistry


@ConstraintRegistry.register_constraint()
class PerAntennaPowerConstraint(BaseConstraint):
    """Distributes power budget across multiple antennas to ensure per-antenna power limits.

    Ensures each antenna in a multi-antenna system (such as MIMO) adheres to its specific
    power budget. This constraint is crucial for systems where each antenna has its own
    power amplifier with individual power limitations :cite:`yu2007transmitter` :cite:`wunder2013energy`.

    Per-antenna power constraints are often more practical than sum-power constraints in real
    MIMO systems, as discussed in :cite:`christopoulos2014weighted` and :cite:`yu2007transmitter`.

    The constraint can be configured either with individual power budgets for each antenna
    or with a uniform power value across all antennas.

    Attributes:
        power_budget (torch.Tensor, optional): Power budget tensor for each antenna
        uniform_power (float, optional): Uniform power level for all antennas
    """

    def __init__(self, power_budget: Optional[torch.Tensor] = None, uniform_power: Optional[float] = None, *args, **kwargs) -> None:
        """Initialize the per-antenna power constraint.

        Args:
            power_budget (torch.Tensor, optional): Power budget for each antenna. Shape should be
                [num_antennas]. Mutually exclusive with uniform_power.
            uniform_power (float, optional): Uniform power value to apply across all antennas.
                Mutually exclusive with power_budget.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            AssertionError: If neither power_budget nor uniform_power is provided

        Note:
            Either power_budget or uniform_power must be provided, but not both.
            If power_budget is provided, its length must match the number of antennas
            in the input signal.
        """
        super().__init__(*args, **kwargs)
        assert (power_budget is not None) or (uniform_power is not None), "Either power_budget or uniform_power must be provided"
        self.power_budget = power_budget
        self.uniform_power = uniform_power

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply per-antenna power constraint.

        Scales the signal from each antenna independently to meet its power budget.
        The second dimension of the input tensor is assumed to be the antenna dimension.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, num_antennas, ...].
                The second dimension must correspond to different antennas.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Power-constrained signal with the same shape as input, where
                each antenna's signal has been scaled to meet its power budget

        Note:
            Power is calculated by averaging the squared magnitude across all dimensions
            except batch and antenna dimensions.
        """
        # Calculate current power per antenna (all dimensions except batch and antenna)
        spatial_dims = tuple(range(2, len(x.shape)))
        antenna_power = torch.mean(torch.abs(x) ** 2, dim=spatial_dims, keepdim=True)

        # Determine target power
        if self.power_budget is not None:
            target_power = self.power_budget.view(1, -1, *([1] * (len(x.shape) - 2)))
        else:  # Use uniform power
            target_power = self.uniform_power * torch.ones_like(antenna_power)

        # Scale to meet power constraints
        scaling_factor = torch.sqrt(target_power / (antenna_power + 1e-8))

        return x * scaling_factor
