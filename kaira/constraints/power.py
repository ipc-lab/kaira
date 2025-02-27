"""Power constraints for transmitted signals.

This module contains constraint implementations that enforce power limitations on signals. Power
constraints are fundamental in communication systems to ensure compliance with regulatory limits,
prevent hardware damage, and optimize energy efficiency.
"""

import torch

from .base import BaseConstraint


class TotalPowerConstraint(BaseConstraint):
    """Normalizes signal to achieve exact total power regardless of input signal power.

    This module applies a constraint on the total power of the input tensor. It ensures that the
    total power does not exceed a specified limit by scaling the signal appropriately.

    The constraint normalizes the signal to exactly match the specified power level,
    regardless of the input signal's power.

    Attributes:
        total_power (float): The maximum allowed total power
        total_power_factor (torch.Tensor): Precomputed square root of total power for efficiency
    """

    def __init__(self, total_power: float) -> None:
        """Initialize the TotalPowerConstraint module.

        Args:
            total_power (float): The target total power for the signal in linear units
                (not dB). The constraint will scale the signal to achieve exactly this
                power level.
        """
        super().__init__()
        self.total_power = total_power
        self.total_power_factor = torch.sqrt(torch.tensor(self.total_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the total power constraint to the input tensor.

        Normalizes the input tensor to have exactly the specified total power.

        Args:
            x (torch.Tensor): The input tensor of any shape

        Returns:
            torch.Tensor: The scaled tensor with the same shape as input, adjusted to
                have exactly the target total power

        Note:
            The power is calculated across all dimensions except the batch dimension.
            A small epsilon (1e-8) is added to the denominator to prevent division by zero.
        """
        dims = self.get_dimensions(x)
        x_norm = torch.norm(x, dim=dims, keepdim=True)
        x = x * self.total_power_factor / (x_norm + 1e-8)
        return x


class AveragePowerConstraint(BaseConstraint):
    """Scales signal to achieve specified average power per sample.

    This module applies a constraint on the average power of the input tensor. It ensures that the
    average power (power per sample) does not exceed a specified limit.

    Unlike the TotalPowerConstraint which constrains the sum of power across all samples,
    this constraint focuses on the average power per sample.

    Attributes:
        average_power (float): The maximum allowed average power
        power_avg_factor (torch.Tensor): Precomputed square root of average power for efficiency
    """

    def __init__(self, average_power: float) -> None:
        """Initialize the AveragePowerConstraint module.

        Args:
            average_power (float): The target average power per sample in linear units
                (not dB). The constraint will scale the signal to achieve exactly this
                average power level.
        """
        super().__init__()
        self.average_power = average_power
        self.power_avg_factor = torch.sqrt(torch.tensor(self.average_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the average power constraint to the input tensor.

        Normalizes the input tensor to have exactly the specified average power.

        Args:
            x (torch.Tensor): The input tensor of any shape

        Returns:
            torch.Tensor: The scaled tensor with the same shape as input, adjusted to
                have exactly the target average power

        Note:
            The power is calculated across all dimensions except the batch dimension.
            A small epsilon (1e-8) is added to the denominator to prevent division by zero.
        """
        dims = self.get_dimensions(x)
        x_norm = torch.norm(x, dim=dims, keepdim=True)
        avg_power_sqrt = self.power_avg_factor * torch.sqrt(
            torch.prod(torch.tensor(x.shape[1:]), 0)
        )
        x = x * self.power_avg_factor * avg_power_sqrt / (x_norm + 1e-8)
        return x


class ComplexTotalPowerConstraint(TotalPowerConstraint):
    """Adjusts complex signal power accounting for real and imaginary component distribution.

    Specialization of TotalPowerConstraint for complex-valued signals. Since complex signals
    distribute power between real and imaginary components, this constraint adjusts the power
    calculation by a factor of sqrt(0.5) to account for this distribution.

    This constraint is particularly useful for baseband complex signals in communications.
    """

    def __init__(self, total_power: float) -> None:
        """Initialize the ComplexTotalPowerConstraint module.

        Args:
            total_power (float): The target total power for the complex signal in linear
                units (not dB). Internally, this is scaled by sqrt(0.5) to account for
                power distribution between real and imaginary components.
        """
        super().__init__(total_power * torch.sqrt(0.5))


class ComplexAveragePowerConstraint(AveragePowerConstraint):
    """Applies average power constraint adjusted for complex signal characteristics.

    Specialization of AveragePowerConstraint for complex-valued signals. Since complex signals
    distribute power between real and imaginary components, this constraint adjusts the power
    calculation by a factor of sqrt(0.5) to account for this distribution.
    """

    def __init__(self, average_power: float) -> None:
        """Initialize the ComplexAveragePowerConstraint module.

        Args:
            average_power (float): The target average power for the complex signal in linear
                units (not dB). Internally, this is scaled by sqrt(0.5) to account for
                power distribution between real and imaginary components.
        """
        super().__init__(average_power * torch.sqrt(0.5))


class PAPRConstraint(BaseConstraint):
    """Reduces peak-to-average power ratio using soft clipping to minimize signal distortion.

    Limits the peak-to-average power ratio of the signal, which is critical in OFDM and
    multicarrier systems to reduce nonlinear distortions and improve power amplifier efficiency.

    This constraint applies soft clipping to signal peaks that would cause the PAPR to
    exceed the specified threshold, while preserving the signal shape as much as possible.

    Attributes:
        max_papr (float): Maximum allowed peak-to-average power ratio in linear units (not dB)
    """

    def __init__(self, max_papr: float = 3.0) -> None:
        """Initialize the PAPR constraint.

        Args:
            max_papr (float, optional): Maximum allowed peak-to-average power ratio in
                linear units (not dB). For reference, a max_papr of 4.0 corresponds to
                approximately 6 dB. Defaults to 3.0.
        """
        super().__init__()
        self.max_papr = max_papr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PAPR constraint to the input tensor.

        Finds signal peaks that cause excessive PAPR and scales them down to meet
        the constraint while preserving the overall signal shape.

        Args:
            x (torch.Tensor): The input tensor of any shape

        Returns:
            torch.Tensor: Signal with constrained PAPR with the same shape as input

        Note:
            This implementation uses a soft clipping approach that only affects
            portions of the signal where the PAPR constraint would be violated,
            preserving other parts of the signal unchanged.
        """
        dims = self.get_dimensions(x)

        # Calculate average power
        avg_power = torch.mean(torch.abs(x) ** 2, dim=dims, keepdim=True)

        # Find peak power
        peak_power = torch.max(torch.abs(x) ** 2, dim=dims, keepdim=True)[0]

        # Calculate current PAPR
        current_papr = peak_power / (avg_power + 1e-8)

        # Apply soft clipping where PAPR exceeds threshold
        excess_indices = current_papr > self.max_papr

        if torch.any(excess_indices):
            correction = torch.sqrt(avg_power * self.max_papr / peak_power)
            scaling = torch.where(excess_indices, correction, torch.ones_like(correction))
            return x * scaling

        return x
