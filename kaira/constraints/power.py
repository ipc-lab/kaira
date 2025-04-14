"""Power constraints for transmitted signals.

This module contains constraint implementations that enforce power limitations on signals. Power
constraints are fundamental in communication systems to ensure compliance with regulatory limits,
prevent hardware damage, and optimize energy efficiency :cite:`goldsmith2005wireless` :cite:`love2003grassmannian`.
"""

import torch

from .base import BaseConstraint
from .registry import ConstraintRegistry


@ConstraintRegistry.register_constraint()
class TotalPowerConstraint(BaseConstraint):
    """Normalizes signal to achieve exact total power regardless of input signal power.

    This module applies a constraint on the total power of the input tensor. It ensures that the
    total power does not exceed a specified limit by scaling the signal appropriately
    :cite:`wunder2013energy`.

    The constraint normalizes the signal to exactly match the specified power level,
    regardless of the input signal's power. It automatically detects complex signals and
    applies the appropriate power scaling, distributing power between real and imaginary
    components as needed.

    Attributes:
        total_power (float): The maximum allowed total power
        total_power_factor (torch.Tensor): Precomputed square root of total power for efficiency
    """

    def __init__(self, total_power: float, *args, **kwargs) -> None:
        """Initialize the TotalPowerConstraint module.

        Args:
            total_power (float): The target total power for the signal in linear units
                (not dB). The constraint will scale the signal to achieve exactly this
                power level for both real and complex signals.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_power = total_power
        self.total_power_factor = torch.sqrt(torch.tensor(self.total_power))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the total power constraint to the input tensor.

        Normalizes the input tensor to have exactly the specified total power.
        Automatically handles both real and complex-valued inputs.

        Args:
            x (torch.Tensor): The input tensor of any shape (real or complex)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The scaled tensor with the same shape as input, adjusted to
                have exactly the target total power

        Note:
            The power is calculated across all dimensions except the batch dimension.
            For complex signals, power is distributed between real and imaginary components.
            A small epsilon (1e-8) is added to the denominator to prevent division by zero.
        """
        # Handle batched data by processing all batch items in parallel
        if x.dim() > 1 and x.shape[0] > 1:
            # For batched data, reshape to [batch_size, -1] to process each batch item independently but in parallel
            original_shape = x.shape
            batch_size = original_shape[0]

            # Reshape for parallel processing
            x_reshaped = x.reshape(batch_size, -1)

            # Process all batch items in parallel
            if torch.is_complex(x):
                current_power = torch.sum(torch.abs(x_reshaped) ** 2, dim=1, keepdim=True)
            else:
                current_power = torch.sum(x_reshaped**2, dim=1, keepdim=True)

            # Handle zero signals in a vectorized way
            zero_mask = current_power < 1e-10

            # Compute scaling factors for all batch items at once
            scale = torch.sqrt(self.total_power / (current_power + 1e-8))

            # Create the output tensor
            output = x_reshaped * scale

            # Handle zero signals
            if torch.any(zero_mask):
                uniform_value = self.total_power_factor / torch.sqrt(torch.tensor(x_reshaped.shape[1]))
                uniform_signal = torch.ones_like(x_reshaped) * uniform_value
                output = torch.where(zero_mask, uniform_signal, output)

            # Reshape back to original shape
            return output.reshape(original_shape)
        else:
            # For non-batched data or single batch item, apply constraint directly
            return self._apply_constraint_to_single_item(x, *args, **kwargs)

    def _apply_constraint_to_single_item(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply constraint to a single batch item or non-batched tensor.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The constrained tensor.
        """
        # Calculate the current total power of the input tensor
        if torch.is_complex(x):
            current_power = torch.sum(torch.abs(x) ** 2)
        else:
            current_power = torch.sum(x**2)

        # For zero signals, create a non-zero uniform signal with the desired power
        if current_power < 1e-10:
            # Create a uniform signal with the desired power
            uniform_signal = torch.ones_like(x) / torch.sqrt(torch.tensor(x.numel()))
            return uniform_signal * self.total_power_factor

        # Compute scaling factor to achieve target power
        scale = torch.sqrt(self.total_power / (current_power + 1e-8))

        # Scale the input to achieve desired total power
        return x * scale


@ConstraintRegistry.register_constraint()
class AveragePowerConstraint(BaseConstraint):
    """Scales signal to achieve specified average power per sample.

    This module applies a constraint on the average power of the input tensor. It ensures that the
    average power (power per sample) does not exceed a specified limit. Average power constraints
    are essential in communications systems for meeting regulatory requirements and optimizing
    signal-to-noise ratio :cite:`goldsmith2005wireless` :cite:`proakis2007digital`.

    Unlike the TotalPowerConstraint which constrains the sum of power across all samples,
    this constraint focuses on the average power per sample. It automatically handles
    both real and complex signals, applying appropriate power scaling for complex signals.

    Attributes:
        average_power (float): The maximum allowed average power
        power_avg_factor (torch.Tensor): Precomputed square root of average power for efficiency
    """

    def __init__(self, average_power: float, *args, **kwargs) -> None:
        """Initialize the AveragePowerConstraint module.

        Args:
            average_power (float): The target average power per sample in linear units
                (not dB). The constraint will scale the signal to achieve exactly this
                average power level for both real and complex signals.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.average_power = average_power
        self.power_avg_factor = torch.sqrt(torch.tensor(self.average_power))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the average power constraint to the input tensor.

        Normalizes the input tensor to have exactly the specified average power.
        Automatically handles both real and complex-valued inputs.

        Args:
            x (torch.Tensor): The input tensor of any shape (real or complex)
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The scaled tensor with the same shape as input, adjusted to
                have exactly the target average power

        Note:
            The power is calculated across all dimensions.
            For complex signals, power is distributed between real and imaginary components.
            A small epsilon (1e-8) is added to the denominator to prevent division by zero.
        """
        # Handle batched data by processing all batch items in parallel
        if x.dim() > 1 and x.shape[0] > 1:
            # For batched data, reshape to [batch_size, -1] to process each batch item independently but in parallel
            original_shape = x.shape
            batch_size = original_shape[0]

            # Reshape for parallel processing
            x_reshaped = x.reshape(batch_size, -1)
            num_elements = x_reshaped.shape[1]

            # Process all batch items in parallel
            if torch.is_complex(x):
                current_power = torch.sum(torch.abs(x_reshaped) ** 2, dim=1, keepdim=True) / num_elements
            else:
                current_power = torch.sum(x_reshaped**2, dim=1, keepdim=True) / num_elements

            # Handle zero signals in a vectorized way
            zero_mask = current_power < 1e-10

            # Compute scaling factors for all batch items at once
            scale = torch.sqrt(self.average_power / (current_power + 1e-8))

            # Create the output tensor
            output = x_reshaped * scale

            # Handle zero signals
            if torch.any(zero_mask):
                uniform_value = self.power_avg_factor * torch.sqrt(torch.tensor(num_elements)) / torch.sqrt(torch.tensor(num_elements))
                uniform_signal = torch.ones_like(x_reshaped) * uniform_value
                output = torch.where(zero_mask, uniform_signal, output)

            # Reshape back to original shape
            return output.reshape(original_shape)
        else:
            # For non-batched data or single batch item, apply constraint directly
            return self._apply_constraint_to_single_item(x, *args, **kwargs)

    def _apply_constraint_to_single_item(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply constraint to a single batch item or non-batched tensor.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The constrained tensor.
        """
        # Calculate the current average power of the input tensor
        num_elements = x.numel()

        if torch.is_complex(x):
            current_power = torch.sum(torch.abs(x) ** 2) / num_elements
        else:
            current_power = torch.sum(x**2) / num_elements

        # For zero or near-zero signals, create a non-zero uniform signal with the desired power
        if current_power < 1e-10:
            # Create a uniform signal with the desired average power
            uniform_signal = torch.ones_like(x) / torch.sqrt(torch.tensor(x.numel()))
            return uniform_signal * self.power_avg_factor * torch.sqrt(torch.tensor(num_elements))

        # Compute scaling factor to achieve target average power
        scale = torch.sqrt(self.average_power / (current_power + 1e-8))

        # Scale the input to achieve desired average power
        return x * scale


@ConstraintRegistry.register_constraint()
class PAPRConstraint(BaseConstraint):
    """Reduces peak-to-average power ratio using soft clipping to minimize signal distortion.

    Limits the peak-to-average power ratio of the signal, which is critical in OFDM and
    multicarrier systems to reduce nonlinear distortions and improve power amplifier efficiency
    :cite:`han2005overview` :cite:`jiang2008overview`.

    This constraint applies soft clipping to signal peaks that would cause the PAPR to
    exceed the specified threshold, while preserving the signal shape as much as possible.
    The PAPR reduction techniques are extensively studied in wireless communications
    :cite:`tellambura1997computation`.

    Attributes:
        max_papr (float): Maximum allowed peak-to-average power ratio in linear units (not dB)
    """

    def __init__(self, max_papr: float = 3.0, *args, **kwargs) -> None:
        """Initialize the PAPR constraint.

        Args:
            max_papr (float, optional): Maximum allowed peak-to-average power ratio in
                linear units (not dB). For reference, a max_papr of 4.0 corresponds to
                approximately 6 dB. Defaults to 3.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.max_papr = max_papr

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply PAPR constraint to the input tensor.

        Finds signal peaks that cause excessive PAPR and scales them down to meet
        the constraint while preserving the overall signal shape.

        Args:
            x (torch.Tensor): The input tensor of any shape
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Signal with constrained PAPR with the same shape as input

        Note:
            This implementation uses a multi-iteration approach to ensure the PAPR
            constraint is strictly enforced even for challenging signals.
        """
        # For PAPRConstraint, we still process batch items individually but use torch.vmap for parallelization
        if x.dim() > 1 and x.shape[0] > 1:
            # Use vmap to parallelize the constraint application across batch dimension
            # This requires PyTorch 1.9+ for torch.vmap
            try:
                # Define a wrapper function that takes a single tensor
                def apply_constraint(single_x):
                    return self._apply_constraint_to_single_item(single_x, *args, **kwargs)

                # Use vmap to vectorize the function across the first dimension (batch)
                vectorized_constraint = torch.vmap(apply_constraint)
                return vectorized_constraint(x)
            except (AttributeError, RuntimeError):
                # Fallback to original implementation if vmap is not available or fails
                batch_size = x.shape[0]
                output = torch.zeros_like(x)

                # Process in parallel using multiple workers if possible
                output = torch.stack([self._apply_constraint_to_single_item(x[i], *args, **kwargs) for i in range(batch_size)])
                return output
        else:
            # For non-batched data or single batch item, apply constraint directly
            return self._apply_constraint_to_single_item(x, *args, **kwargs)

    def _apply_constraint_to_single_item(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply strict PAPR constraint to a single tensor using multiple iterations.

        Args:
            x (torch.Tensor): The input tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The constrained tensor.
        """
        self.get_dimensions(x)
        result = x.clone()

        # Use multiple iterations of clipping to ensure PAPR constraint is met
        max_iterations = 15  # Increased from 10 for better convergence
        # Use a stricter safety margin to ensure we're comfortably under the limit
        target_papr = self.max_papr * 0.9  # Reduced from 0.95 for stricter enforcement

        for i in range(max_iterations):
            # Calculate average power
            avg_power = torch.mean(torch.abs(result) ** 2)

            # Calculate peak power
            peak_power = torch.max(torch.abs(result) ** 2)

            # Calculate current PAPR
            current_papr = peak_power / (avg_power + 1e-8)

            # Check if constraint is already satisfied with margin
            if current_papr <= self.max_papr * 0.98:  # Stricter check for termination
                break

            # Calculate maximum allowed amplitude based on target PAPR
            max_amplitude = torch.sqrt(avg_power * target_papr)

            # Apply hard clipping to peaks
            magnitudes = torch.abs(result)
            excess_mask = magnitudes > max_amplitude

            if torch.any(excess_mask):
                # Normalize excessive values by their magnitude to preserve phase (complex) or sign (real)
                normalized = result[excess_mask] / (magnitudes[excess_mask] + 1e-8)

                # Apply clipping while preserving signal phase/sign
                result[excess_mask] = normalized * max_amplitude

                # For later iterations, apply more aggressive clipping
                if i > max_iterations // 2:
                    factor = 0.95 - 0.05 * (i - max_iterations // 2)
                    stricter_max_amp = torch.sqrt(avg_power * target_papr) * factor
                    magnitudes = torch.abs(result)
                    stricter_mask = magnitudes > stricter_max_amp
                    if torch.any(stricter_mask):
                        # Division by magnitude preserves phase (complex) or sign (real)
                        normalized = result[stricter_mask] / (magnitudes[stricter_mask] + 1e-8)
                        result[stricter_mask] = normalized * stricter_max_amp

        # Final check and hard clipping as a safety measure
        avg_power = torch.mean(torch.abs(result) ** 2)
        final_max_amplitude = torch.sqrt(avg_power * self.max_papr * 0.98)
        magnitudes = torch.abs(result)
        final_excess_mask = magnitudes > final_max_amplitude

        if torch.any(final_excess_mask):
            # Final hard clipping to ensure we're under the limit
            # This preserves phase for complex signals and sign for real signals
            normalized = result[final_excess_mask] / (magnitudes[final_excess_mask] + 1e-8)
            result[final_excess_mask] = normalized * final_max_amplitude

        return result
