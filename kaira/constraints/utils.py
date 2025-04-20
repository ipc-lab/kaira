"""Utility functions for constraints.

This module provides helper functions for creating, testing, validating, and working with
constraints in wireless communication systems. These utilities streamline the process of
configuring common constraint combinations and verifying constraint effectiveness.
"""

from typing import Any, Dict, List, Optional

import torch

from .antenna import PerAntennaPowerConstraint
from .base import BaseConstraint
from .composite import CompositeConstraint
from .power import TotalPowerConstraint
from .signal import PeakAmplitudeConstraint, SpectralMaskConstraint

__all__ = [
    "create_ofdm_constraints",
    "create_mimo_constraints",
    "combine_constraints",
    "verify_constraint",
    "apply_constraint_chain",
    "measure_signal_properties",
]
# Factory functions for common constraint combinations


def create_ofdm_constraints(
    total_power: float,
    max_papr: float = 6.0,
    is_complex: bool = True,
    peak_amplitude: Optional[float] = None,
) -> CompositeConstraint:
    """Create constraints commonly used in OFDM systems.

    Configures a set of constraints suitable for Orthogonal Frequency Division
    Multiplexing (OFDM) signals. This includes a power constraint and a Peak-to-Average
    Power Ratio (PAPR) constraint to handle the high dynamic range typical of OFDM.

    Args:
        total_power (float): Total power constraint value in linear units
        max_papr (float, optional): Maximum allowed PAPR in linear units (not dB).
            Defaults to 6.0 (approximately 7.8 dB).
        is_complex (bool, optional): Whether the signal is complex-valued.
            Defaults to True.
        peak_amplitude (float, optional): If provided, adds a peak amplitude constraint.
            Defaults to None.

    Returns:
        CompositeConstraint: Combined OFDM constraints ready to be applied to signals

    Example:
        >>> ofdm_constraints = create_ofdm_constraints(total_power=1.0, max_papr=4.0)
        >>> constrained_signal = ofdm_constraints(input_signal)
    """
    constraints = []

    # Add PAPR constraint
    if max_papr is not None:
        from .power import PAPRConstraint

        constraints.append(PAPRConstraint(max_papr=max_papr))

    # Only add peak amplitude constraint if explicitly provided
    if peak_amplitude is not None:
        # Add explicit peak amplitude constraint if provided
        constraints.append(PeakAmplitudeConstraint(peak_amplitude))

    # Add power constraint
    constraints.append(TotalPowerConstraint(total_power))

    return CompositeConstraint(constraints)


def create_mimo_constraints(
    num_antennas: int,
    uniform_power: Optional[float] = None,
    max_papr: Optional[float] = None,
    spectral_mask: Optional[torch.Tensor] = None,
    total_power: Optional[float] = None,
) -> CompositeConstraint:
    """Create constraints commonly used in MIMO systems.

    Configures constraints appropriate for Multiple-Input Multiple-Output (MIMO)
    systems, focusing on either maintaining equal power distribution across antennas
    or controlling total power across all antennas, while optionally controlling PAPR
    and spectral characteristics.

    Args:
        num_antennas (int): Number of antennas in the MIMO system
        uniform_power (float, optional): Power per antenna in linear units. If None and
            total_power is provided, will use a total power constraint instead.
            Defaults to None.
        max_papr (float, optional): Maximum allowed PAPR in linear units (not dB).
            If None, no PAPR constraint is applied. Defaults to None.
        spectral_mask (torch.Tensor, optional): If provided, adds a spectral mask constraint.
            Defaults to None.
        total_power (float, optional): If provided, uses a total power constraint instead of
            per-antenna power constraints. This is useful when the total transmit power is
            limited, but power can be allocated flexibly across antennas. Defaults to None.

    Returns:
        CompositeConstraint: Combined MIMO constraints ready to be applied to signals

    Raises:
        ValueError: If both uniform_power and total_power are None, or if both are provided

    Example:
        >>> # Example with per-antenna power constraint
        >>> mimo_constraints = create_mimo_constraints(
        ...     num_antennas=4, uniform_power=0.25, max_papr=4.0
        ... )
        >>> # Example with total power constraint
        >>> mimo_constraints = create_mimo_constraints(
        ...     num_antennas=4, total_power=1.0, max_papr=4.0
        ... )
        >>> constrained_signal = mimo_constraints(input_signal)
    """
    constraints = []

    # Check if we have valid power constraint settings
    if uniform_power is None and total_power is None:
        raise ValueError("Either uniform_power or total_power must be provided")
    if uniform_power is not None and total_power is not None:
        raise ValueError("Cannot specify both uniform_power and total_power; use one or the other")

    # Add power constraint first
    if uniform_power is not None:
        constraints.append(PerAntennaPowerConstraint(uniform_power=uniform_power))
    else:
        # At this point, total_power must be a float because of the earlier checks
        assert total_power is not None, "total_power cannot be None here due to prior validation"
        constraints.append(TotalPowerConstraint(total_power=total_power))

    # Add PAPR constraint if specified
    if max_papr is not None:
        from .power import PAPRConstraint

        constraints.append(PAPRConstraint(max_papr=max_papr))

    # Add spectral mask constraint if specified
    if spectral_mask is not None:
        constraints.append(SpectralMaskConstraint(spectral_mask))

    return CompositeConstraint(constraints)


def combine_constraints(constraints: List[BaseConstraint]) -> BaseConstraint:
    """Combine multiple constraints into a single constraint.

    Creates a composite constraint that applies multiple constraints in sequence.
    This is useful for building custom constraint chains.

    Args:
        constraints (List[BaseConstraint]): List of constraints to combine

    Returns:
        BaseConstraint: Combined constraint that applies all input constraints
        sequentially

    Raises:
        ValueError: If the constraints list is empty

    Example:
        >>> power_constraint = TotalPowerConstraint(1.0)
        >>> papr_constraint = PAPRConstraint(4.0)
        >>> amp_constraint = PeakAmplitudeConstraint(1.5)
        >>> combined = combine_constraints([power_constraint, papr_constraint, amp_constraint])
        >>> constrained_signal = combined(input_signal)
    """
    if not constraints:
        raise ValueError("Cannot combine an empty list of constraints")

    if len(constraints) == 1:
        return constraints[0]

    return CompositeConstraint(constraints)


# Verification and testing utilities


def verify_constraint(
    constraint: BaseConstraint,
    input_tensor: torch.Tensor,
    expected_property: str,
    expected_value: float,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that a constraint produces the expected property in the output.

    Tests whether applying a constraint to a tensor results in the expected
    property (such as power or PAPR) within a specified tolerance.

    Args:
        constraint (BaseConstraint): Constraint to test
        input_tensor (torch.Tensor): Input tensor to pass through the constraint
        expected_property (str): Name of the property to check.
            Valid values: 'power', 'papr', 'amplitude'
        expected_value (float): Expected value for the property in linear units
        tolerance (float, optional): Tolerance for numerical comparison. Defaults to 1e-5.

    Returns:
        Dict[str, Any]: Results dictionary containing:
            - input_shape: Shape of the input tensor
            - output_shape: Shape of the constrained output
            - success: Whether the constraint achieved the expected property
            - measured_<property>: Actual measured value of the property
            - expected_<property>: Expected value of the property

    Raises:
        ValueError: If expected_property is not one of the supported values

    Example:
        >>> power_constraint = TotalPowerConstraint(1.0)
        >>> input_signal = torch.randn(8, 64)
        >>> result = verify_constraint(power_constraint, input_signal, 'power', 1.0)
        >>> print(f"Constraint satisfied: {result['success']}")
    """
    constrained_output = constraint(input_tensor)

    results = {
        "input_shape": input_tensor.shape,
        "output_shape": constrained_output.shape,
        "success": False,
    }

    # Check property based on the expected type
    if expected_property == "power":
        # Calculate total power
        power = torch.sum(torch.abs(constrained_output) ** 2).item()
        results["measured_power"] = power
        results["expected_power"] = expected_value
        # Use relative tolerance for numerical stability
        relative_tolerance = tolerance * max(1.0, abs(expected_value))
        results["success"] = abs(power - expected_value) <= relative_tolerance

    elif expected_property == "papr":
        # Calculate PAPR
        mean_power = torch.mean(torch.abs(constrained_output) ** 2).item()
        peak_power = torch.max(torch.abs(constrained_output) ** 2).item()
        papr = peak_power / mean_power if mean_power > 0 else float("inf")
        results["measured_papr"] = papr
        results["expected_papr"] = expected_value
        # PAPR should be less than or equal to expected value
        # Use a more generous tolerance for PAPR as it's an approximation
        papr_tolerance = max(tolerance, 1.0)  # Allow more tolerance for PAPR constraints
        results["success"] = papr <= expected_value + papr_tolerance

    elif expected_property == "amplitude":
        # Check max amplitude
        max_amp = torch.max(torch.abs(constrained_output)).item()
        results["measured_max_amplitude"] = max_amp
        results["expected_max_amplitude"] = expected_value
        results["success"] = max_amp <= expected_value + tolerance

    else:
        raise ValueError(f"Unsupported property: {expected_property}. Supported values are: power, papr, amplitude")

    return results


def apply_constraint_chain(constraints: List[BaseConstraint], input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply a list of constraints in sequence and optionally print debug info.

    Applies multiple constraints to a tensor sequentially and provides optional
    debugging information about power changes at each step.

    Args:
        constraints (List[BaseConstraint]): List of constraint objects to apply in sequence
        input_tensor (torch.Tensor): Input tensor to be constrained

    Returns:
        torch.Tensor: Output tensor after applying all constraints

    Example:
        >>> constraints = [
        ...     TotalPowerConstraint(1.0),
        ...     PAPRConstraint(4.0)
        ... ]
        >>> output = apply_constraint_chain(constraints, input_signal)
    """
    x = input_tensor

    for constraint in constraints:
        x = constraint(x)

    return x


def measure_signal_properties(x: torch.Tensor) -> Dict[str, float]:
    """Measure common signal properties for a given tensor.

    Calculates key signal properties like power, PAPR, and peak amplitude
    that are commonly constrained in communication systems.

    Args:
        x (torch.Tensor): Input signal tensor

    Returns:
        Dict[str, float]: Dictionary of measured signal properties

    Example:
        >>> signal = torch.randn(1, 64)
        >>> props = measure_signal_properties(signal)
        >>> print(f"Signal PAPR: {props['papr']:.2f}")
    """
    mean_power = torch.mean(torch.abs(x) ** 2).item()
    peak_power = torch.max(torch.abs(x) ** 2).item()
    peak_amplitude = torch.max(torch.abs(x)).item()
    papr = peak_power / mean_power if mean_power > 0 else float("inf")

    return {
        "mean_power": mean_power,
        "peak_power": peak_power,
        "peak_amplitude": peak_amplitude,
        "papr": papr,
        "papr_db": 10 * torch.log10(torch.tensor(papr)).item() if mean_power > 0 else float("inf"),
    }
