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
from .power import ComplexTotalPowerConstraint, PAPRConstraint, TotalPowerConstraint
from .signal import PeakAmplitudeConstraint, SpectralMaskConstraint

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

    # Add power constraint based on signal type
    if is_complex:
        constraints.append(ComplexTotalPowerConstraint(total_power))
    else:
        constraints.append(TotalPowerConstraint(total_power))

    # Add PAPR constraint
    constraints.append(PAPRConstraint(max_papr))

    # Add peak amplitude constraint if specified
    if peak_amplitude is not None:
        constraints.append(PeakAmplitudeConstraint(peak_amplitude))

    return CompositeConstraint(constraints)


def create_mimo_constraints(
    num_antennas: int,
    uniform_power: float,
    max_papr: Optional[float] = None,
    spectral_mask: Optional[torch.Tensor] = None,
) -> CompositeConstraint:
    """Create constraints commonly used in MIMO systems.

    Configures constraints appropriate for Multiple-Input Multiple-Output (MIMO)
    systems, focusing on maintaining equal power distribution across antennas
    while optionally controlling PAPR.

    Args:
        num_antennas (int): Number of antennas in the MIMO system
        uniform_power (float): Power per antenna in linear units
        max_papr (float, optional): Maximum allowed PAPR in linear units (not dB).
            If None, no PAPR constraint is applied. Defaults to None.
        spectral_mask (torch.Tensor, optional): If provided, adds a spectral mask constraint.
            Defaults to None.

    Returns:
        CompositeConstraint: Combined MIMO constraints ready to be applied to signals

    Example:
        >>> mimo_constraints = create_mimo_constraints(
        ...     num_antennas=4, uniform_power=0.25, max_papr=4.0
        ... )
        >>> constrained_signal = mimo_constraints(input_signal)
    """
    constraints = [PerAntennaPowerConstraint(uniform_power=uniform_power)]

    if max_papr is not None:
        constraints.append(PAPRConstraint(max_papr))

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
        power = torch.mean(torch.abs(constrained_output) ** 2).item()
        results["measured_power"] = power
        results["expected_power"] = expected_value
        results["success"] = abs(power - expected_value) <= tolerance

    elif expected_property == "papr":
        # Calculate PAPR
        mean_power = torch.mean(torch.abs(constrained_output) ** 2).item()
        peak_power = torch.max(torch.abs(constrained_output) ** 2).item()
        papr = peak_power / mean_power if mean_power > 0 else float("inf")
        results["measured_papr"] = papr
        results["expected_papr"] = expected_value
        # PAPR should be less than or equal to expected value
        results["success"] = papr <= expected_value + tolerance

    elif expected_property == "amplitude":
        # Check max amplitude
        max_amp = torch.max(torch.abs(constrained_output)).item()
        results["measured_max_amplitude"] = max_amp
        results["expected_max_amplitude"] = expected_value
        results["success"] = max_amp <= expected_value + tolerance

    else:
        raise ValueError(
            f"Unsupported property: {expected_property}. Supported values are: power, papr, amplitude"
        )

    return results


def apply_constraint_chain(
    constraints: List[BaseConstraint], input_tensor: torch.Tensor, verbose: bool = False
) -> torch.Tensor:
    """Apply a list of constraints in sequence and optionally print debug info.

    Applies multiple constraints to a tensor sequentially and provides optional
    debugging information about power changes at each step.

    Args:
        constraints (List[BaseConstraint]): List of constraint objects to apply in sequence
        input_tensor (torch.Tensor): Input tensor to be constrained
        verbose (bool, optional): Whether to print debug information about
            power changes. Defaults to False.

    Returns:
        torch.Tensor: Output tensor after applying all constraints

    Example:
        >>> constraints = [
        ...     TotalPowerConstraint(1.0),
        ...     PAPRConstraint(4.0)
        ... ]
        >>> output = apply_constraint_chain(constraints, input_signal, verbose=True)
    """
    x = input_tensor

    if verbose:
        print(f"Input shape: {x.shape}, power: {torch.mean(torch.abs(x)**2).item():.6f}")

    for i, constraint in enumerate(constraints):
        x_prev = x
        x = constraint(x)

        if verbose:
            power_before = torch.mean(torch.abs(x_prev) ** 2).item()
            power_after = torch.mean(torch.abs(x) ** 2).item()
            print(
                f"[{i}] {constraint.__class__.__name__}: "
                f"power before={power_before:.6f}, "
                f"after={power_after:.6f}, "
                f"change={power_after/power_before:.6f}x"
            )

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
