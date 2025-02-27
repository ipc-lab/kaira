"""Utility functions for constraints.

This module provides helper functions for creating, testing, validating, and working with constraints.
"""

import torch
from typing import Dict, Any, Optional, List, Union

from .base import BaseConstraint, CompositeConstraint
from .power import (
    TotalPowerConstraint,
    AveragePowerConstraint, 
    ComplexTotalPowerConstraint,
    PAPRConstraint,
)
from .antenna import PerAntennaPowerConstraint
from .signal import PeakAmplitudeConstraint, SpectralMaskConstraint


# Factory functions

def create_ofdm_constraints(
    total_power: float, 
    max_papr: float = 6.0,
    is_complex: bool = True
) -> CompositeConstraint:
    """Create constraints commonly used in OFDM systems.
    
    Args:
        total_power (float): Total power constraint value
        max_papr (float): Maximum allowed PAPR
        is_complex (bool): Whether the signal is complex-valued
        
    Returns:
        CompositeConstraint: Combined OFDM constraints
    """
    constraints = []
    
    # Add power constraint based on signal type
    if is_complex:
        constraints.append(ComplexTotalPowerConstraint(total_power))
    else:
        constraints.append(TotalPowerConstraint(total_power))
    
    # Add PAPR constraint
    constraints.append(PAPRConstraint(max_papr))
    
    return CompositeConstraint(constraints)


def create_mimo_constraints(
    num_antennas: int,
    uniform_power: float,
    max_papr: Optional[float] = None
) -> CompositeConstraint:
    """Create constraints commonly used in MIMO systems.
    
    Args:
        num_antennas (int): Number of antennas
        uniform_power (float): Power per antenna
        max_papr (float, optional): Maximum allowed PAPR
        
    Returns:
        CompositeConstraint: Combined MIMO constraints
    """
    constraints = [PerAntennaPowerConstraint(uniform_power=uniform_power)]
    
    if max_papr is not None:
        constraints.append(PAPRConstraint(max_papr))
        
    return CompositeConstraint(constraints)


def combine_constraints(constraints: List[BaseConstraint]) -> BaseConstraint:
    """Combine multiple constraints into a single constraint.
    
    Args:
        constraints (List[BaseConstraint]): List of constraints to combine
        
    Returns:
        BaseConstraint: Combined constraint
    """
    if len(constraints) == 1:
        return constraints[0]
    return CompositeConstraint(constraints)


# Utility functions (from original utils.py)

def verify_constraint(
    constraint: BaseConstraint,
    input_tensor: torch.Tensor,
    expected_property: str,
    expected_value: float,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that a constraint produces the expected property in the output.
    
    Args:
        constraint (BaseConstraint): Constraint to test
        input_tensor (torch.Tensor): Input tensor to pass through the constraint
        expected_property (str): Name of the property to check ('power', 'papr', etc.)
        expected_value (float): Expected value for the property
        tolerance (float): Tolerance for numerical comparison
        
    Returns:
        Dict[str, Any]: Results dictionary with success flag and measurements
    """
    constrained_output = constraint(input_tensor)
    
    results = {
        'input_shape': input_tensor.shape,
        'output_shape': constrained_output.shape,
        'success': False,
    }
    
    # Check property based on the expected type
    if expected_property == 'power':
        # Calculate total power
        power = torch.mean(torch.abs(constrained_output)**2).item()
        results['measured_power'] = power
        results['expected_power'] = expected_value
        results['success'] = abs(power - expected_value) <= tolerance
        
    elif expected_property == 'papr':
        # Calculate PAPR
        mean_power = torch.mean(torch.abs(constrained_output)**2).item()
        peak_power = torch.max(torch.abs(constrained_output)**2).item()
        papr = peak_power / mean_power if mean_power > 0 else float('inf')
        results['measured_papr'] = papr
        results['expected_papr'] = expected_value
        # PAPR should be less than or equal to expected value
        results['success'] = papr <= expected_value + tolerance
        
    elif expected_property == 'amplitude':
        # Check max amplitude
        max_amp = torch.max(torch.abs(constrained_output)).item()
        results['measured_max_amplitude'] = max_amp
        results['expected_max_amplitude'] = expected_value
        results['success'] = max_amp <= expected_value + tolerance
    
    return results


def apply_constraint_chain(
    constraints: list,
    input_tensor: torch.Tensor,
    verbose: bool = False
) -> torch.Tensor:
    """Apply a list of constraints in sequence and optionally print debug info.
    
    Args:
        constraints (list): List of constraint objects
        input_tensor (torch.Tensor): Input tensor
        verbose (bool): Whether to print debug information
        
    Returns:
        torch.Tensor: Output after applying all constraints
    """
    x = input_tensor
    
    if verbose:
        print(f"Input shape: {x.shape}, power: {torch.mean(torch.abs(x)**2).item():.6f}")
    
    for i, constraint in enumerate(constraints):
        x_prev = x
        x = constraint(x)
        
        if verbose:
            power_before = torch.mean(torch.abs(x_prev)**2).item()
            power_after = torch.mean(torch.abs(x)**2).item()
            print(f"[{i}] {constraint.__class__.__name__}: "
                  f"power before={power_before:.6f}, "
                  f"after={power_after:.6f}, "
                  f"change={power_after/power_before:.6f}x")
    
    return x
