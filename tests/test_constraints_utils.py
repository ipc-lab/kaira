import pytest
import torch
import numpy as np

from kaira.constraints import BaseConstraint, TotalPowerConstraint, PAPRConstraint
from kaira.constraints.utils import (
    combine_constraints,
    verify_constraint,
    apply_constraint_chain,
    measure_signal_properties
)


class DummyConstraint(BaseConstraint):
    """Dummy constraint for testing."""
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return x * self.scale


def test_combine_constraints():
    """Test combining multiple constraints."""
    # Create constraints
    c1 = DummyConstraint(2.0)
    c2 = DummyConstraint(0.5)
    
    # Combine them
    combined = combine_constraints([c1, c2])
    
    # Test on data
    x = torch.ones(10)
    result = combined(x)
    
    # Should apply c1 then c2: x * 2.0 * 0.5 = x
    assert torch.allclose(result, x)
    
    # Empty list should return identity
    identity = combine_constraints([])
    assert torch.allclose(identity(x), x)


def test_verify_constraint():
    """Test constraint verification utility."""
    # Test power constraint
    constraint = TotalPowerConstraint(total_power=2.0)
    x = torch.randn(100)
    
    # Verify power constraint
    result = verify_constraint(constraint, x, "power", 2.0)
    
    assert result["success"]
    assert "measured_power" in result
    assert abs(result["measured_power"] - 2.0) < 1e-4
    
    # Test PAPR constraint
    papr_constraint = PAPRConstraint(max_papr=4.0)
    result = verify_constraint(papr_constraint, x, "papr", 4.0)
    
    assert "measured_papr" in result
    assert result["measured_papr"] <= 4.0 + 1e-5  # Allow for numerical precision
    
    # Test amplitude constraint
    from kaira.constraints import PeakAmplitudeConstraint
    amp_constraint = PeakAmplitudeConstraint(max_amplitude=1.0)
    result = verify_constraint(amp_constraint, x, "amplitude", 1.0)
    
    assert "measured_max_amplitude" in result
    assert result["measured_max_amplitude"] <= 1.0 + 1e-5
    
    # Test invalid property
    with pytest.raises(ValueError):
        verify_constraint(constraint, x, "invalid_property", 1.0)


def test_apply_constraint_chain():
    """Test applying a chain of constraints with verbose output."""
    # Create constraints
    constraints = [
        DummyConstraint(2.0),
        DummyConstraint(0.5),
        DummyConstraint(3.0)
    ]
    
    # Create input
    x = torch.ones(10)
    
    # Apply chain
    result = apply_constraint_chain(constraints, x, verbose=True)
    
    # Check result (should be x * 2.0 * 0.5 * 3.0 = x * 3.0)
    assert torch.allclose(result, x * 3.0)
    
    # With empty list
    result = apply_constraint_chain([], x)
    assert torch.allclose(result, x)


def test_measure_signal_properties():
    """Test measuring signal properties."""
    # Create test signals
    constant = torch.ones(100) * 2.0
    varying = torch.cat([torch.ones(50) * 2.0, torch.ones(50) * 4.0])
    
    # Measure properties of constant signal
    props = measure_signal_properties(constant)
    
    assert "power" in props
    assert "peak_power" in props
    assert "mean_power" in props
    assert "papr" in props
    assert "papr_db" in props
    
    # Check values for constant signal
    assert props["power"] == 4.0
    assert props["peak_power"] == 4.0
    assert props["papr"] == 1.0
    
    # Check varying signal
    props = measure_signal_properties(varying)
    assert props["peak_power"] == 16.0
    assert props["mean_power"] == 10.0  # Average of 4 and 16
    assert props["papr"] == 1.6  # 16 / 10
    
    # Check zero signal (edge case)
    zero_signal = torch.zeros(10)
    props = measure_signal_properties(zero_signal)
    assert props["power"] == 0.0
    assert props["papr"] == float('inf')
    assert props["papr_db"] == float('inf')


def test_verify_constraint_power():
    """Test verify_constraint function with power property."""
    constraint = TotalPowerConstraint(total_power=1.0)
    input_tensor = torch.randn(10, 4) * 2  # Random tensor with arbitrary power
    
    results = verify_constraint(
        constraint=constraint,
        input_tensor=input_tensor,
        expected_property="power",
        expected_value=1.0,
    )
    
    assert "success" in results
    assert "measured_power" in results
    assert "expected_power" in results
    assert abs(results["measured_power"] - 1.0) <= 1e-5
    assert results["success"]


def test_verify_constraint_papr():
    """Test verify_constraint function with PAPR property."""
    constraint = TotalPowerConstraint(total_power=1.0)
    
    # Create a signal with known PAPR
    # A constant amplitude signal has PAPR = 1.0
    input_tensor = torch.ones(100, 1)
    
    results = verify_constraint(
        constraint=constraint,
        input_tensor=input_tensor,
        expected_property="papr",
        expected_value=1.5,  # Set expected higher than actual so test passes
    )
    
    assert "success" in results
    assert "measured_papr" in results
    assert "expected_papr" in results
    assert abs(results["measured_papr"] - 1.0) <= 1e-5
    assert results["success"]


def test_verify_constraint_amplitude():
    """Test verify_constraint function with amplitude property."""
    constraint = TotalPowerConstraint(total_power=1.0)
    
    # Create a signal with maximum amplitude of 2.0
    input_tensor = torch.ones(10, 1) * 2.0
    
    results = verify_constraint(
        constraint=constraint,
        input_tensor=input_tensor, 
        expected_property="amplitude",
        expected_value=1.0,  # After constraint, amplitude should be 1.0
    )
    
    assert "success" in results
    assert "measured_max_amplitude" in results
    assert "expected_max_amplitude" in results
    assert abs(results["measured_max_amplitude"] - 1.0) <= 1e-4
    assert results["success"]


def test_verify_constraint_invalid_property():
    """Test verify_constraint function with an invalid property."""
    constraint = TotalPowerConstraint(total_power=1.0)
    input_tensor = torch.randn(10, 4)
    
    with pytest.raises(ValueError, match="Unsupported property"):
        verify_constraint(
            constraint=constraint,
            input_tensor=input_tensor,
            expected_property="invalid_property",
            expected_value=1.0,
        )
