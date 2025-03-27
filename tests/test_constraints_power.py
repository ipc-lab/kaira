import pytest
import torch
from kaira.constraints import TotalPowerConstraint, AveragePowerConstraint, PAPRConstraint

def test_total_power_constraint():
    constraint = TotalPowerConstraint(total_power=1.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.isclose(torch.sum(torch.abs(y) ** 2), torch.tensor(1.0), rtol=1e-4, atol=1e-4)

def test_average_power_constraint():
    constraint = AveragePowerConstraint(average_power=0.1)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    assert torch.isclose(torch.mean(torch.abs(y) ** 2), torch.tensor(0.1), rtol=1e-4, atol=1e-4)

def test_papr_constraint():
    constraint = PAPRConstraint(max_papr=3.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = constraint(x)
    avg_power = torch.mean(torch.abs(y) ** 2)
    peak_power = torch.max(torch.abs(y) ** 2)
    papr = peak_power / avg_power
    assert papr <= 3.0

def test_total_power_constraint_complex():
    """Test TotalPowerConstraint with complex-valued input."""
    constraint = TotalPowerConstraint(total_power=1.0)
    # Create a complex tensor
    x = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    y = constraint(x)
    # Verify total power is 1.0
    assert torch.isclose(torch.sum(torch.abs(y) ** 2), torch.tensor(1.0), rtol=1e-4, atol=1e-4)

def test_average_power_constraint_complex():
    """Test AveragePowerConstraint with complex-valued input."""
    constraint = AveragePowerConstraint(average_power=0.5)
    # Create a complex tensor
    x = torch.complex(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]))
    y = constraint(x)
    # Verify average power is 0.5
    assert torch.isclose(torch.mean(torch.abs(y) ** 2), torch.tensor(0.5), rtol=1e-4, atol=1e-4)

def test_papr_constraint_no_scaling_needed():
    """Test PAPRConstraint when the input already meets the constraint."""
    # Create a signal with PAPR = 1 (constant amplitude)
    x = torch.ones(10) * 2.0
    constraint = PAPRConstraint(max_papr=3.0)
    y = constraint(x)
    # No scaling should be applied
    assert torch.allclose(x, y)
    # Verify PAPR
    avg_power = torch.mean(torch.abs(y) ** 2)
    peak_power = torch.max(torch.abs(y) ** 2)
    papr = peak_power / avg_power
    assert papr <= 3.0

def test_papr_constraint_scaling_needed():
    """Test PAPRConstraint when scaling is needed to meet the constraint."""
    # Create a signal with high PAPR
    x = torch.ones(10)
    x[0] = 10.0  # This creates a high peak
    
    constraint = PAPRConstraint(max_papr=2.0)
    y = constraint(x)
    
    # Signal should be modified
    assert not torch.allclose(x, y)
    
    # Verify PAPR is now within limits
    avg_power = torch.mean(torch.abs(y) ** 2)
    peak_power = torch.max(torch.abs(y) ** 2)
    papr = peak_power / avg_power
    assert papr <= 2.0 + 1e-5  # Add a small tolerance
