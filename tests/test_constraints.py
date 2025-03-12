# tests/test_constraints.py
import pytest
import torch

from kaira.constraints import AveragePowerConstraint, TotalPowerConstraint


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_total_power_constraint(random_tensor, power):
    """Test total power constraint with different power levels."""
    constraint = TotalPowerConstraint(total_power=power)
    output = constraint(random_tensor)

    # Check shape preservation
    assert output.shape == random_tensor.shape

    # Check total power constraint is satisfied
    total_power = torch.sum(output**2)
    assert torch.isclose(total_power, torch.tensor(power), rtol=1e-5)


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_average_power_constraint(random_tensor, power):
    """Test average power constraint with different power levels."""
    constraint = AveragePowerConstraint(average_power=power)
    output = constraint(random_tensor)

    # Check average power constraint is satisfied
    avg_power = torch.mean(output**2)
    assert torch.isclose(avg_power, torch.tensor(power), rtol=1e-5)


def test_complex_constraints():
    """Test complex-valued power constraints."""
    x = torch.randn(4, 2, 32, 32)  # Complex-valued input
    power = 1.0

    # Test complex total power constraint
    total_constraint = TotalPowerConstraint(total_power=power)
    total_output = total_constraint(x)
    assert torch.isclose(torch.sum(total_output**2), torch.tensor(power), rtol=1e-5)

    # Test complex average power constraint
    avg_constraint = AveragePowerConstraint(average_power=power)
    avg_output = avg_constraint(x)
    assert torch.isclose(torch.mean(avg_output**2), torch.tensor(power), rtol=1e-5)
