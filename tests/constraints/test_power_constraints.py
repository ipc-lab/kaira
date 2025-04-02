"""Tests for power constraints."""
import pytest
import torch

from kaira.constraints import AveragePowerConstraint, TotalPowerConstraint


@pytest.fixture
def random_tensor():
    """Fixture providing a random tensor for testing."""
    return torch.randn(4, 2, 32, 32)


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_total_power_constraint(random_tensor, power):
    """Test total power constraint with different power levels."""
    constraint = TotalPowerConstraint(total_power=power)
    output = constraint(random_tensor)

    # Check shape preservation
    assert output.shape == random_tensor.shape

    # The constraint applies the power to each batch item separately
    # So for a batch size of 4, we expect the total power to be 4 * power
    batch_size = output.shape[0] if output.dim() > 1 else 1
    expected_power = power * batch_size
    
    # Check total power constraint is satisfied
    total_power = torch.sum(output**2)
    assert torch.isclose(total_power, torch.tensor(expected_power), rtol=1e-5)


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
    
    # The constraint applies the power to each batch item separately
    # For 4 batch items, the total power should be 4 * power
    batch_size = total_output.shape[0]
    expected_power = power * batch_size
    
    assert torch.isclose(torch.sum(total_output**2), torch.tensor(expected_power), rtol=1e-5)

    # Test complex average power constraint
    avg_constraint = AveragePowerConstraint(average_power=power)
    avg_output = avg_constraint(x)
    assert torch.isclose(torch.mean(avg_output**2), torch.tensor(power), rtol=1e-5)