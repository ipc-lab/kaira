# tests/test_constraints_antenna.py
import pytest
import torch

from kaira.constraints import PerAntennaPowerConstraint


@pytest.fixture
def multi_antenna_signal():
    """Fixture providing a random multi-antenna signal for testing."""
    torch.manual_seed(42)
    # Create a tensor with shape [batch_size, num_antennas, time_steps]
    # representing a multi-antenna signal
    return torch.randn(8, 4, 32)


def test_per_antenna_power_constraint_uniform():
    """Test PerAntennaPowerConstraint with uniform power across antennas."""
    # Create constraint with uniform power
    uniform_power = 0.5
    constraint = PerAntennaPowerConstraint(uniform_power=uniform_power)

    # Create test signal
    batch_size = 8
    num_antennas = 4
    time_steps = 32
    signal = torch.randn(batch_size, num_antennas, time_steps)

    # Apply constraint
    constrained_signal = constraint(signal)

    # Check output shape matches input
    assert constrained_signal.shape == signal.shape

    # Calculate power per antenna in constrained signal
    spatial_dims = (2,)  # time dimension
    constrained_power = torch.mean(constrained_signal**2, dim=spatial_dims)

    # Check that each antenna's power matches the uniform power
    for batch_idx in range(batch_size):
        for antenna_idx in range(num_antennas):
            assert torch.isclose(constrained_power[batch_idx, antenna_idx], torch.tensor(uniform_power), rtol=1e-5)


def test_per_antenna_power_constraint_budget(multi_antenna_signal):
    """Test PerAntennaPowerConstraint with individual power budgets."""
    # Get signal dimensions
    batch_size, num_antennas, time_steps = multi_antenna_signal.shape

    # Create power budget tensor with different values for each antenna
    power_budget = torch.tensor([0.1, 0.2, 0.5, 1.0])
    constraint = PerAntennaPowerConstraint(power_budget=power_budget)

    # Apply constraint
    constrained_signal = constraint(multi_antenna_signal)

    # Calculate power per antenna in constrained signal
    spatial_dims = (2,)  # time dimension
    constrained_power = torch.mean(constrained_signal**2, dim=spatial_dims)

    # Check that each antenna's power matches its budget
    for batch_idx in range(batch_size):
        for antenna_idx in range(num_antennas):
            assert torch.isclose(constrained_power[batch_idx, antenna_idx], power_budget[antenna_idx], rtol=1e-5)


def test_per_antenna_power_constraint_multi_dimensional():
    """Test PerAntennaPowerConstraint with multi-dimensional signals."""
    # Create a more complex signal with shape [batch_size, num_antennas, height, width]
    # representing a spatial signal from multiple antennas
    batch_size = 4
    num_antennas = 3
    height = 16
    width = 16
    signal = torch.randn(batch_size, num_antennas, height, width)

    # Create constraint with uniform power
    uniform_power = 0.75
    constraint = PerAntennaPowerConstraint(uniform_power=uniform_power)

    # Apply constraint
    constrained_signal = constraint(signal)

    # Check output shape matches input
    assert constrained_signal.shape == signal.shape

    # Calculate power per antenna in constrained signal (averaged over spatial dimensions)
    spatial_dims = (2, 3)  # height and width dimensions
    constrained_power = torch.mean(constrained_signal**2, dim=spatial_dims)

    # Check that each antenna's power matches the uniform power
    for batch_idx in range(batch_size):
        for antenna_idx in range(num_antennas):
            assert torch.isclose(constrained_power[batch_idx, antenna_idx], torch.tensor(uniform_power), rtol=1e-5)


def test_per_antenna_power_constraint_complex():
    """Test PerAntennaPowerConstraint with complex-valued signals."""
    # Create a complex signal with shape [batch_size, num_antennas, time_steps]
    batch_size = 6
    num_antennas = 4
    time_steps = 24
    real_part = torch.randn(batch_size, num_antennas, time_steps)
    imag_part = torch.randn(batch_size, num_antennas, time_steps)
    complex_signal = torch.complex(real_part, imag_part)

    # Create power budget tensor with different values for each antenna
    power_budget = torch.tensor([0.3, 0.6, 0.9, 1.2])
    constraint = PerAntennaPowerConstraint(power_budget=power_budget)

    # Apply constraint
    constrained_signal = constraint(complex_signal)

    # Check output shape and type match input
    assert constrained_signal.shape == complex_signal.shape
    assert constrained_signal.dtype == complex_signal.dtype

    # Calculate power per antenna in constrained signal
    spatial_dims = (2,)  # time dimension
    # For complex signals, power is the squared magnitude: |z|² = real² + imag²
    constrained_power = torch.mean(torch.abs(constrained_signal) ** 2, dim=spatial_dims)

    # Check that each antenna's power matches its budget
    for batch_idx in range(batch_size):
        for antenna_idx in range(num_antennas):
            assert torch.isclose(constrained_power[batch_idx, antenna_idx], power_budget[antenna_idx], rtol=1e-5)


def test_per_antenna_power_constraint_initialization_error():
    """Test that PerAntennaPowerConstraint raises appropriate errors for invalid initialization."""
    # Test with neither power_budget nor uniform_power
    with pytest.raises(AssertionError):
        PerAntennaPowerConstraint()
