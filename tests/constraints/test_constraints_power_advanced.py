"""Tests for advanced power constraint scenarios."""
import pytest
import torch
import numpy as np

from kaira.constraints import AveragePowerConstraint, TotalPowerConstraint, PAPRConstraint
from kaira.constraints.composite import CompositeConstraint


@pytest.fixture
def complex_signal():
    """Fixture providing a complex signal for testing."""
    torch.manual_seed(42)
    n_samples = 1000
    real = torch.randn(n_samples)
    imag = torch.randn(n_samples)
    return torch.complex(real, imag)


@pytest.fixture
def multi_dimensional_signal():
    """Fixture providing a multi-dimensional real-valued signal."""
    torch.manual_seed(42)
    # Emulate an image batch: [batch_size, channels, height, width]
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def zero_signal():
    """Fixture providing a zero-valued signal for testing edge cases."""
    return torch.zeros(100)


def test_power_constraints_zero_signal(zero_signal):
    """Test power constraints with zero-valued signal (edge case)."""
    # Total power constraint
    power = 1.0
    total_constraint = TotalPowerConstraint(total_power=power)
    result = total_constraint(zero_signal)
    
    # The result should no longer be zero (since scaling to achieve power = 1.0)
    assert not torch.allclose(result, zero_signal)
    # The result should have total power close to the target
    assert torch.isclose(torch.sum(result**2), torch.tensor(power), rtol=1e-5)
    
    # Average power constraint
    avg_constraint = AveragePowerConstraint(average_power=power/len(zero_signal))
    result = avg_constraint(zero_signal)
    
    # The result should no longer be zero
    assert not torch.allclose(result, zero_signal)
    # The result should have average power close to the target
    assert torch.isclose(torch.mean(result**2), torch.tensor(power/len(zero_signal)), rtol=1e-5)


def test_complex_power_constraints_multi_channel(complex_signal):
    """Test power constraints with multi-channel complex signals."""
    # Create a multi-channel complex signal
    batch_size = 2
    channels = 3
    # Reshape the complex signal to [batch_size, channels, samples]
    multi_channel = complex_signal[:batch_size * channels * 100].reshape(batch_size, channels, 100)
    
    # Apply total power constraint
    power = 2.0
    constraint = TotalPowerConstraint(total_power=power)
    result = constraint(multi_channel)
    
    # Check shape preservation
    assert result.shape == multi_channel.shape
    
    # Check power constraint is satisfied
    measured_power = torch.sum(torch.abs(result)**2).item()
    assert abs(measured_power - power) < 1e-4


def test_composite_power_constraints(multi_dimensional_signal):
    """Test combining multiple power constraints and verify their application order."""
    # Set up constraints with clearly different values
    total_power = 5.0
    avg_power = 0.001
    papr_max = 3.0
    
    # Create individual constraints
    total_constraint = TotalPowerConstraint(total_power=total_power)
    avg_constraint = AveragePowerConstraint(average_power=avg_power)
    papr_constraint = PAPRConstraint(max_papr=papr_max)
    
    # Apply constraints in different orders
    # Order 1: Total -> Average -> PAPR
    composite1 = CompositeConstraint([total_constraint, avg_constraint, papr_constraint])
    result1 = composite1(multi_dimensional_signal)
    
    # Order 2: Average -> PAPR -> Total
    composite2 = CompositeConstraint([avg_constraint, papr_constraint, total_constraint])
    result2 = composite2(multi_dimensional_signal)
    
    # Verify constraints based on the order (last constraint should dominate)
    # For composite1, PAPR should be the key constraint
    measured_avg_power1 = torch.mean(torch.abs(result1)**2).item()
    peak_power1 = torch.max(torch.abs(result1)**2).item()
    measured_papr1 = peak_power1 / (measured_avg_power1 + 1e-8)
    assert measured_papr1 <= papr_max + 1e-5
    
    # For composite2, Total Power should be the key constraint
    measured_total_power2 = torch.sum(torch.abs(result2)**2).item()
    assert abs(measured_total_power2 - total_power) < 1e-4
    
    # The results should differ due to different application order
    assert not torch.allclose(result1, result2)


def test_power_constraint_with_asymmetric_signal():
    """Test power constraints with asymmetric signals."""
    # Create a signal with specific asymmetric distribution
    torch.manual_seed(123)
    n_samples = 1000
    # Create exponentially distributed values (long tail)
    signal = -torch.log(torch.rand(n_samples)) 
    
    # Apply constraints
    power = 2.0
    total_constraint = TotalPowerConstraint(total_power=power)
    avg_constraint = AveragePowerConstraint(average_power=power/n_samples)
    
    # Apply total power constraint
    total_result = total_constraint(signal)
    measured_total_power = torch.sum(total_result**2).item()
    assert abs(measured_total_power - power) < 1e-4
    
    # Apply average power constraint
    avg_result = avg_constraint(signal)
    measured_avg_power = torch.mean(avg_result**2).item()
    assert abs(measured_avg_power - power/n_samples) < 1e-5
    
    # Verify signal shape is preserved (relative distribution)
    # Calculate correlation between original and constrained signals
    total_corr = torch.corrcoef(torch.stack([signal, total_result]))[0, 1].item()
    avg_corr = torch.corrcoef(torch.stack([signal, avg_result]))[0, 1].item()
    
    # High correlation indicates shape preservation
    assert total_corr > 0.99
    assert avg_corr > 0.99


def test_papr_constraint_with_high_papr_signal():
    """Test PAPR constraint with signals having extreme PAPR values."""
    # Create a signal with very high PAPR (impulse-like)
    n_samples = 1000
    signal = torch.zeros(n_samples)
    signal[0] = 10.0  # Single large peak
    
    # Calculate original PAPR (should be very high)
    orig_mean_power = torch.mean(signal**2).item()
    orig_peak_power = torch.max(signal**2).item()
    orig_papr = orig_peak_power / (orig_mean_power + 1e-8)
    
    # Apply PAPR constraint with different max values
    for max_papr in [2.0, 5.0, 10.0]:
        constraint = PAPRConstraint(max_papr=max_papr)
        result = constraint(signal)
        
        # Calculate resulting PAPR
        mean_power = torch.mean(result**2).item()
        peak_power = torch.max(result**2).item()
        measured_papr = peak_power / (mean_power + 1e-8)
        
        # PAPR should be reduced and close to or below the target
        # Allow some tolerance since PAPR constraint uses an approximation
        assert measured_papr <= max_papr * 1.5
        assert measured_papr < orig_papr


def test_power_constraints_on_batched_data(multi_dimensional_signal):
    """Test power constraints applied to batched data."""
    # The multi_dimensional_signal has shape [batch_size, channels, height, width]
    # Test that constraints are applied across all dimensions except batch
    
    batch_size = multi_dimensional_signal.shape[0]
    total_elements_per_batch = np.prod(multi_dimensional_signal.shape[1:])
    power = 2.0
    
    # Apply total power constraint
    total_constraint = TotalPowerConstraint(total_power=power)
    total_result = total_constraint(multi_dimensional_signal)
    
    # Each batch sample should have the same total power
    for i in range(batch_size):
        batch_power = torch.sum(total_result[i]**2).item()
        assert abs(batch_power - power) < 1e-4
    
    # Apply average power constraint (per element)
    avg_power = power / total_elements_per_batch
    avg_constraint = AveragePowerConstraint(average_power=avg_power)
    avg_result = avg_constraint(multi_dimensional_signal)
    
    # Each batch sample should have the same average power
    for i in range(batch_size):
        batch_avg_power = torch.mean(avg_result[i]**2).item()
        assert abs(batch_avg_power - avg_power) < 1e-5