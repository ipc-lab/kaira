import pytest
import torch
import numpy as np

from kaira.utils.snr import (
    noise_power_to_snr,
    add_noise_for_snr,
    estimate_signal_power,
    snr_to_noise_power,
    snr_db_to_linear,
    snr_linear_to_db,
    calculate_snr
)


def test_noise_power_to_snr_float_inputs():
    """Test noise_power_to_snr with float inputs."""
    signal_power = 10.0
    noise_power = 1.0
    
    snr_db = noise_power_to_snr(signal_power, noise_power)
    
    assert isinstance(snr_db, torch.Tensor)
    assert torch.isclose(snr_db, torch.tensor(10.0), rtol=1e-5)


def test_noise_power_to_snr_tensor_inputs():
    """Test noise_power_to_snr with tensor inputs."""
    signal_power = torch.tensor([10.0, 20.0, 30.0])
    noise_power = torch.tensor([1.0, 2.0, 3.0])
    
    snr_db = noise_power_to_snr(signal_power, noise_power)
    
    assert isinstance(snr_db, torch.Tensor)
    assert torch.allclose(snr_db, torch.tensor([10.0, 10.0, 10.0]), rtol=1e-5)


def test_noise_power_to_snr_mixed_inputs():
    """Test noise_power_to_snr with mixed float and tensor inputs."""
    signal_power = 10.0
    noise_power = torch.tensor([1.0, 2.0, 5.0])
    
    snr_db = noise_power_to_snr(signal_power, noise_power)
    
    assert isinstance(snr_db, torch.Tensor)
    assert torch.allclose(snr_db, torch.tensor([10.0, 7.0, 3.0]), rtol=1e-5)


def test_noise_power_to_snr_zero_noise():
    """Test noise_power_to_snr raises error with zero noise power."""
    signal_power = 10.0
    noise_power = 0.0
    
    with pytest.raises(ValueError, match="Noise power cannot be zero"):
        noise_power_to_snr(signal_power, noise_power)
    
    # Also test with tensor containing zeros
    noise_power_tensor = torch.tensor([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="Noise power cannot be zero"):
        noise_power_to_snr(signal_power, noise_power_tensor)


def test_add_noise_for_snr_real_signal():
    """Test add_noise_for_snr with real signal."""
    torch.manual_seed(42)  # For reproducibility
    
    # Create a test signal
    signal = torch.ones(1000) * 2.0  # Uniform signal with power 4.0
    target_snr_db = 10.0  # 10 dB SNR
    
    # Add noise to achieve target SNR
    noisy_signal, noise = add_noise_for_snr(signal, target_snr_db)
    
    # Check shapes
    assert noisy_signal.shape == signal.shape
    assert noise.shape == signal.shape
    
    # Calculate achieved SNR
    signal_power = torch.mean(signal**2)
    noise_power = torch.mean(noise**2)
    achieved_snr_db = 10 * torch.log10(signal_power / noise_power)
    
    # Check that we're close to target SNR (allow some variance due to random noise)
    assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)


def test_add_noise_for_snr_complex_signal():
    """Test add_noise_for_snr with complex signal."""
    torch.manual_seed(42)  # For reproducibility
    
    # Create a complex test signal
    real_part = torch.ones(1000)
    imag_part = torch.ones(1000)
    signal = torch.complex(real_part, imag_part)
    
    target_snr_db = 15.0  # 15 dB SNR
    
    # Add noise to achieve target SNR
    noisy_signal, noise = add_noise_for_snr(signal, target_snr_db)
    
    # Check shapes
    assert noisy_signal.shape == signal.shape
    assert noise.shape == signal.shape
    
    # Check that result is complex
    assert torch.is_complex(noisy_signal)
    assert torch.is_complex(noise)
    
    # Calculate achieved SNR
    signal_power = torch.mean(torch.abs(signal)**2)
    noise_power = torch.mean(torch.abs(noise)**2)
    achieved_snr_db = 10 * torch.log10(signal_power / noise_power)
    
    # Check that we're close to target SNR (allow some variance due to random noise)
    assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)


def test_add_noise_for_snr_with_dimension():
    """Test add_noise_for_snr with specific dimension reduction."""
    torch.manual_seed(42)  # For reproducibility
    
    # Create a 2D test signal with different powers per channel
    signal = torch.zeros(3, 1000)
    signal[0, :] = 1.0  # Power 1.0
    signal[1, :] = 2.0  # Power 4.0
    signal[2, :] = 3.0  # Power 9.0
    
    target_snr_db = 20.0  # 20 dB SNR
    
    # Add noise to achieve target SNR, reducing along dimension 1
    noisy_signal, noise = add_noise_for_snr(signal, target_snr_db, dim=1)
    
    # Check that different noise powers were applied to each channel
    noise_power_per_channel = torch.mean(noise**2, dim=1)
    
    # Each channel should have a different noise power based on its signal power
    assert noise_power_per_channel[0] < noise_power_per_channel[1]
    assert noise_power_per_channel[1] < noise_power_per_channel[2]
    
    # Calculate achieved SNR per channel
    for i in range(3):
        signal_power = torch.mean(signal[i]**2)
        noise_power = torch.mean(noise[i]**2)
        achieved_snr_db = 10 * torch.log10(signal_power / noise_power)
        
        # Each channel should be close to target SNR
        assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)


def test_estimate_signal_power_real_signal():
    """Test estimate_signal_power with real signal."""
    # Create a signal with known power
    signal = torch.ones(100) * 2.0  # Power should be 4.0
    
    # Calculate power
    power = estimate_signal_power(signal)
    
    assert isinstance(power, torch.Tensor)
    assert torch.isclose(power, torch.tensor(4.0))
    
    # Test with dimension reduction
    signal_2d = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
    power_per_dim = estimate_signal_power(signal_2d, dim=1)
    
    assert power_per_dim.shape == torch.Size([3])
    assert torch.allclose(power_per_dim, torch.tensor([1.0, 4.0, 9.0]))
    
    # Test with keepdim=True
    power_keepdim = estimate_signal_power(signal_2d, dim=1, keepdim=True)
    assert power_keepdim.shape == torch.Size([3, 1])


def test_estimate_signal_power_complex_signal():
    """Test estimate_signal_power with complex signal."""
    # Create a complex signal with known power
    real_part = torch.ones(100)
    imag_part = torch.ones(100)
    signal = torch.complex(real_part, imag_part)  # |1+1j|^2 = 2, so power should be 2.0
    
    # Calculate power
    power = estimate_signal_power(signal)
    
    assert isinstance(power, torch.Tensor)
    assert torch.isclose(power, torch.tensor(2.0))
    
    # Test with varying amplitude complex signal
    real_part = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
    imag_part = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
    complex_signal = torch.complex(real_part, imag_part)
    
    # Calculate power per first dimension with keepdim=True
    power_per_dim = estimate_signal_power(complex_signal, dim=1, keepdim=True)
    
    # For complex numbers |a+bj|^2 = a^2 + b^2
    # So for [1+1j, 2+2j, 3+3j] we expect powers [2, 8, 18]
    assert power_per_dim.shape == torch.Size([3, 1])
    assert torch.allclose(power_per_dim, torch.tensor([[2.0], [8.0], [18.0]]))


def test_complete_snr_pipeline():
    """Test a complete pipeline of SNR-related functions."""
    torch.manual_seed(42)  # For reproducibility
    
    # Start with a clean signal
    original_signal = torch.randn(5, 1000)
    
    # Target SNR
    target_snr_db = 15.0
    
    # Convert to linear
    target_snr_linear = snr_db_to_linear(target_snr_db)
    assert torch.isclose(snr_linear_to_db(target_snr_linear), torch.tensor(target_snr_db))
    
    # Add noise to achieve target SNR
    noisy_signal, _ = add_noise_for_snr(original_signal, target_snr_db)
    
    # Calculate achieved SNR
    measured_snr_db = calculate_snr(original_signal, noisy_signal)
    
    # Should be close to target (within reasonable tolerance due to randomness)
    assert torch.isclose(measured_snr_db, torch.tensor(target_snr_db), rtol=0.1)
    
    # Estimate signal power
    estimated_power = estimate_signal_power(original_signal)
    direct_power = torch.mean(original_signal**2)
    assert torch.isclose(estimated_power, direct_power)
    
    # Calculate required noise power for target SNR
    required_noise_power = snr_to_noise_power(estimated_power, target_snr_db)
    
    # Verify the relationship
    computed_snr = noise_power_to_snr(estimated_power, required_noise_power)
    assert torch.isclose(computed_snr, torch.tensor(target_snr_db))
