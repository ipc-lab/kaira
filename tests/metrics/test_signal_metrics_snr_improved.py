"""Improved tests for Signal-to-Noise Ratio (SNR) metrics."""
import pytest
import torch
import numpy as np
from kaira.metrics.signal.snr import SignalToNoiseRatio


def test_snr_initialization():
    """Test SNR metric initialization with default and custom names."""
    # Default initialization
    snr_default = SignalToNoiseRatio()
    assert snr_default.name == "SNR"
    
    # Custom name
    snr_custom = SignalToNoiseRatio(name="CustomSNR")
    assert snr_custom.name == "CustomSNR"


def test_snr_calculation_scalar():
    """Test SNR calculation with basic scalar signals."""
    snr_metric = SignalToNoiseRatio()
    
    # Simple test case - clean signal with known noise
    signal = torch.tensor([1.0, -1.0, 1.0, -1.0])
    noise = torch.tensor([0.1, -0.1, 0.1, -0.1])
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Calculate expected SNR
    signal_power = torch.mean(signal**2)  # Should be 1.0
    noise_power = torch.mean(noise**2)    # Should be 0.01
    expected_snr_db = 10 * torch.log10(signal_power / noise_power)  # Should be 20dB
    
    # Check result
    assert torch.isclose(snr_db, expected_snr_db, rtol=1e-4)


def test_snr_calculation_batched():
    """Test SNR calculation with batched signals."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a batch of signals
    batch_size = 3
    signal = torch.tensor([
        [1.0, -1.0, 1.0, -1.0],
        [0.5, -0.5, 0.5, -0.5],
        [2.0, -2.0, 2.0, -2.0]
    ])
    
    # Create batch of noise with different powers
    noise = torch.tensor([
        [0.1, -0.1, 0.1, -0.1],    # SNR should be high
        [0.25, -0.25, 0.25, -0.25], # SNR should be medium
        [1.0, -1.0, 1.0, -1.0]      # SNR should be low
    ])
    
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Verify shape
    assert snr_db.shape == torch.Size([batch_size])
    
    # Calculate expected SNRs
    expected_snrs = []
    for i in range(batch_size):
        signal_power = torch.mean(signal[i]**2)
        noise_power = torch.mean(noise[i]**2)
        expected_snr = 10 * torch.log10(signal_power / noise_power)
        expected_snrs.append(expected_snr)
    
    expected_snr_tensor = torch.tensor(expected_snrs)
    
    # Check results
    assert torch.allclose(snr_db, expected_snr_tensor, rtol=1e-4)
    
    # Verify that SNRs decrease as noise increases
    assert snr_db[0] > snr_db[1] > snr_db[2]


def test_snr_complex_signals():
    """Test SNR calculation with complex signals."""
    snr_metric = SignalToNoiseRatio()
    
    # Create complex signal
    signal = torch.complex(torch.tensor([1.0, 0.0, -1.0, 0.0]), 
                          torch.tensor([0.0, 1.0, 0.0, -1.0]))
    
    # Create complex noise
    noise = torch.complex(torch.tensor([0.1, 0.1, 0.1, 0.1]), 
                         torch.tensor([0.1, 0.1, 0.1, 0.1]))
    
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Calculate expected SNR for complex signals
    signal_power = torch.mean(torch.abs(signal)**2)
    noise_power = torch.mean(torch.abs(noise)**2)
    expected_snr_db = 10 * torch.log10(signal_power / noise_power)
    
    # Check result
    assert torch.isclose(snr_db, expected_snr_db, rtol=1e-4)


def test_snr_zero_noise():
    """Test SNR calculation with zero noise (perfect signal)."""
    snr_metric = SignalToNoiseRatio()
    
    # Create signal
    signal = torch.tensor([1.0, -1.0, 1.0, -1.0])
    
    # Perfect reproduction (zero noise)
    noisy_signal = signal.clone()
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Result should be infinity
    assert torch.isinf(snr_db)
    assert snr_db > 0  # Positive infinity


def test_snr_zero_signal():
    """Test SNR calculation with zero signal."""
    snr_metric = SignalToNoiseRatio()
    
    # Zero signal
    signal = torch.zeros(4)
    
    # Some noise
    noise = torch.tensor([0.1, -0.1, 0.1, -0.1])
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Result should be very negative (approaching -infinity)
    # But due to epsilon in the calculation, it won't be exactly -infinity
    assert snr_db < -100


def test_snr_compute_with_stats():
    """Test compute_with_stats method of SNR metric."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a batch of signals with different SNRs
    signal = torch.tensor([
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0]
    ])
    
    # Different noise levels
    noise_levels = [0.1, 0.2, 0.3]
    noises = []
    for level in noise_levels:
        noise = torch.tensor([level, -level, level, -level])
        noises.append(noise)
    
    noisy_signal = signal.clone()
    for i, noise in enumerate(noises):
        noisy_signal[i] = signal[i] + noise
    
    # Calculate mean and std of SNR
    mean_snr, std_snr = snr_metric.compute_with_stats(signal, noisy_signal)
    
    # Calculate expected SNRs
    expected_snrs = []
    for i, level in enumerate(noise_levels):
        signal_power = torch.mean(signal[i]**2)
        noise_power = level**2
        expected_snr = 10 * torch.log10(signal_power / noise_power)
        expected_snrs.append(expected_snr)
    
    expected_mean = sum(expected_snrs) / len(expected_snrs)
    expected_std = torch.tensor(expected_snrs).std()
    
    # Check results
    assert torch.isclose(mean_snr, torch.tensor(expected_mean), rtol=1e-3)
    assert torch.isclose(std_snr, torch.tensor(expected_std), rtol=1e-3)


def test_snr_reset():
    """Test reset method of SNR metric."""
    snr_metric = SignalToNoiseRatio()
    
    # SNR has no state to reset, so this is a no-op
    # Just verify it doesn't raise an exception
    snr_metric.reset()
    
    # Verify the metric still works after reset
    signal = torch.tensor([1.0, -1.0])
    noise = torch.tensor([0.1, -0.1])
    noisy_signal = signal + noise
    
    # This should not raise an exception
    snr_db = snr_metric(signal, noisy_signal)
    assert snr_db is not None


def test_snr_different_shapes():
    """Test SNR calculation with signals of different shapes."""
    snr_metric = SignalToNoiseRatio()
    
    # Test with 1D signal
    signal_1d = torch.tensor([1.0, -1.0, 1.0, -1.0])
    noise_1d = torch.tensor([0.1, -0.1, 0.1, -0.1])
    noisy_signal_1d = signal_1d + noise_1d
    
    snr_1d = snr_metric(signal_1d, noisy_signal_1d)
    assert snr_1d.dim() == 0  # Scalar output
    
    # Test with 2D signal (single batch)
    signal_2d = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    noise_2d = torch.tensor([[0.1, -0.1, 0.1, -0.1]])
    noisy_signal_2d = signal_2d + noise_2d
    
    snr_2d = snr_metric(signal_2d, noisy_signal_2d)
    assert snr_2d.shape == torch.Size([1])  # Single batch output
    
    # Values should be the same
    assert torch.isclose(snr_1d, snr_2d[0], rtol=1e-5)