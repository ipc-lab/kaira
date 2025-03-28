"""Tests for Signal-to-Noise Ratio (SNR) metric with improved coverage."""
import numpy as np
import pytest
import torch

from kaira.metrics.signal.snr import SignalToNoiseRatio


def test_snr_initialization():
    """Test initialization of SignalToNoiseRatio metric."""
    # Initialize with default name
    snr = SignalToNoiseRatio()
    assert snr.name == "SNR"
    
    # Initialize with custom name
    custom_snr = SignalToNoiseRatio(name="CustomSNR")
    assert custom_snr.name == "CustomSNR"


def test_snr_with_clean_signal():
    """Test SNR calculation with clean signal (no noise)."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a clean signal with itself (zero noise)
    signal = torch.ones(100)
    
    # The SNR should be very high (approaching infinity)
    result = snr_metric(signal, signal)
    assert torch.isinf(result)
    assert result > 0  # Positive infinity


def test_snr_with_known_noise_level():
    """Test SNR calculation with controlled noise level."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a signal with known power
    signal = torch.ones(1000) * 2.0  # Signal power = 4.0
    
    # Create known noise with power 0.4 (Actual SNR = 10 dB)
    noise = torch.randn_like(signal) * np.sqrt(0.4)
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Should be close to 10 dB
    assert 9.5 < snr_db < 10.5


def test_snr_with_complex_signal():
    """Test SNR calculation with complex-valued signal."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a complex signal
    signal = torch.complex(torch.ones(100), torch.zeros(100))
    
    # Add known noise (SNR = 6 dB)
    noise_power = 0.25  # For 6 dB
    noise = torch.complex(
        torch.randn_like(signal.real) * np.sqrt(noise_power/2),
        torch.randn_like(signal.imag) * np.sqrt(noise_power/2)
    )
    noisy_signal = signal + noise
    
    # Calculate SNR
    snr_db = snr_metric(signal, noisy_signal)
    
    # Should be close to 6 dB
    assert 5.5 < snr_db < 6.5


def test_snr_with_batched_signals():
    """Test SNR calculation with batched signals."""
    snr_metric = SignalToNoiseRatio()
    
    # Create a batch of signals with different powers
    batch_size = 3
    signal = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],  # Power = 1.0
        [2.0, 2.0, 2.0, 2.0],  # Power = 4.0
        [3.0, 3.0, 3.0, 3.0],  # Power = 9.0
    ])
    
    # Create noisy signals with different SNRs
    noise = torch.tensor([
        [0.1, -0.1, 0.1, -0.1],    # Noise power = 0.01 (SNR = 20 dB)
        [0.2, -0.2, 0.2, -0.2],    # Noise power = 0.04 (SNR = 20 dB)
        [0.9, -0.9, 0.9, -0.9],    # Noise power = 0.81 (SNR = 10.45 dB)
    ])
    
    noisy_signal = signal + noise
    
    # Calculate SNR for each batch element
    snr_values = snr_metric(signal, noisy_signal)
    
    # Should have one SNR value per batch element
    assert snr_values.shape == torch.Size([batch_size])
    
    # Check individual SNR values
    assert 19.5 < snr_values[0] < 20.5  # First signal: ~20 dB
    assert 19.5 < snr_values[1] < 20.5  # Second signal: ~20 dB
    assert 10.0 < snr_values[2] < 11.0  # Third signal: ~10.45 dB


def test_snr_compute_with_stats():
    """Test compute_with_stats method for aggregated statistics."""
    snr_metric = SignalToNoiseRatio()
    
    # Create signals with different SNRs
    clean_signal = torch.ones((5, 100))  # Batch of 5 signals
    
    # Create 5 different noise levels
    noise_levels = [0.1, 0.02, 0.4, 0.015, 0.2]  # Corresponds to different SNRs
    
    noisy_signal = clean_signal.clone()
    for i, level in enumerate(noise_levels):
        noise = torch.randn(100) * np.sqrt(level)
        noisy_signal[i] = clean_signal[i] + noise
    
    # Compute with statistics
    mean_snr, std_snr = snr_metric.compute_with_stats(clean_signal, noisy_signal)
    
    # Check that mean and std are computed and are reasonable
    assert mean_snr.dim() == 0  # Should be a scalar
    assert std_snr.dim() == 0  # Should be a scalar
    
    # Mean SNR should be positive
    assert mean_snr > 0
    
    # Standard deviation should be non-negative
    assert std_snr >= 0