"""Tests for AWGN channel with improved test coverage."""
import pytest
import torch

from kaira.channels.analog import AWGNChannel


def test_awgn_channel_initialization():
    """Test proper initialization of AWGN channel with different parameters."""
    # Initialize with avg_noise_power
    channel = AWGNChannel(avg_noise_power=0.1)
    # Use isclose instead of direct equality for floating point comparison
    assert torch.isclose(channel.avg_noise_power, torch.tensor(0.1), rtol=1e-5)
    assert channel.snr_db is None

    # Initialize with snr_db
    channel = AWGNChannel(snr_db=10.0)
    assert channel.snr_db == 10.0
    assert channel.avg_noise_power is None

    # Test initialization error when neither parameter is provided
    with pytest.raises(ValueError, match="Either avg_noise_power or snr_db must be provided"):
        AWGNChannel()


def test_awgn_channel_with_real_input():
    """Test AWGN channel with real-valued input."""
    # Create a channel with fixed noise power
    noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Create a test signal
    x = torch.ones(1000)
    y = channel(x)
    
    # Output should have same shape as input
    assert y.shape == x.shape
    
    # Output should be real-valued
    assert not torch.is_complex(y)
    
    # Check statistics of the noise
    noise = y - x
    measured_noise_power = torch.mean(noise ** 2).item()
    
    # Noise power should be close to the specified value (with some statistical variation)
    assert abs(measured_noise_power - noise_power) < 0.02


def test_awgn_channel_with_complex_input():
    """Test AWGN channel with complex-valued input."""
    # Create a channel with fixed noise power
    noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Create a complex test signal
    x = torch.complex(torch.ones(1000), torch.zeros(1000))
    y = channel(x)
    
    # Output should have same shape as input
    assert y.shape == x.shape
    
    # Output should be complex-valued
    assert torch.is_complex(y)
    
    # Check statistics of the noise
    noise = y - x
    measured_noise_power = torch.mean(torch.abs(noise) ** 2).item()
    
    # For complex noise, power is split between real and imaginary parts,
    # so the variance of each component is noise_power/2
    assert abs(measured_noise_power - noise_power) < 0.02


def test_awgn_channel_with_snr():
    """Test AWGN channel with SNR-based noise power."""
    # Create a channel with SNR specification
    snr_db = 10.0  # 10 dB
    channel = AWGNChannel(snr_db=snr_db)
    
    # Create a test signal with known power
    x = torch.ones(1000) * 2.0  # Signal power = 4.0
    y = channel(x)
    
    # Calculate expected noise power based on SNR
    signal_power = 4.0
    expected_noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Check statistics of the noise
    noise = y - x
    measured_noise_power = torch.mean(noise ** 2).item()
    
    # Noise power should be close to the expected value
    assert abs(measured_noise_power - expected_noise_power) < 0.05


def test_awgn_channel_with_pregenerated_noise():
    """Test AWGN channel with pre-generated noise."""
    channel = AWGNChannel(avg_noise_power=0.1)
    
    # Create a test signal
    x = torch.ones(10)
    
    # Create pre-generated noise
    noise = torch.randn_like(x) * 0.2
    
    # Apply channel with pre-generated noise
    y = channel(x, noise=noise)
    
    # Output should be exactly input + noise
    assert torch.allclose(y, x + noise)


def test_awgn_channel_with_batched_input():
    """Test AWGN channel with batched input."""
    channel = AWGNChannel(avg_noise_power=0.1)
    
    # Create a batched test signal
    x = torch.ones(5, 100)  # 5 batches of 100 samples
    y = channel(x)
    
    # Output should have same shape as input
    assert y.shape == x.shape
    
    # Each batch should be processed independently
    for i in range(5):
        noise_i = y[i] - x[i]
        assert 0.05 < torch.mean(noise_i ** 2).item() < 0.15