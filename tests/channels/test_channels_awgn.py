import torch
import pytest
import numpy as np
from kaira.channels import AWGNChannel

@pytest.fixture
def input_signal():
    """Create a test signal for channel testing."""
    torch.manual_seed(42)
    return torch.randn(10, 100)  # [batch_size, seq_length]

@pytest.fixture
def complex_signal():
    """Create a complex test signal for channel testing."""
    torch.manual_seed(42)
    real = torch.randn(10, 100)
    imag = torch.randn(10, 100)
    return torch.complex(real, imag)

def test_awgn_channel_initialization():
    """Test that AWGNChannel requires either avg_noise_power or snr_db."""
    # Valid initializations
    channel1 = AWGNChannel(avg_noise_power=0.1)
    channel2 = AWGNChannel(snr_db=10.0)
    
    assert hasattr(channel1, 'avg_noise_power')
    assert hasattr(channel2, 'snr_db')
    
    # Invalid initialization - should raise error
    with pytest.raises(ValueError):
        AWGNChannel()

def test_awgn_channel_with_noise_power(input_signal):
    """Test AWGNChannel with specified noise power."""
    noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Run input through channel
    output = channel(input_signal)
    
    # Check shape preservation
    assert output.shape == input_signal.shape
    
    # Measure actual noise power
    noise = output - input_signal
    measured_noise_power = torch.mean(noise**2).item()
    
    # Allow for some statistical variation
    assert np.isclose(measured_noise_power, noise_power, rtol=0.2)

def test_awgn_channel_with_snr(input_signal):
    """Test AWGNChannel with specified SNR."""
    snr_db = 10.0  # 10dB SNR
    channel = AWGNChannel(snr_db=snr_db)
    
    # Run input through channel
    output = channel(input_signal)
    
    # Check shape preservation
    assert output.shape == input_signal.shape
    
    # Calculate actual SNR
    signal_power = torch.mean(input_signal**2).item()
    noise = output - input_signal
    noise_power = torch.mean(noise**2).item()
    measured_snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Allow for some statistical variation
    assert np.isclose(measured_snr_db, snr_db, rtol=0.3)

def test_awgn_channel_with_complex_input(complex_signal):
    """Test AWGNChannel with complex input."""
    noise_power = 0.2
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Run input through channel
    output = channel(complex_signal)
    
    # Check shape and type preservation
    assert output.shape == complex_signal.shape
    assert output.dtype == complex_signal.dtype
    
    # Calculate noise components
    noise = output - complex_signal
    real_noise_power = torch.mean(noise.real**2).item()
    imag_noise_power = torch.mean(noise.imag**2).item()
    
    # For complex AWGN, noise power is divided equally between real and imaginary parts
    # Total noise should be close to specified noise power
    total_noise_power = real_noise_power + imag_noise_power
    assert np.isclose(total_noise_power, noise_power, rtol=0.2)
    
    # Real and imaginary noise powers should be approximately equal
    assert np.isclose(real_noise_power, imag_noise_power, rtol=0.3)