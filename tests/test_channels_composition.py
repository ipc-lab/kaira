"""Tests for sequential channel composition."""
import pytest
import torch

from kaira.channels import AWGNChannel, PhaseNoiseChannel, RayleighFadingChannel, PerfectChannel


@pytest.fixture
def complex_signal():
    """Fixture providing a complex signal for testing."""
    torch.manual_seed(42)
    n_samples = 1000
    real = torch.randn(n_samples)
    imag = torch.randn(n_samples)
    return torch.complex(real, imag)


def test_sequential_channel_composition(complex_signal):
    """Test that channels can be applied in sequence to model composite effects."""
    # Create individual channels
    phase_noise = PhaseNoiseChannel(phase_noise_std=0.1)
    fading = RayleighFadingChannel()
    awgn = AWGNChannel(snr_db=15)
    
    # Apply channels in sequence
    output1 = awgn(fading(phase_noise(complex_signal)))
    
    # Check that output shape matches input
    assert output1.shape == complex_signal.shape
    assert torch.is_complex(output1)
    
    # The composite channel should change the signal (not be identity)
    assert not torch.allclose(output1, complex_signal)


def test_channel_composition_order_matters(complex_signal):
    """Test that the order of channel composition affects the output."""
    # Create channels
    phase_noise = PhaseNoiseChannel(phase_noise_std=0.2)
    awgn = AWGNChannel(snr_db=10)
    
    # Apply in different orders
    order1 = awgn(phase_noise(complex_signal))
    order2 = phase_noise(awgn(complex_signal))
    
    # The outputs should be different due to different application order
    assert not torch.allclose(order1, order2)


def test_perfect_channel_in_composition(complex_signal):
    """Test that PerfectChannel correctly passes through the signal in compositions."""
    awgn = AWGNChannel(snr_db=15)
    perfect = PerfectChannel()
    
    # Apply with perfect channel in the middle
    output1 = awgn(complex_signal)
    output2 = awgn(perfect(complex_signal))
    
    # The outputs should be identical since perfect channel is identity
    assert torch.allclose(output1, output2)


def test_multiple_awgn_channel_composition(complex_signal):
    """Test that multiple AWGN channels compose to create higher noise level."""
    # Create two AWGN channels with the same SNR
    snr_db = 20
    awgn1 = AWGNChannel(snr_db=snr_db)
    awgn2 = AWGNChannel(snr_db=snr_db)
    
    # Apply a single channel
    output1 = awgn1(complex_signal)
    
    # Apply two channels in sequence
    output2 = awgn2(awgn1(complex_signal))
    
    # Calculate signal power
    signal_power = torch.mean(torch.abs(complex_signal)**2).item()
    
    # Calculate noise power in each case
    noise_power1 = torch.mean(torch.abs(output1 - complex_signal)**2).item()
    noise_power2 = torch.mean(torch.abs(output2 - complex_signal)**2).item()
    
    # Second case should have approximately twice the noise power
    # (allowing for some statistical variation)
    assert 1.5 < noise_power2 / noise_power1 < 2.5


def test_phase_and_fading_composition(complex_signal):
    """Test composition of phase noise and fading channels."""
    # Create channels
    phase_noise = PhaseNoiseChannel(phase_noise_std=0.1)
    fading = RayleighFadingChannel()
    
    # Apply channels
    output = fading(phase_noise(complex_signal))
    
    # Output should be complex
    assert torch.is_complex(output)
    
    # Shape should be preserved
    assert output.shape == complex_signal.shape
    
    # Signal should be different after applying channels
    assert not torch.allclose(output, complex_signal)
    
    # The amplitude distribution should be changed by Rayleigh fading
    original_amp = torch.abs(complex_signal)
    output_amp = torch.abs(output)
    
    # Mean amplitude should change due to Rayleigh fading
    assert abs(torch.mean(output_amp).item() - torch.mean(original_amp).item()) > 0.1