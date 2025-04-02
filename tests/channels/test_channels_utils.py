import numpy as np
import torch

from kaira.channels import AWGNChannel, LaplacianChannel, PhaseNoiseChannel
from kaira.utils import snr_db_to_linear, snr_linear_to_db


def test_awgn_channel_noise_addition():
    """Test that AWGN channel adds the right amount of noise."""
    # Create a deterministic input signal
    input_signal = torch.ones(1000, 1)

    # Set specific SNR
    snr_db = 10.0  # 10 dB SNR

    # Create channel
    channel = AWGNChannel(snr_db=snr_db)

    # Apply channel
    output_signal = channel(input_signal)

    # Calculate actual SNR
    signal_power = torch.mean(input_signal**2).item()
    noise_power = torch.mean((output_signal - input_signal) ** 2).item()
    measured_snr = signal_power / noise_power
    measured_snr_db = 10 * np.log10(measured_snr)

    # Check that measured SNR is close to expected SNR
    assert abs(measured_snr_db - snr_db) < 1.0  # Allow for some statistical variation


def test_laplacian_channel_distribution():
    """Test that LaplacianChannel produces Laplacian distributed noise."""
    # Create an input signal
    input_signal = torch.zeros(10000, 1)

    # Set a fixed scale for deterministic testing
    scale = 1.0

    # Create channel
    channel = LaplacianChannel(scale=scale)

    # Apply channel to get noise
    output_signal = channel(input_signal)
    noise = output_signal  # Since input is zero, output is just noise

    # Convert to numpy for statistics
    noise_np = noise.detach().cpu().numpy().flatten()

    # Check properties of Laplacian distribution
    # 1. Mean should be close to 0
    assert abs(np.mean(noise_np)) < 0.1

    # 2. Variance should be close to 2*scale^2
    expected_variance = 2 * scale**2
    # Allow more tolerance due to sampling variability
    assert abs(np.var(noise_np) - expected_variance) < 1.5


def test_phase_noise_channel_distribution():
    """Test that PhaseNoiseChannel adds the expected amount of phase noise."""
    # Create a complex input signal with unit magnitude
    batch_size = 1000
    theta = torch.linspace(0, 2 * np.pi, batch_size).unsqueeze(1)
    input_signal = torch.complex(torch.cos(theta), torch.sin(theta))

    # Set phase noise standard deviation
    phase_std = 0.2  # radians

    # Create channel
    channel = PhaseNoiseChannel(phase_noise_std=phase_std)

    # Apply channel
    output_signal = channel(input_signal)

    # Calculate phase difference
    input_phase = torch.angle(input_signal)
    output_phase = torch.angle(output_signal)
    phase_diff = torch.remainder(output_phase - input_phase + np.pi, 2 * np.pi) - np.pi

    # Check statistics of phase noise
    measured_std = torch.std(phase_diff).item()

    # The measured std should be close to the configured std
    assert abs(measured_std - phase_std) < 0.05


def test_snr_conversion_functions():
    """Test SNR conversion functions."""
    # Test linear to dB conversion
    snr_linear = torch.tensor([1.0, 10.0, 100.0])
    snr_db = snr_linear_to_db(snr_linear)
    expected_db = torch.tensor([0.0, 10.0, 20.0])
    assert torch.allclose(snr_db, expected_db, rtol=1e-5)

    # Test dB to linear conversion
    snr_db = torch.tensor([0.0, 10.0, 20.0])
    snr_linear = snr_db_to_linear(snr_db)
    expected_linear = torch.tensor([1.0, 10.0, 100.0])
    assert torch.allclose(snr_linear, expected_linear, rtol=1e-5)
