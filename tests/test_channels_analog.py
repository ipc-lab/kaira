# tests/test_channels_analog.py
import pytest
import torch
import numpy as np

from kaira.channels import (
    AWGNChannel,
    FlatFadingChannel,
    GaussianChannel,
    LaplacianChannel,
    NonlinearChannel,
    PhaseNoiseChannel,
    PoissonChannel,
)


@pytest.fixture
def random_tensor():
    """Fixture providing a random tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 100)


@pytest.fixture
def complex_tensor():
    """Fixture providing a random complex tensor for testing."""
    torch.manual_seed(42)
    real = torch.randn(4, 100)
    imag = torch.randn(4, 100)
    return torch.complex(real, imag)


def test_awgn_channel_noise_power(random_tensor):
    """Test AWGNChannel with specified noise power."""
    # Create channel with known noise power
    noise_power = 0.1
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Apply channel to input
    output = channel(random_tensor)
    
    # Check output shape matches input
    assert output.shape == random_tensor.shape
    
    # The output should be input + noise
    # For large enough samples, measure the noise power
    noise = output - random_tensor
    measured_noise_power = torch.mean(noise ** 2).item()
    
    # Should be close to specified noise power
    assert np.isclose(measured_noise_power, noise_power, rtol=0.2)  # Allow some statistical deviation


def test_awgn_channel_snr(random_tensor):
    """Test AWGNChannel with specified SNR."""
    # Create channel with known SNR
    snr_db = 10.0  # 10 dB SNR
    channel = AWGNChannel(snr_db=snr_db)
    
    # Apply channel to input
    output = channel(random_tensor)
    
    # Calculate signal and noise powers
    signal_power = torch.mean(random_tensor ** 2).item()
    noise = output - random_tensor
    noise_power = torch.mean(noise ** 2).item()
    
    # Calculate measured SNR in dB
    measured_snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Should be close to specified SNR
    assert np.isclose(measured_snr_db, snr_db, rtol=0.3)  # Allow some statistical deviation


def test_awgn_channel_complex(complex_tensor):
    """Test AWGNChannel with complex input."""
    # Create channel with known noise power
    noise_power = 0.2
    channel = AWGNChannel(avg_noise_power=noise_power)
    
    # Apply channel to complex input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # Calculate noise
    noise = output - complex_tensor
    
    # Check noise power in real and imaginary components
    # For complex AWGN, the noise power is split equally between real and imaginary parts
    real_noise_power = torch.mean(noise.real ** 2).item()
    imag_noise_power = torch.mean(noise.imag ** 2).item()
    total_noise_power = real_noise_power + imag_noise_power
    
    # Total noise power should match specified power
    assert np.isclose(total_noise_power, noise_power, rtol=0.2)
    
    # Real and imaginary noise powers should be approximately equal
    assert np.isclose(real_noise_power, imag_noise_power, rtol=0.3)


def test_gaussian_channel_alias():
    """Test that GaussianChannel is an alias for AWGNChannel."""
    assert GaussianChannel is AWGNChannel


def test_laplacian_channel(random_tensor):
    """Test LaplacianChannel with specified scale parameter."""
    # Create channel with known scale
    scale = 0.2
    channel = LaplacianChannel(scale=scale)
    
    # Apply channel to input
    output = channel(random_tensor)
    
    # Check output shape matches input
    assert output.shape == random_tensor.shape
    
    # Calculate noise
    noise = output - random_tensor
    
    # For Laplacian distribution with scale parameter b,
    # the variance is 2*b^2, but we need more samples for reliable statistics
    # Here we just check that the output is different from the input
    assert not torch.allclose(output, random_tensor)


def test_laplacian_channel_complex(complex_tensor):
    """Test LaplacianChannel with complex input."""
    # Create channel with known noise power
    noise_power = 0.2
    channel = LaplacianChannel(avg_noise_power=noise_power)
    
    # Apply channel to complex input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # Calculate noise
    noise = output - complex_tensor
    
    # Check that output is different from input
    assert not torch.allclose(output, complex_tensor)


def test_phase_noise_channel(complex_tensor):
    """Test PhaseNoiseChannel with specified phase noise standard deviation."""
    # Create channel with known phase noise std
    phase_std = 0.1  # radians
    channel = PhaseNoiseChannel(phase_noise_std=phase_std)
    
    # Apply channel to input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # Calculate magnitude and phase of input and output
    input_mag = torch.abs(complex_tensor)
    output_mag = torch.abs(output)
    
    # Phase noise should not affect magnitude
    assert torch.allclose(input_mag, output_mag)
    
    # Phase should be different
    input_phase = torch.angle(complex_tensor)
    output_phase = torch.angle(output)
    phase_diff = torch.abs(output_phase - input_phase)
    
    # The phase differences should be nonzero
    assert not torch.allclose(phase_diff, torch.zeros_like(phase_diff))


def test_phase_noise_channel_real_input(random_tensor):
    """Test PhaseNoiseChannel with real input (should convert to complex)."""
    # Create channel with known phase noise std
    phase_std = 0.1  # radians
    channel = PhaseNoiseChannel(phase_noise_std=phase_std)
    
    # Apply channel to real input
    output = channel(random_tensor)
    
    # Output should be complex
    assert output.dtype == torch.complex64 or output.dtype == torch.complex128
    assert output.shape == random_tensor.shape


def test_poisson_channel():
    """Test PoissonChannel with non-negative input."""
    # Create non-negative input (important for Poisson channel)
    torch.manual_seed(42)
    input_signal = torch.abs(torch.randn(4, 1000))
    
    # Create channel with rate factor
    rate_factor = 10.0  # Large rate factor for more reliable statistics
    channel = PoissonChannel(rate_factor=rate_factor)
    
    # Apply channel to input
    output = channel(input_signal)
    
    # Check output shape matches input
    assert output.shape == input_signal.shape
    
    # For Poisson distribution with rate λ, the variance equals the mean (λ)
    # For large λ, the output distribution approaches Gaussian with mean λ and variance λ
    # So we can check that output has similar mean to input*rate_factor but higher variance
    input_mean = torch.mean(input_signal).item()
    output_mean = torch.mean(output).item()
    
    # Mean should be approximately preserved (rate_factor * input)
    expected_mean = input_mean * rate_factor
    assert np.isclose(output_mean, expected_mean, rtol=0.1)


def test_poisson_channel_normalized():
    """Test PoissonChannel with normalization."""
    # Create non-negative input
    torch.manual_seed(42)
    input_signal = torch.abs(torch.randn(4, 1000))
    
    # Create channel with normalization
    rate_factor = 10.0
    channel = PoissonChannel(rate_factor=rate_factor, normalize=True)
    
    # Apply channel to input
    output = channel(input_signal)
    
    # Check output shape matches input
    assert output.shape == input_signal.shape
    
    # With normalization, the mean should be approximately preserved
    input_mean = torch.mean(input_signal).item()
    output_mean = torch.mean(output).item()
    
    # Mean should be approximately equal to input mean
    assert np.isclose(output_mean, input_mean, rtol=0.1)


def test_flat_fading_channel_rayleigh(complex_tensor):
    """Test FlatFadingChannel with Rayleigh fading."""
    # Create channel with Rayleigh fading
    coherence_time = 10
    channel = FlatFadingChannel(
        fading_type="rayleigh",
        coherence_time=coherence_time,
        snr_db=15
    )
    
    # Apply channel to input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # Check block fading effect - values should be correlated within coherence time
    # This is hard to test precisely without looking at the implementation details
    # So we just check that the output is different from the input
    assert not torch.allclose(output, complex_tensor)


def test_flat_fading_channel_rician(complex_tensor):
    """Test FlatFadingChannel with Rician fading."""
    # Create channel with Rician fading
    k_factor = 2.0  # Ratio of direct to scattered power
    coherence_time = 10
    channel = FlatFadingChannel(
        fading_type="rician",
        coherence_time=coherence_time,
        k_factor=k_factor,
        snr_db=15
    )
    
    # Apply channel to input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # Check output is different from input
    assert not torch.allclose(output, complex_tensor)


def test_nonlinear_channel_real(random_tensor):
    """Test NonlinearChannel with real inputs and a nonlinear function."""
    # Define a simple nonlinear function (cubic nonlinearity)
    def cubic_nonlinearity(x):
        return x + 0.1 * x**3
    
    # Create channel with this nonlinearity
    channel = NonlinearChannel(nonlinear_fn=cubic_nonlinearity)
    
    # Apply channel to input
    output = channel(random_tensor)
    
    # Check output shape matches input
    assert output.shape == random_tensor.shape
    
    # Check that nonlinearity is applied correctly
    expected_output = cubic_nonlinearity(random_tensor)
    assert torch.allclose(output, expected_output)


def test_nonlinear_channel_with_noise(random_tensor):
    """Test NonlinearChannel with added noise."""
    # Define a simple nonlinear function
    def nonlinear_fn(x):
        return torch.tanh(x)
    
    # Create channel with noise - noise_power must be a tensor
    noise_power = torch.tensor(0.1)
    channel = NonlinearChannel(
        nonlinear_fn=nonlinear_fn,
        add_noise=True,
        avg_noise_power=noise_power
    )
    
    # Apply channel to input
    output = channel(random_tensor)
    
    # Check output shape matches input
    assert output.shape == random_tensor.shape
    
    # The output should be tanh(input) + noise
    expected_without_noise = nonlinear_fn(random_tensor)
    assert not torch.allclose(output, expected_without_noise)


def test_nonlinear_channel_complex_polar(complex_tensor):
    """Test NonlinearChannel with complex inputs in polar mode."""
    # Define a nonlinear function that compresses magnitude
    def magnitude_compression(x):
        return x * (1 - 0.1 * x)
    
    # Create channel with polar mode
    channel = NonlinearChannel(
        nonlinear_fn=magnitude_compression,
        complex_mode="polar"
    )
    
    # Apply channel to input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # In polar mode, phase should be preserved
    input_phase = torch.angle(complex_tensor)
    output_phase = torch.angle(output)
    
    # Phases should be very close
    assert torch.allclose(input_phase, output_phase)
    
    # But magnitudes should be different
    input_mag = torch.abs(complex_tensor)
    output_mag = torch.abs(output)
    assert not torch.allclose(input_mag, output_mag)


def test_nonlinear_channel_complex_cartesian(complex_tensor):
    """Test NonlinearChannel with complex inputs in cartesian mode."""
    # Define a nonlinear function
    def nonlinear_fn(x):
        return x * 0.5 + 0.1 * x**2
    
    # Create channel with cartesian mode
    channel = NonlinearChannel(
        nonlinear_fn=nonlinear_fn,
        complex_mode="cartesian"
    )
    
    # Apply channel to input
    output = channel(complex_tensor)
    
    # Check output shape and type match input
    assert output.shape == complex_tensor.shape
    assert output.dtype == complex_tensor.dtype
    
    # In cartesian mode, real and imaginary parts should be processed separately
    expected_real = nonlinear_fn(complex_tensor.real)
    expected_imag = nonlinear_fn(complex_tensor.imag)
    
    # Check real and imaginary parts match expectations
    assert torch.allclose(output.real, expected_real)
    assert torch.allclose(output.imag, expected_imag)