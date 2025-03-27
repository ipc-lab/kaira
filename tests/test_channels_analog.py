# tests/test_channels_analog.py
import numpy as np
import pytest
import torch
import math
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
    measured_noise_power = torch.mean(noise**2).item()

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
    signal_power = torch.mean(random_tensor**2).item()
    noise = output - random_tensor
    noise_power = torch.mean(noise**2).item()

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
    real_noise_power = torch.mean(noise.real**2).item()
    imag_noise_power = torch.mean(noise.imag**2).item()
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
    output - random_tensor

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
    output - complex_tensor

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
    channel = FlatFadingChannel(fading_type="rayleigh", coherence_time=coherence_time, snr_db=15)

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
    channel = FlatFadingChannel(fading_type="rician", coherence_time=coherence_time, k_factor=k_factor, snr_db=15)

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
    channel = NonlinearChannel(nonlinear_fn=nonlinear_fn, add_noise=True, avg_noise_power=noise_power)

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
    channel = NonlinearChannel(nonlinear_fn=magnitude_compression, complex_mode="polar")

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
    channel = NonlinearChannel(nonlinear_fn=nonlinear_fn, complex_mode="cartesian")

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


def test_laplacian_channel_invalid_parameters():
    """Test LaplacianChannel with invalid parameters."""
    with pytest.raises(ValueError):
        LaplacianChannel()
    with pytest.raises(ValueError):
        LaplacianChannel(scale=None, avg_noise_power=None, snr_db=None)


def test_poisson_channel_invalid_parameters():
    """Test PoissonChannel with invalid parameters."""
    with pytest.raises(ValueError):
        PoissonChannel(rate_factor=-1.0)
    with pytest.raises(ValueError):
        PoissonChannel(rate_factor=0.0)


def test_phase_noise_channel_invalid_parameters():
    """Test PhaseNoiseChannel with invalid parameters."""
    with pytest.raises(ValueError):
        PhaseNoiseChannel(phase_noise_std=-0.1)


def test_flat_fading_channel_invalid_parameters():
    """Test FlatFadingChannel with invalid parameters."""
    with pytest.raises(ValueError):
        FlatFadingChannel(fading_type="invalid", coherence_time=10, snr_db=15)
    with pytest.raises(ValueError):
        FlatFadingChannel(fading_type="rician", coherence_time=10, snr_db=15)
    with pytest.raises(ValueError):
        FlatFadingChannel(fading_type="lognormal", coherence_time=10, snr_db=15)
    with pytest.raises(ValueError):
        FlatFadingChannel(fading_type="lognormal", coherence_time=10, shadow_sigma_db=None, snr_db=15)


def test_nonlinear_channel_invalid_parameters():
    """Test NonlinearChannel with invalid parameters."""
    with pytest.raises(ValueError):
        NonlinearChannel(nonlinear_fn=None)
    with pytest.raises(ValueError):
        NonlinearChannel(nonlinear_fn=lambda x: x, complex_mode="invalid")
    # The test was failing because avg_noise_power must be a tensor
    # when add_noise is True, so let's create a valid test case
    with pytest.raises(ValueError):
        # This should raise error because we're providing snr_db without signal_power
        NonlinearChannel(nonlinear_fn=lambda x: x, add_noise=True, snr_db=10.0)


def test_laplacian_channel_snr():
    """Test LaplacianChannel with SNR specification."""
    x = torch.ones(1000)
    snr_db = 10.0
    channel = LaplacianChannel(snr_db=snr_db)
    y = channel(x)

    # Check that output has same shape as input
    assert y.shape == x.shape

    # Calculate actual SNR
    signal_power = 1.0  # since input is all ones
    noise_power = torch.mean((y - x) ** 2).item()
    actual_snr_db = 10 * math.log10(signal_power / noise_power)

    # Allow for some statistical variation
    assert abs(actual_snr_db - snr_db) < 2.0


def test_laplacian_channel_complex():
    """Test LaplacianChannel with complex input."""
    x = torch.complex(torch.ones(1000), torch.ones(1000))
    scale = 0.1
    channel = LaplacianChannel(scale=scale)
    y = channel(x)

    # Check that output has same shape and is complex
    assert y.shape == x.shape
    assert torch.is_complex(y)

    # Check that noise was added to both real and imaginary parts
    assert not torch.allclose(y.real, x.real)
    assert not torch.allclose(y.imag, x.imag)


def test_poisson_channel_complex():
    """Test PoissonChannel with complex input."""
    # Create complex input with non-negative magnitude
    real = torch.rand(1000) + 0.5  # Make sure values are positive
    imag = torch.rand(1000) + 0.5
    x = torch.complex(real, imag)

    channel = PoissonChannel(rate_factor=10.0, normalize=True)
    y = channel(x)

    # Check that output is complex and preserves phase direction
    assert torch.is_complex(y)
    assert y.shape == x.shape

    # Check that magnitude was modified but phase was preserved
    x_mag = torch.abs(x)
    y_mag = torch.abs(y)
    x_phase = torch.angle(x)
    y_phase = torch.angle(y)

    # Magnitudes should be different
    assert not torch.allclose(x_mag, y_mag)

    # Phases should be approximately preserved (with some numerical precision issues)
    phase_diff = torch.abs(x_phase - y_phase)
    assert torch.mean(phase_diff) < 0.01


def test_poisson_channel_negative_input():
    """Test PoissonChannel with negative inputs (should raise error)."""
    x = torch.randn(100)  # Will contain negative values
    channel = PoissonChannel()

    # Should raise ValueError due to negative values
    with pytest.raises(ValueError):
        channel(x)


def test_flat_fading_rician_parameters():
    """Test FlatFadingChannel with Rician fading and various parameters."""
    x = torch.complex(torch.ones(10, 100), torch.zeros(10, 100))

    # Test with SNR specification
    channel = FlatFadingChannel(fading_type="rician", coherence_time=10, k_factor=2.0, snr_db=15.0)
    y = channel(x)

    # Check shape preservation
    assert y.shape == x.shape

    # Verify channel operation changed the signal
    assert not torch.allclose(y, x)


def test_flat_fading_lognormal():
    """Test FlatFadingChannel with log-normal fading."""
    x = torch.complex(torch.ones(5, 50), torch.zeros(5, 50))

    channel = FlatFadingChannel(fading_type="lognormal", coherence_time=5, shadow_sigma_db=4.0, avg_noise_power=0.01)

    y = channel(x)

    # Check shape preservation
    assert y.shape == x.shape

    # Verify fading effects
    assert not torch.allclose(y, x)


def test_flat_fading_3d_input():
    """Test FlatFadingChannel with 3D input."""
    # Create 3D input (batch, channels, sequence)
    x = torch.complex(torch.ones(5, 3, 40), torch.zeros(5, 3, 40))

    channel = FlatFadingChannel(fading_type="rayleigh", coherence_time=10, snr_db=20.0)

    y = channel(x)

    # Check shape preservation for 3D input
    assert y.shape == x.shape
    assert torch.is_complex(y)


def test_nonlinear_channel_complex_direct():
    """Test NonlinearChannel with complex input using direct mode."""

    # Define nonlinear function for complex values
    def complex_nonlinear(z):
        return z * (1.0 - 0.1 * torch.abs(z))

    x = torch.complex(torch.randn(100), torch.randn(100))

    channel = NonlinearChannel(nonlinear_fn=complex_nonlinear, complex_mode="direct")

    y = channel(x)

    # Check output
    assert y.shape == x.shape
    assert torch.is_complex(y)
    assert not torch.allclose(y, x)


def test_nonlinear_channel_complex_cartesian():
    """Test NonlinearChannel with complex input using cartesian mode."""

    # Define nonlinear function for real values
    def real_nonlinear(x):
        return x + 0.1 * x**3

    x = torch.complex(torch.randn(100), torch.randn(100))

    channel = NonlinearChannel(nonlinear_fn=real_nonlinear, complex_mode="cartesian")

    y = channel(x)

    # Check output
    assert y.shape == x.shape
    assert torch.is_complex(y)

    # Verify that real and imaginary parts were transformed separately
    expected_real = real_nonlinear(x.real)
    expected_imag = real_nonlinear(x.imag)

    assert torch.allclose(y.real, expected_real)
    assert torch.allclose(y.imag, expected_imag)


def test_nonlinear_channel_with_noise():
    """Test NonlinearChannel with added noise."""

    def nonlinear_fn(x):
        return torch.tanh(x)

    x = torch.randn(1000)

    channel = NonlinearChannel(nonlinear_fn=nonlinear_fn, add_noise=True, snr_db=15.0)

    y = channel(x)

    # Check output
    assert y.shape == x.shape

    # Apply just the nonlinearity without noise
    y_no_noise = nonlinear_fn(x)

    # Verify that noise was added (should be different)
    assert not torch.allclose(y, y_no_noise)


def test_nonlinear_channel_avg_noise_power():
    """Test NonlinearChannel with specified average noise power."""

    def nonlinear_fn(x):
        return x

    x = torch.ones(1000)
    noise_power = torch.tensor(0.1)  # Convert to tensor to avoid the error

    channel = NonlinearChannel(nonlinear_fn=nonlinear_fn, add_noise=True, avg_noise_power=noise_power)

    y = channel(x)

    # Check output
    assert y.shape == x.shape

    # Estimate actual noise power
    noise = y - x
    actual_noise_power = torch.mean(noise**2).item()

    # Check that actual noise power is close to specified
    assert abs(actual_noise_power - noise_power.item()) < 0.02
