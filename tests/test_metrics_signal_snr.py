import pytest
import torch

from kaira.metrics.signal.snr import SignalToNoiseRatio


@pytest.fixture
def signal_data():
    """Fixture for creating sample signal data."""
    torch.manual_seed(42)
    signal = torch.randn(1, 1000)  # Original signal
    noise = 0.1 * torch.randn(1, 1000)  # Noise
    noisy_signal = signal + noise  # Noisy signal
    return signal, noisy_signal


def test_snr_initialization():
    snr = SignalToNoiseRatio()
    assert isinstance(snr, SignalToNoiseRatio)


def test_snr_computation(signal_data):
    signal, noisy_signal = signal_data
    snr = SignalToNoiseRatio()
    snr_value = snr(signal, noisy_signal)
    assert isinstance(snr_value, torch.Tensor)
    assert snr_value.ndim == 0  # Scalar output
    assert snr_value > 0  # SNR should be positive


@pytest.mark.parametrize("snr_db", [-10, 0, 10, 20])
def test_snr_db_values(signal_data, snr_db):
    signal, _ = signal_data

    # Create noisy signal with specific SNR using correct power calculation
    signal_power = torch.mean(signal**2).item()
    noise_power = signal_power / (10 ** (snr_db / 10))  # Calculate required noise power for desired SNR

    # Generate noise with the exact power needed
    noise = torch.randn_like(signal)
    noise_scale = torch.sqrt(torch.tensor(noise_power) / torch.mean(noise**2))
    scaled_noise = noise * noise_scale

    # Create noisy signal
    noisy_signal = signal + scaled_noise

    # Calculate SNR
    snr = SignalToNoiseRatio()
    snr_value = snr(signal, noisy_signal)

    # Check the calculated SNR is close to the expected value
    assert abs(snr_value.item() - snr_db) < 1.0  # Allow for some numerical error


def test_snr_perfect_signal():
    signal = torch.randn(1, 100)
    snr = SignalToNoiseRatio()
    snr_value = snr(signal, signal)
    assert torch.isinf(snr_value)


def test_snr_batch_computation():
    batch_size = 3
    signal = torch.randn(batch_size, 100)
    noise = 0.1 * torch.randn_like(signal)
    noisy_signal = signal + noise

    snr = SignalToNoiseRatio()
    snr_values = snr(noisy_signal, signal)
    assert snr_values.shape == (batch_size,)
    assert torch.all(snr_values > 0)


def test_snr_complex_signal():
    signal = torch.complex(torch.randn(1, 100), torch.randn(1, 100))
    noise = 0.1 * torch.complex(torch.randn_like(signal.real), torch.randn_like(signal.imag))
    noisy_signal = signal + noise

    snr = SignalToNoiseRatio()
    snr_value = snr(noisy_signal, signal)
    assert isinstance(snr_value, torch.Tensor)
    assert snr_value > 0


def test_snr_with_different_dimensions():
    """Test SNR computation with inputs of different dimensions."""
    # Create 1D signals
    signal_1d = torch.randn(100)
    noise_1d = 0.1 * torch.randn(100)
    noisy_1d = signal_1d + noise_1d
    
    # Create 2D signals (batch of 1D signals)
    signal_2d = torch.randn(2, 100)
    noise_2d = 0.1 * torch.randn(2, 100)
    noisy_2d = signal_2d + noise_2d
    
    # Create 3D signals (e.g., spectrograms)
    signal_3d = torch.randn(2, 10, 10)
    noise_3d = 0.1 * torch.randn(2, 10, 10)
    noisy_3d = signal_3d + noise_3d
    
    # Compute SNR for each
    snr_metric = SignalToNoiseRatio()
    
    snr_1d = snr_metric(signal_1d, noisy_1d)  # Corrected order: original signal first, noisy second
    snr_2d = snr_metric(signal_2d, noisy_2d)
    snr_3d = snr_metric(signal_3d, noisy_3d)
    
    # Check that results are valid
    assert isinstance(snr_1d, torch.Tensor)
    assert isinstance(snr_2d, torch.Tensor)
    assert isinstance(snr_3d, torch.Tensor)
    
    # Check dimensions
    assert snr_1d.ndim == 0  # Scalar
    
    # Per the implementation, if signal has batch dimension > 1, output is a tensor with batch size
    assert snr_2d.ndim == 1  # Vector (one SNR per batch item)
    assert snr_3d.ndim == 1  # Vector (one SNR per batch item)


def test_snr_with_zero_signal():
    """Test SNR computation with zero signal (edge case)."""
    # Create zero signal
    signal = torch.zeros(100)
    noise = torch.randn(100)  # Some noise
    noisy_signal = signal + noise  # Noisy signal is just noise
    
    snr_metric = SignalToNoiseRatio()
    
    # Compute SNR between zero signal and noise
    snr = snr_metric(signal, noisy_signal)
    
    # SNR should be very small (approaching negative infinity) when signal is zero
    assert snr < -100  # Very low SNR


def test_snr_db_calculation():
    """Test SNR computation in decibels."""
    # Create signals with known SNR
    signal = torch.ones(100)  # Signal with power = 1
    noise = 0.1 * torch.ones(100)  # Noise with power = 0.01
    noisy_signal = signal + noise
    
    # Expected SNR in linear scale: signal_power / noise_power = 1 / 0.01 = 100
    # Expected SNR in dB: 10 * log10(100) = 20 dB
    
    snr_metric = SignalToNoiseRatio()
    snr_value = snr_metric(signal, noisy_signal)
    
    # Verify the result is in dB
    assert torch.isclose(snr_value, torch.tensor(20.0), rtol=1e-1)
