import pytest
import torch
from kaira.metrics.signal.snr import SignalToNoiseRatio

@pytest.fixture
def signal_data():
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
    noise_power = signal_power / (10**(snr_db/10))  # Calculate required noise power for desired SNR
    
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
