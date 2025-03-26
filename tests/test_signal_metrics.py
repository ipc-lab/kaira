# tests/test_signal_metrics.py
import pytest
import torch

from kaira.metrics.signal import (
    BER,
    BLER,
    FER,
    SER,
    SNR,
    BitErrorRate,
    BlockErrorRate,
    FrameErrorRate,
    SignalToNoiseRatio,
    SymbolErrorRate,
)


def test_bit_error_rate_computation():
    """Test BitErrorRate computation."""
    # Create test data - 75% match, 25% errors
    transmitted = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).float()
    received = torch.tensor([0, 1, 1, 1, 0, 0, 0, 1]).float()
    
    # Initialize BER metric
    ber_metric = BitErrorRate()
    
    # Test forward computation
    error_rate = ber_metric(transmitted, received)
    assert error_rate.item() == 0.25
    
    # Test update and compute (accumulated statistics)
    ber_metric.reset()
    ber_metric.update(transmitted, received)
    assert ber_metric.compute().item() == 0.25
    
    # Test with different threshold
    ber_metric = BitErrorRate(threshold=0.7)
    received_analog = torch.tensor([0.1, 0.9, 0.6, 0.8, 0.2, 0.3, 0.1, 0.9]).float()
    error_rate = ber_metric(transmitted, received_analog)
    assert error_rate.item() == 0.125  # Only one bit (0.6) is misclassified


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_bit_error_rate_with_different_thresholds(threshold):
    """Test BitErrorRate with different decision thresholds."""
    transmitted = torch.tensor([0, 1, 0, 1]).float()
    received = torch.tensor([0.2, 0.6, 0.4, 0.8]).float()
    
    ber_metric = BitErrorRate(threshold=threshold)
    error_rate = ber_metric(transmitted, received)
    
    # Calculate expected errors based on threshold
    transmitted_bits = (transmitted > threshold).bool()
    received_bits = (received > threshold).bool()
    expected_errors = (transmitted_bits != received_bits).float().mean().item()
    
    assert error_rate.item() == expected_errors


def test_bit_error_rate_batched():
    """Test BitErrorRate with batched data."""
    # Create batched test data (2 samples)
    transmitted = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]).float()
    received = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1]]).float()
    
    ber_metric = BitErrorRate()
    error_rate = ber_metric(transmitted, received)
    
    # Check shape and values
    assert error_rate.shape == torch.Size([2])
    assert error_rate[0].item() == 0.25  # 1/4 errors in first sample
    assert error_rate[1].item() == 0.5   # 2/4 errors in second sample


def test_block_error_rate_computation():
    """Test BlockErrorRate computation."""
    # Create test data - 1st block correct, 2nd has errors, 3rd correct, 4th has errors
    transmitted = torch.tensor([
        [1, 1, 1, 1],  # Block 1
        [0, 0, 0, 0],  # Block 2
        [1, 0, 1, 0],  # Block 3
        [0, 1, 0, 1],  # Block 4
    ]).float()
    
    received = torch.tensor([
        [1, 1, 1, 1],  # Block 1 - correct
        [0, 0, 1, 0],  # Block 2 - has error
        [1, 0, 1, 0],  # Block 3 - correct
        [1, 1, 0, 1],  # Block 4 - has error
    ]).float()
    
    # Initialize BLER metric
    bler_metric = BlockErrorRate()
    
    # Test forward computation
    block_error_rate = bler_metric(transmitted, received)
    assert block_error_rate.item() == 0.5  # 2 out of 4 blocks have errors
    
    # Test update and compute
    bler_metric.reset()
    bler_metric.update(transmitted, received)
    assert bler_metric.compute().item() == 0.5


def test_symbol_error_rate_computation():
    """Test SymbolErrorRate computation."""
    # Create test data with symbols (multi-bit values)
    transmitted = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    received = torch.tensor([0, 1, 3, 3, 1, 1, 0, 3])
    
    # Initialize SER metric
    ser_metric = SymbolErrorRate()
    
    # Test forward computation
    symbol_error_rate = ser_metric(transmitted, received)
    assert symbol_error_rate.item() == 0.375  # 3 out of 8 symbols are wrong
    
    # Test update and compute
    ser_metric.reset()
    ser_metric.update(transmitted, received)
    assert ser_metric.compute().item() == 0.375


def test_frame_error_rate_computation():
    """Test FrameErrorRate computation."""
    # Create test data - 2 frames, first has error, second is correct
    transmitted = torch.tensor([
        [1, 1, 1, 1],  # Frame 1
        [0, 0, 0, 0],  # Frame 2
    ]).float()
    
    received = torch.tensor([
        [1, 0, 1, 1],  # Frame 1 - has error
        [0, 0, 0, 0],  # Frame 2 - correct
    ]).float()
    
    # Initialize FER metric
    fer_metric = FrameErrorRate()
    
    # Test forward computation
    frame_error_rate = fer_metric(transmitted, received)
    assert frame_error_rate.item() == 0.5  # 1 out of 2 frames has errors
    
    # Test update and compute
    fer_metric.reset()
    fer_metric.update(transmitted, received)
    assert fer_metric.compute().item() == 0.5


def test_signal_to_noise_ratio_computation():
    """Test SignalToNoiseRatio computation."""
    # Create clean signal and noisy signal
    clean_signal = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0])
    noise = torch.tensor([0.1, -0.2, 0.05, 0.1, -0.05])
    noisy_signal = clean_signal + noise
    
    # Initialize SNR metric
    snr_metric = SignalToNoiseRatio()
    
    # Test forward computation
    snr_value = snr_metric(clean_signal, noisy_signal)
    
    # Calculate expected SNR
    signal_power = (clean_signal ** 2).mean()
    noise_power = (noise ** 2).mean()
    expected_snr = 10 * torch.log10(signal_power / noise_power)
    
    assert torch.isclose(snr_value, expected_snr, rtol=1e-4)
    
    # Test update and compute
    snr_metric.reset()
    snr_metric.update(clean_signal, noisy_signal)
    assert torch.isclose(snr_metric.compute(), expected_snr, rtol=1e-4)


def test_signal_to_noise_ratio_with_batch():
    """Test SignalToNoiseRatio with batched data."""
    # Create batched clean signal and noisy signal
    clean_signal = torch.tensor([
        [1.0, -1.0, 0.5],
        [0.5, -0.5, 0.0]
    ])
    noise = torch.tensor([
        [0.1, -0.1, 0.05],
        [0.05, 0.1, -0.05]
    ])
    noisy_signal = clean_signal + noise
    
    # Initialize SNR metric
    snr_metric = SignalToNoiseRatio()
    
    # Test forward computation
    snr_values = snr_metric(clean_signal, noisy_signal)
    
    # Check shape
    assert snr_values.shape == torch.Size([2])
    
    # Calculate expected SNR for each sample
    for i in range(2):
        signal_power = (clean_signal[i] ** 2).mean()
        noise_power = (noise[i] ** 2).mean()
        expected_snr = 10 * torch.log10(signal_power / noise_power)
        assert torch.isclose(snr_values[i], expected_snr, rtol=1e-4)


def test_metrics_aliases():
    """Test that metric aliases work properly."""
    assert BitErrorRate is BER
    assert BlockErrorRate is BLER
    assert FrameErrorRate is FER
    assert SymbolErrorRate is SER
    assert SignalToNoiseRatio is SNR