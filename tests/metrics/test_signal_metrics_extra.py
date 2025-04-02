def test_signal_to_noise_ratio_db():
    """Test SignalToNoiseRatio with dB input/output."""
    snr = SignalToNoiseRatio(mode="db")

    # Create signal and noise
    signal = torch.ones(10) * 2.0  # Signal with power 4.0
    noise = torch.ones(10) * 1.0  # Noise with power 1.0
    noisy_signal = signal + noise

    # SNR should be 10*log10(4/1) = 6.02 dB
    result = snr(noisy_signal, signal)
    assert round(result.item(), 1) == 6.0

    # Test with zero signal (edge case)
    zero_signal = torch.zeros(10)
    result = snr(noise, zero_signal)
    assert result.item() == float("-inf")  # SNR is -infinity for zero signal

    # Test with linear mode
    snr = SignalToNoiseRatio(mode="linear")
    result = snr(noisy_signal, signal)
    assert round(result.item(), 1) == 4.0  # Signal power / noise power


def test_signal_to_noise_ratio_linear():
    """Test SignalToNoiseRatio with linear input/output."""
    snr = SignalToNoiseRatio(mode="linear")

    # Create signal and noise
    signal = torch.ones(10) * 2.0  # Signal with power 4.0
    noise = torch.ones(10) * 1.0  # Noise with power 1.0
    noisy_signal = signal + noise

    # Signal power = 4, noise power = 1
    # SNR = 4/1 = 4
    result = snr(noisy_signal, signal)
    assert result.item() == pytest.approx(4.0, abs=0.1)
