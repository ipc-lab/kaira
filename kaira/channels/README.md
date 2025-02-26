# Kaira Channels Package

The `kaira.channels` package provides implementations of various communication channel models used in signal processing and wireless communications.

## Available Channels

### Perfect Channel
- `PerfectChannel`: Ideal channel with no distortion or noise (identity function)

### Noise Channels
- `AWGNChannel`: Additive White Gaussian Noise channel for real-valued signals
- `ComplexAWGNChannel`: Additive White Gaussian Noise channel for complex-valued signals

### Fading Channels
- `RayleighChannel`: Rayleigh fading for non-line-of-sight wireless communications
- `RicianChannel`: Rician fading for channels with both line-of-sight and scattered paths
- `FrequencySelectiveChannel`: Models multipath propagation with frequency-dependent fading

### Hardware Impairment Models
- `PhaseNoiseChannel`: Models phase noise from oscillators in RF hardware
- `IQImbalanceChannel`: Models amplitude and phase imbalance between I/Q components
- `NonlinearChannel`: Models nonlinear distortion in power amplifiers and other components

## Usage Example

```python
import torch
from kaira.channels import AWGNChannel, RayleighChannel

# Create a signal
signal = torch.randn(64, 1, 128)  # batch_size=64, channels=1, signal_length=128

# Add noise with 20dB SNR
awgn_channel = AWGNChannel(snr_db=20)
noisy_signal = awgn_channel(signal)

# Apply Rayleigh fading with 15dB SNR
fading_channel = RayleighChannel(snr_db=15)
faded_signal = fading_channel(signal)

# Calculate resulting SNR
from kaira.channels.utils import calculate_snr
resulting_snr = calculate_snr(signal, noisy_signal)
print(f"Resulting SNR: {resulting_snr:.2f} dB")
```

## Utility Functions

The package also provides utility functions: