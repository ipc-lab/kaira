"""Communication Channel Models for Signal Processing.

This module provides various channel models used in communication systems and signal
processing applications. Each channel model simulates different types of signal
distortion and noise that occur in real-world communication systems.

Available Channels:
    - AWGNChannel: Additive White Gaussian Noise channel
    - ComplexAWGNChannel: Complex-valued AWGN channel
    - PerfectChannel: Ideal channel with no distortion
    - RayleighChannel: Rayleigh fading channel for wireless communications
    - RicianChannel: Rician fading channel with line-of-sight component
    - FrequencySelectiveChannel: Channel with frequency-selective fading
    - PhaseNoiseChannel: Channel that introduces phase noise
    - IQImbalanceChannel: Channel modeling I/Q imbalance in hardware
    - NonlinearChannel: Channel with polynomial nonlinearity
    - RappModel: Rapp model for power amplifier nonlinearity

Each channel implements a forward() method that takes an input tensor and returns
the output tensor after applying the channel effects.

Example:
    >>> from kaira.channels import AWGNChannel
    >>> channel = AWGNChannel(avg_noise_power=0.1)
    >>> input_signal = torch.randn(32, 1, 16, 16)
    >>> noisy_signal = channel(input_signal)
"""

from .awgn import AWGNChannel, ComplexAWGNChannel
from .fading import FrequencySelectiveChannel, RayleighChannel, RicianChannel
from .impairments import IQImbalanceChannel, PhaseNoiseChannel
from .nonlinear import NonlinearChannel, RappModel
from .perfect import PerfectChannel
from .testing import (
    evaluate_channel_ber,
    measure_snr_vs_param,
    plot_ber_vs_snr,
    plot_snr_vs_param,
)
from .utils import calculate_snr, evaluate_ber, noise_power_to_snr, snr_to_noise_power
from .visualization import (
    plot_channel_response,
    plot_constellation,
    plot_impulse_response,
)

__all__ = [
    # Channel models
    "PerfectChannel",
    "AWGNChannel",
    "ComplexAWGNChannel",
    "RayleighChannel",
    "RicianChannel",
    "FrequencySelectiveChannel",
    "PhaseNoiseChannel",
    "IQImbalanceChannel",
    "NonlinearChannel",
    "RappModel",
    # Utility functions
    "snr_to_noise_power",
    "noise_power_to_snr",
    "calculate_snr",
    "evaluate_ber",
    # Visualization utilities
    "plot_channel_response",
    "plot_constellation",
    "plot_impulse_response",
    # Testing utilities
    "measure_snr_vs_param",
    "plot_snr_vs_param",
    "evaluate_channel_ber",
    "plot_ber_vs_snr",
]
