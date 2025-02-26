"""Testing Utilities for Channel Performance Evaluation."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.core import BaseChannel

from .utils import calculate_snr, evaluate_ber


def measure_snr_vs_param(
    channel_factory: Callable[[float], BaseChannel],
    param_values: List[float],
    param_name: str,
    input_signal: torch.Tensor,
    num_trials: int = 5,
) -> Tuple[List[float], List[float], List[float]]:
    """Measure SNR vs a swept parameter value.

    Args:
        channel_factory: Function that creates a channel given a parameter value
        param_values: List of parameter values to test
        param_name: Name of the parameter being tested
        input_signal: Input signal tensor
        num_trials: Number of trials for each parameter value

    Returns:
        Tuple containing parameter values, mean SNRs, and std deviation of SNRs
    """
    results = []

    for param in param_values:
        # Create channel with current parameter value
        channel = channel_factory(param)

        # Run multiple trials
        trial_snrs = []
        for _ in range(num_trials):
            with torch.no_grad():
                output_signal = channel(input_signal)
                snr = calculate_snr(input_signal, output_signal).item()
                trial_snrs.append(snr)

        # Record mean and std deviation
        results.append((param, np.mean(trial_snrs), np.std(trial_snrs)))

    # Unzip results
    params, means, stds = zip(*results)

    return list(params), list(means), list(stds)


def plot_snr_vs_param(
    param_values: List[float],
    snr_means: List[float],
    snr_stds: List[float],
    param_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot SNR vs parameter value with error bars.

    Args:
        param_values: List of parameter values
        snr_means: List of mean SNR values
        snr_stds: List of standard deviations of SNR values
        param_name: Name of the parameter (for labeling)
        ax: Optional matplotlib axes to plot on

    Returns:
        The matplotlib axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        param_values, snr_means, yerr=snr_stds, fmt="-o", capsize=5, markersize=8, linewidth=2
    )

    ax.set_xlabel(param_name)
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"SNR vs {param_name}")
    ax.grid(True, alpha=0.3)

    return ax


def evaluate_channel_ber(
    channel: BaseChannel,
    modulator: Callable[[torch.Tensor], torch.Tensor],
    demodulator: Callable[[torch.Tensor], torch.Tensor],
    snr_values: List[float],
    num_bits: int = 100000,
    batch_size: int = 10000,
) -> Dict[float, float]:
    """Evaluate Bit Error Rate performance of a channel across SNR values.

    Args:
        channel: Channel model to test
        modulator: Function that maps bits to symbols
        demodulator: Function that maps symbols back to bits
        snr_values: List of SNR values (in dB) to test
        num_bits: Total number of bits to test
        batch_size: Number of bits per batch

    Returns:
        Dictionary mapping SNR values to measured BER
    """
    results = {}

    for snr_db in snr_values:
        # Configure channel for this SNR
        channel.snr_db = snr_db

        total_errors = 0
        total_bits = 0

        for _ in range(num_bits // batch_size):
            # Generate random bits
            bits = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

            # Modulate bits to symbols
            symbols = modulator(bits)

            # Pass through channel
            received_symbols = channel(symbols)

            # Demodulate
            received_bits = demodulator(received_symbols)

            # Count errors
            errors = (bits != received_bits).sum().item()

            total_errors += errors
            total_bits += batch_size

        # Calculate BER
        ber = total_errors / total_bits
        results[snr_db] = ber

    return results


def plot_ber_vs_snr(
    ber_results: Dict[float, float],
    theoretical_func: Optional[Callable[[float], float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot BER vs SNR with optional theoretical curve.

    Args:
        ber_results: Dictionary mapping SNR values to BER
        theoretical_func: Optional function for theoretical BER curve
        ax: Optional matplotlib axes to plot on

    Returns:
        The matplotlib axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by SNR for proper line plotting
    snr_values = sorted(ber_results.keys())
    ber_values = [ber_results[snr] for snr in snr_values]

    # Plot simulated BER
    ax.semilogy(snr_values, ber_values, "-o", label="Simulated")

    # Add theoretical curve if provided
    if theoretical_func:
        theo_ber = [theoretical_func(snr) for snr in snr_values]
        ax.semilogy(snr_values, theo_ber, "--", label="Theoretical")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs SNR Performance")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    return ax
