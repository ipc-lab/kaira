"""Utilities for benchmarking and analysis of modulation schemes."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable, Type
import time
from pathlib import Path
from kaira.core import BaseModulator, BaseDemodulator

from .utils import calculate_theoretical_ber
import pandas as pd


def awgn_channel(x: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, float]:
    """Apply Additive White Gaussian Noise (AWGN) channel to signal.
    
    Args:
        x: Input complex symbols
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Tuple of (noisy symbols, noise variance)
    """
    # Calculate signal power
    signal_power = torch.mean(torch.abs(x)**2).item()
    
    # Convert SNR from dB to linear scale
    snr_linear = 10**(snr_db / 10.0)
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # Division by 2 for complex noise
    
    # Generate complex Gaussian noise
    noise_real = torch.randn_like(x.real) * noise_std
    noise_imag = torch.randn_like(x.imag) * noise_std
    noise = torch.complex(noise_real, noise_imag)
    
    # Add noise to signal
    y = x + noise
    
    return y, noise_power


def measure_ber(
    modulator: Modulator, 
    demodulator: Demodulator, 
    snr_db: Union[float, List[float]], 
    n_bits: int = 100000,
    batch_size: int = 10000,
    device: Optional[torch.device] = None
) -> Dict[float, float]:
    """Measure Bit Error Rate (BER) for a given modulation scheme over a range of SNR values.
    
    Args:
        modulator: Modulation module
        demodulator: Demodulation module
        snr_db: Signal-to-noise ratio in dB (can be a single value or list)
        n_bits: Total number of bits to simulate
        batch_size: Number of bits per batch for computation efficiency
        device: Device to run simulation on (defaults to CPU)
        
    Returns:
        Dictionary mapping SNR values to measured BER
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    modulator = modulator.to(device)
    demodulator = demodulator.to(device)
    
    # Convert single SNR to list
    if isinstance(snr_db, (int, float)):
        snr_db = [snr_db]
    
    # Make SNR list a tensor for easier iteration
    snr_values = torch.tensor(snr_db, dtype=torch.float32)
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_bits / batch_size))
    actual_bits = n_batches * batch_size
    
    # Dictionary to store results
    ber_results = {}
    
    # For each SNR value
    for snr in snr_values:
        total_errors = 0
        
        # Reset any stateful components
        if hasattr(modulator, 'reset_state'):
            modulator.reset_state()
        if hasattr(demodulator, 'reset_state'):
            demodulator.reset_state()
        
        # Process data in batches
        for _ in range(n_batches):
            # Generate random bits
            bits = torch.randint(0, 2, (batch_size,), dtype=torch.float32, device=device)
            
            # Modulate
            symbols = modulator(bits)
            
            # Pass through AWGN channel
            noisy_symbols, noise_var = awgn_channel(symbols, snr.item())
            
            # Demodulate (hard decision)
            decoded_bits = demodulator(noisy_symbols)
            
            # Count bit errors - handle potential dimension differences
            if bits.shape != decoded_bits.shape:
                # For differential schemes, we lose the first symbol
                min_len = min(bits.shape[0], decoded_bits.shape[0])
                bit_errors = torch.sum(bits[:min_len] != decoded_bits[:min_len]).item()
                total_errors += bit_errors
            else:
                bit_errors = torch.sum(bits != decoded_bits).item()
                total_errors += bit_errors
        
        # Calculate BER
        ber = total_errors / actual_bits
        ber_results[snr.item()] = ber
    
    return ber_results


def plot_ber_curve(
    ber_results: Dict[float, float],
    modulation_name: str,
    theoretical: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Figure:
    """Plot Bit Error Rate (BER) curve.
    
    Args:
        ber_results: Dictionary mapping SNR values to measured BER
        modulation_name: Name of modulation scheme for labeling
        theoretical: Whether to include theoretical BER curve for comparison
        ax: Optional existing axis to plot on
        **kwargs: Additional arguments to pass to plot function
        
    Returns:
        Matplotlib figure object
    """
    # Sort SNR values
    snr_values = sorted(ber_results.keys())
    ber_values = [ber_results[snr] for snr in snr_values]
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        
    # Plot simulated BER
    ax.semilogy(snr_values, ber_values, 'o-', label=f'{modulation_name} Simulated', **kwargs)
    
    # Add theoretical curve if requested
    if theoretical:
        try:
            theoretical_ber = calculate_theoretical_ber(snr_values, modulation_name.lower())
            ax.semilogy(snr_values, theoretical_ber, '--', label=f'{modulation_name} Theoretical')
        except ValueError:
            # Theoretical curve not available for this modulation
            pass
    
    # Add labels and grid
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('Bit Error Rate vs SNR')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Set reasonable y-axis limits
    ax.set_ylim(bottom=1e-6, top=1.0)
    
    return fig


def compare_modulation_schemes(
    modulators: Dict[str, Tuple[Modulator, Demodulator]],
    snr_range: List[float],
    n_bits: int = 100000,
    batch_size: int = 10000,
    theoretical: bool = True,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare BER performance of multiple modulation schemes.
    
    Args:
        modulators: Dictionary mapping scheme names to (modulator, demodulator) pairs
        snr_range: List of SNR values to test
        n_bits: Number of bits to simulate per SNR point
        batch_size: Batch size for computation efficiency
        theoretical: Whether to include theoretical curves
        device: Device to run on (defaults to CUDA if available, else CPU)
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure with BER comparison
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+']
    
    for i, (name, (mod, demod)) in enumerate(modulators.items()):
        print(f"Testing {name}...")
        start_time = time.time()
        
        # Measure BER
        ber_results = measure_ber(
            modulator=mod,
            demodulator=demod,
            snr_db=snr_range,
            n_bits=n_bits,
            batch_size=batch_size,
            device=device
        )
        
        elapsed = time.time() - start_time
        print(f"Completed {name} in {elapsed:.2f} seconds")
        
        # Plot with unique color and marker
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plot_ber_curve(
            ber_results,
            name,
            theoretical=theoretical,
            ax=ax,
            color=color,
            marker=marker
        )
    
    # Enhanced formatting
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=1e-6, top=1.0)
    ax.set_xlim(left=min(snr_range), right=max(snr_range))
    ax.set_title('Bit Error Rate Comparison of Modulation Schemes', fontsize=14)
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.legend(fontsize=10)
    
    # Add text with simulation parameters
    param_text = f"Simulation: {n_bits} bits per SNR point"
    ax.annotate(param_text, xy=(0.02, 0.02), xycoords='figure fraction', fontsize=8)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def measure_throughput(
    modulator: Modulator,
    demodulator: Optional[Demodulator] = None,
    n_bits: int = 1000000,
    batch_size: int = 10000,
    n_runs: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Measure throughput (processing speed) of modulation/demodulation.
    
    Args:
        modulator: Modulation module to benchmark
        demodulator: Optional demodulation module to benchmark
        n_bits: Number of bits to process
        batch_size: Batch size for processing
        n_runs: Number of repeated runs for averaging
        device: Device to run benchmarks on
        
    Returns:
        Dictionary with throughput metrics in Mbps
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    modulator = modulator.to(device)
    if demodulator is not None:
        demodulator = demodulator.to(device)
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_bits / batch_size))
    actual_bits = n_batches * batch_size
    
    # Generate random bits once to exclude data generation from timing
    all_bits = torch.randint(0, 2, (actual_bits,), dtype=torch.float32, device=device)
    
    results = {}
    
    # Time modulation
    mod_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for i in range(n_batches):
            batch = all_bits[i * batch_size:(i + 1) * batch_size]
            symbols = modulator(batch)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
        elapsed = time.time() - start_time
        mod_times.append(elapsed)
    
    avg_mod_time = sum(mod_times) / n_runs
    mod_throughput = actual_bits / avg_mod_time / 1e6  # Mbps
    results['modulation_throughput_mbps'] = mod_throughput
    
    # Time demodulation if provided
    if demodulator is not None:
        # First, generate symbols
        all_symbols = []
        for i in range(0, actual_bits, batch_size):
            batch = all_bits[i:i+batch_size]
            symbols = modulator(batch)
            all_symbols.append(symbols)
        
        # Now time demodulation
        demod_times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            for symbols_batch in all_symbols:
                _ = demodulator(symbols_batch)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                
            elapsed = time.time() - start_time
            demod_times.append(elapsed)
        
        avg_demod_time = sum(demod_times) / n_runs
        demod_throughput = actual_bits / avg_demod_time / 1e6  # Mbps
        results['demodulation_throughput_mbps'] = demod_throughput
    
    return results


def benchmark_modulation_schemes(
    modulators: Dict[str, Tuple[Modulator, Demodulator]],
    n_bits: int = 1000000,
    batch_size: int = 10000,
    device: Optional[torch.device] = None
) -> 'pd.DataFrame':
    """Benchmark modulation schemes for processing speed.
    
    Args:
        modulators: Dictionary mapping scheme names to (modulator, demodulator) pairs
        n_bits: Number of bits to process
        batch_size: Batch size for processing
        device: Device to run benchmarks on
        
    Returns:
        DataFrame with benchmark results
    """
    import pandas as pd
    
    results = []
    
    for name, (mod, demod) in modulators.items():
        print(f"Benchmarking {name}...")
        
        # Measure throughput
        throughput = measure_throughput(
            modulator=mod,
            demodulator=demod,
            n_bits=n_bits,
            batch_size=batch_size,
            device=device
        )
        
        # Add to results
        results.append({
            'Modulation': name,
            'Bits per Symbol': mod.bits_per_symbol,
            'Modulation Throughput (Mbps)': throughput['modulation_throughput_mbps'],
            'Demodulation Throughput (Mbps)': throughput.get('demodulation_throughput_mbps', float('nan'))
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add spectral efficiency
    df['Spectral Efficiency (bps/Hz)'] = df['Bits per Symbol']
    
    return df