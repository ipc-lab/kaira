"""Example usage of the Kaira modulations library."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.modulations import (
    ConstellationVisualizer,
    compare_modulation_schemes,
    create_modem,
    measure_ber,
)


def basic_modulation_example():
    """Basic modulation and demodulation example."""
    print("Basic modulation and demodulation example:")
    
    # Create modulator and demodulator
    modulator, demodulator = create_modem("qpsk")
    
    # Generate some random bits
    n_bits = 1000
    bits = torch.randint(0, 2, (n_bits,), dtype=torch.float)
    print(f"Original bits shape: {bits.shape}")
    
    # Modulate
    symbols = modulator(bits)
    print(f"Modulated symbols shape: {symbols.shape}")
    print(f"Bits per symbol: {modulator.bits_per_symbol}")
    
    # Add noise
    snr_db = 10  # 10 dB SNR
    signal_power = torch.mean(torch.abs(symbols) ** 2).item()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)
    
    noise = torch.complex(
        torch.randn_like(symbols.real) * noise_std,
        torch.randn_like(symbols.imag) * noise_std
    )
    noisy_symbols = symbols + noise
    
    # Hard decision demodulation
    received_bits_hard = demodulator(noisy_symbols)
    bit_errors_hard = torch.sum(torch.abs(bits - received_bits_hard)).item()
    ber_hard = bit_errors_hard / n_bits
    print(f"Hard decision BER @ {snr_db} dB: {ber_hard:.6f}")
    
    # Soft decision demodulation
    llrs = demodulator(noisy_symbols, noise_power)
    received_bits_soft = (llrs < 0).float()
    bit_errors_soft = torch.sum(torch.abs(bits - received_bits_soft)).item()
    ber_soft = bit_errors_soft / n_bits
    print(f"Soft decision BER @ {snr_db} dB: {ber_soft:.6f}")
    
    print()


def visualization_example():
    """Example of constellation visualization."""
    print("Constellation visualization example:")
    
    # Create different modulators
    bpsk_mod, _ = create_modem("bpsk")
    qpsk_mod, _ = create_modem("qpsk")
    qam16_mod, _ = create_modem("qam", order=16)
    
    # Visualize constellations
    visualizer = ConstellationVisualizer(qam16_mod)
    fig = visualizer.plot_basic()
    plt.savefig("16qam_constellation.png")
    print("Saved 16-QAM constellation plot to 16qam_constellation.png")
    
    # Visualize with noise
    fig_noise = visualizer.plot_with_noise(snr_db=15)
    plt.savefig("16qam_constellation_noisy.png")
    print("Saved noisy 16-QAM constellation plot to 16qam_constellation_noisy.png")
    
    # Visualize with phase noise
    fig_phase = visualizer.plot_with_phase_noise(phase_std=0.1)
    plt.savefig("16qam_constellation_phase_noise.png")
    print("Saved phase-noisy 16-QAM constellation plot to 16qam_constellation_phase_noise.png")
    
    print()


def ber_analysis_example():
    """Example of BER analysis."""
    print("BER analysis example:")
    
    # Create modulator-demodulator pairs
    bpsk_mod, bpsk_demod = create_modem("bpsk")
    qpsk_mod, qpsk_demod = create_modem("qpsk")
    qam16_mod, qam16_demod = create_modem("qam", order=16)
    
    # Measure BER for BPSK
    snr_values = [0, 4, 8, 12]
    print(f"Measuring BER for BPSK at SNR values: {snr_values} dB")
    snr_db, ber = measure_ber(bpsk_mod, bpsk_demod, n_bits=10000, snr_db=snr_values)
    for snr, ber_val in zip(snr_db, ber):
        print(f"  SNR = {snr} dB: BER = {ber_val:.6f}")
    
    # Compare multiple schemes
    print("\nComparing modulation schemes...")
    schemes = ["bpsk", "qpsk", "16qam"]
    snr_range = np.arange(0, 21, 4)
    results, fig = compare_modulation_schemes(
        schemes, snr_range, n_bits=10000, 
        title="BER Comparison of Modulation Schemes"
    )
    plt.savefig("ber_comparison.png")
    print("Saved BER comparison plot to ber_comparison.png")
    
    print()


def batch_processing_example():
    """Example of batch processing with modulations."""
    print("Batch processing example:")
    
    # Create modulator and demodulator
    qam_mod, qam_demod = create_modem("qam", order=16)
    
    # Generate batched data
    batch_size = 5
    n_bits_per_example = 1024
    bits = torch.randint(0, 2, (batch_size, n_bits_per_example), dtype=torch.float)
    print(f"Batch shape: {bits.shape}")
    
    # Modulate (supports batching)
    symbols = qam_mod(bits)
    print(f"Modulated symbols batch shape: {symbols.shape}")
    
    # Add noise
    noise = torch.complex(
        torch.randn_like(symbols.real) * 0.05,
        torch.randn_like(symbols.imag) * 0.05
    )
    noisy_symbols = symbols + noise
    
    # Demodulate (supports batching)
    received_bits = qam_demod(noisy_symbols)
    print(f"Demodulated bits batch shape: {received_bits.shape}")
    
    # Check BER for each example in batch
    for i in range(batch_size):
        errors = torch.sum(torch.abs(bits[i] - received_bits[i])).item()
        ber = errors / n_bits_per_example
        print(f"  Example {i+1}: BER = {ber:.6f}")
    
    print()


def gpu_example():
    """Example of using GPU if available."""
    print("GPU acceleration example:")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available! Using CUDA.")
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
    
    # Create modulator and demodulator
    modulator, demodulator = create_modem("qpsk")
    
    # Move to device
    modulator = modulator.to(device)
    demodulator = demodulator.to(device)
    
    # Generate data on device
    n_bits = 10000
    bits = torch.randint(0, 2, (n_bits,), dtype=torch.float, device=device)
    
    # Process on device
    start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    
    if device.type == "cuda":
        start_time.record()
    symbols = modulator(bits)
    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"Time to modulate {n_bits} bits: {elapsed_time:.3f} ms")
    
    # Add noise
    noise = torch.complex(
        torch.randn_like(symbols.real, device=device) * 0.1,
        torch.randn_like(symbols.imag, device=device) * 0.1
    )
    noisy_symbols = symbols + noise
    
    if device.type == "cuda":
        start_time.record()
    received_bits = demodulator(noisy_symbols)
    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"Time to demodulate {n_bits} bits: {elapsed_time:.3f} ms")
    
    # Check results
    bit_errors = torch.sum(torch.abs(bits - received_bits)).item()
    ber = bit_errors / n_bits
    print(f"Bit error rate: {ber:.6f}")


if __name__ == "__main__":
    print("Kaira Modulations Library Demo\n")
    
    basic_modulation_example()
    visualization_example()
    ber_analysis_example()
    batch_processing_example()
    gpu_example()
    
    print("\nDemo completed!")
