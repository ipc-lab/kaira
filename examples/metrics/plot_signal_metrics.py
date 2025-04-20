"""
==========================================
Signal and Error Rate Metrics
==========================================

This example demonstrates the usage of signal and error rate metrics
in the Kaira library, including BER (Bit Error Rate), BLER (Block Error Rate),
SER (Symbol Error Rate), FER (Frame Error Rate), and SNR (Signal-to-Noise Ratio).
These metrics are essential for evaluating the performance of communication systems.
"""

from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.metrics.signal import (
    BitErrorRate,
    BlockErrorRate,
    FrameErrorRate,
    SignalToNoiseRatio,
    SymbolErrorRate,
)
from kaira.modulations import QAMDemodulator, QAMModulator
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Initialize Metrics
# -------------------------------------------------------------------------------------------------------------------------------
ber_metric = BitErrorRate()  # or BER()
bler_metric = BlockErrorRate()  # or BLER()
ser_metric = SymbolErrorRate()  # or SER()
fer_metric = FrameErrorRate()  # or FER()
snr_metric = SignalToNoiseRatio()  # or SNR()

# %%
# 1. Basic Metric Usage
# ------------------------------------------------------------------------
# Demonstrate basic usage of each metric

# Generate random bits
n_bits = 1000
bits = torch.randint(0, 2, (1, n_bits))

# Introduce some errors
error_probability = 0.05
errors = torch.rand(1, n_bits) < error_probability
received_bits = torch.logical_xor(bits, errors).int()

# Calculate BER
ber_value = ber_metric(received_bits, bits)

print(f"True error rate: {error_probability}")
print(f"Measured BER: {ber_value.item():.5f}")

# %%
# Visualize bit errors
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.imshow(bits.view(25, 40), cmap="binary", aspect="auto")
plt.title("Original Bits")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(errors.int().view(25, 40), cmap="binary", aspect="auto")
plt.title("Error Locations")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(received_bits.view(25, 40), cmap="binary", aspect="auto")
plt.title("Received Bits")
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
# 2. Block Error Rate (BLER) and Frame Error Rate (FER)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate BLER by reshaping bits into blocks

# Reshape bits into blocks of 10 bits each
block_size = 10
n_blocks = n_bits // block_size
block_bits = bits[:, : n_blocks * block_size].reshape(1, n_blocks, block_size)
received_block_bits = received_bits[:, : n_blocks * block_size].reshape(1, n_blocks, block_size)

# Calculate BLER (a block has an error if any bit in it has an error)
bler_value = bler_metric(received_block_bits, block_bits)

# Calculate FER (treating each block as a frame)
fer_value = fer_metric(received_block_bits, block_bits)

print(f"Block Error Rate: {bler_value.item():.5f}")
print(f"Frame Error Rate: {fer_value.item():.5f}")  # For this example, FER and BLER are the same

# %%
# Visualize the difference between bit errors and block errors
# Actually count the blocks with errors for visualization
blocks_with_errors = torch.any(torch.logical_xor(block_bits, received_block_bits), dim=-1).int()
block_error_rate = blocks_with_errors.float().mean().item()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(["BER", "BLER/FER"], [ber_value.item(), bler_value.item()])
plt.title("Error Rate Comparison")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(blocks_with_errors.view(10, n_blocks // 10), cmap="binary", aspect="auto")
plt.title(f"Blocks with Errors (BLER = {block_error_rate:.3f})")
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
# 3. Symbol Error Rate (SER) with QAM Modulation
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Demonstrate SER using 16-QAM modulation

# Create QAM modulator and demodulator
qam_order: Literal[4, 16, 64, 256] = 16
bits_per_symbol = int(np.log2(qam_order))
modulator = QAMModulator(order=qam_order)
demodulator = QAMDemodulator(order=qam_order)

# Generate random bits for QAM
n_symbols = 1000
n_bits_qam = n_symbols * bits_per_symbol
qam_bits = torch.randint(0, 2, (1, n_bits_qam))

# Modulate
qam_symbols = modulator(qam_bits)

# Add noise (simulating channel effects)
noise_power = 0.05
noise = torch.sqrt(torch.tensor(noise_power)) * torch.randn_like(qam_symbols)
received_symbols = qam_symbols + noise

# Demodulate
received_bits = demodulator(received_symbols)

# Calculate BER
qam_ber = ber_metric(received_bits, qam_bits)

# Calculate SER (reshape bits to calculate symbol errors)
# A symbol has an error if any of its bits are wrong
symbol_bits = qam_bits.reshape(1, n_symbols, bits_per_symbol)
received_symbol_bits = received_bits.reshape(1, n_symbols, bits_per_symbol)
qam_ser = ser_metric(received_symbol_bits, symbol_bits)

print(f"16-QAM BER: {qam_ber.item():.5f}")
print(f"16-QAM SER: {qam_ser.item():.5f}")

# %%
# Visualize QAM constellation and received symbols
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(qam_symbols.real.flatten().numpy(), qam_symbols.imag.flatten().numpy(), label="Transmitted", alpha=0.7, s=30)
plt.scatter(received_symbols.real.flatten().numpy(), received_symbols.imag.flatten().numpy(), label="Received", alpha=0.3, s=10, c="r")
plt.grid(True)
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.title("16-QAM Constellation")
plt.legend()

# Compare BER and SER
plt.subplot(1, 3, 2)
plt.bar(["BER", "SER"], [qam_ber.item(), qam_ser.item()])
plt.title("QAM Error Rate Comparison")
plt.ylim(0, max(qam_ser.item() * 1.2, 0.01))
plt.grid(axis="y", alpha=0.3)

# Visualize error distribution
error_mask = torch.any(torch.logical_xor(symbol_bits, received_symbol_bits), dim=-1).int()
plt.subplot(1, 3, 3)
plt.imshow(error_mask.reshape(25, 40), cmap="binary", aspect="auto")
plt.title("Symbol Errors")
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
# 4. Evaluating Communication System Performance
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate BER and SER over different SNR values

# SNR range in dB
snr_db_range = np.arange(0, 21, 2)
qam_orders: List[Literal[4, 16, 64]] = [4, 16, 64]

# Store results
ber_results: Dict[Literal[4, 16, 64], List[float]] = {order: [] for order in qam_orders}
ser_results: Dict[Literal[4, 16, 64], List[float]] = {order: [] for order in qam_orders}
measured_snr: Dict[Literal[4, 16, 64], List[float]] = {order: [] for order in qam_orders}

# Define parameters
n_symbols = 10000

for qam_order in qam_orders:
    bits_per_symbol = int(np.log2(qam_order))
    modulator = QAMModulator(order=qam_order)
    demodulator = QAMDemodulator(order=qam_order)

    # Generate random bits
    n_bits_qam = n_symbols * bits_per_symbol
    qam_bits = torch.randint(0, 2, (1, n_bits_qam))

    # Modulate
    qam_symbols = modulator(qam_bits)

    for snr_db in snr_db_range:
        # Calculate noise power from SNR
        noise_power = snr_to_noise_power(1.0, snr_db)  # Assuming average signal power of 1.0

        # Create channel
        channel = AWGNChannel(avg_noise_power=noise_power)

        # Transmit through channel
        received_symbols = channel(qam_symbols)

        # Measure SNR
        actual_snr = snr_metric(received_symbols, qam_symbols).item()
        measured_snr[qam_order].append(actual_snr)

        # Demodulate
        received_bits = demodulator(received_symbols)

        # Calculate BER
        ber = ber_metric(received_bits, qam_bits).item()
        ber_results[qam_order].append(ber)

        # Calculate SER
        symbol_bits = qam_bits.reshape(1, n_symbols, bits_per_symbol)
        received_symbol_bits = received_bits.reshape(1, n_symbols, bits_per_symbol)
        ser = ser_metric(received_symbol_bits, symbol_bits).item()
        ser_results[qam_order].append(ser)

# %%
# Plot BER vs SNR
plt.figure(figsize=(15, 5))

# BER vs SNR for different QAM orders
plt.subplot(1, 3, 1)
markers = ["o-", "s-", "^-"]
for order, marker in zip(qam_orders, markers):
    plt.semilogy(snr_db_range, ber_results[order], marker, label=f"{order}-QAM")
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SNR")
plt.legend()

# SER vs SNR for different QAM orders
plt.subplot(1, 3, 2)
for order, marker in zip(qam_orders, markers):
    plt.semilogy(snr_db_range, ser_results[order], marker, label=f"{order}-QAM")
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.title("SER vs SNR")
plt.legend()

# BER vs SER for different QAM orders
plt.subplot(1, 3, 3)
for order, marker in zip(qam_orders, markers):
    plt.loglog(ser_results[order], ber_results[order], marker, label=f"{order}-QAM")
plt.grid(True)
plt.xlabel("Symbol Error Rate (SER)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER vs SER")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# 5. Block Error Rate vs SNR
# ---------------------------------------------------------------------------------------------------
# Analyze how block size affects BLER
block_sizes = [10, 50, 100]
qam_order_bler: Literal[4, 16, 64, 256] = 16
bits_per_symbol = int(np.log2(qam_order_bler))
bler_vs_snr: Dict[int, List[float]] = {block_size: [] for block_size in block_sizes}

# Create modulator/demodulator
modulator = QAMModulator(order=qam_order_bler)
demodulator = QAMDemodulator(order=qam_order_bler)

# Generate random bits (use more bits for larger block sizes)
n_symbols = 10000
n_bits_qam = n_symbols * bits_per_symbol
qam_bits = torch.randint(0, 2, (1, n_bits_qam))

# Modulate
qam_symbols = modulator(qam_bits)

for snr_db in snr_db_range:
    # Calculate noise power from SNR
    noise_power = snr_to_noise_power(1.0, snr_db)

    # Create channel
    channel = AWGNChannel(avg_noise_power=noise_power)

    # Transmit through channel
    received_symbols = channel(qam_symbols)

    # Demodulate
    received_bits = demodulator(received_symbols)

    # Calculate BLER for different block sizes
    for block_size in block_sizes:
        usable_bits = (n_bits_qam // block_size) * block_size
        blocks = qam_bits[:, :usable_bits].reshape(1, -1, block_size)
        received_blocks = received_bits[:, :usable_bits].reshape(1, -1, block_size)

        bler = bler_metric(received_blocks, blocks).item()
        bler_vs_snr[block_size].append(bler)

# %%
# Plot BLER vs SNR for different block sizes
plt.figure(figsize=(10, 6))
for block_size in block_sizes:
    plt.semilogy(snr_db_range, bler_vs_snr[block_size], "o-", label=f"Block Size = {block_size}")
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Block Error Rate (BLER)")
plt.title("BLER vs SNR for Different Block Sizes")
plt.legend()

# Add vertical lines at BLER thresholds
for threshold in [0.1, 0.01, 0.001]:
    plt.axhline(y=threshold, color="r", linestyle="--", alpha=0.3)
    plt.text(0.5, threshold, f"BLER = {threshold}", ha="left", va="bottom", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# 6. Comparing Multiple Metrics on the Same System
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Compare BER, BLER, SER, and FER on a 16-QAM system

# Setup parameters
qam_order_multi: Literal[4, 16, 64, 256] = 16
bits_per_symbol = int(np.log2(qam_order_multi))
block_size = 20  # bits per block
frame_size = 100  # bits per frame

# Create modulator/demodulator
modulator = QAMModulator(order=qam_order_multi)
demodulator = QAMDemodulator(order=qam_order_multi)

# Generate random bits
n_symbols = 10000
n_bits = n_symbols * bits_per_symbol
bits = torch.randint(0, 2, (1, n_bits))

# Modulate
symbols = modulator(bits)

# Store results
metrics: Dict[str, List[float]] = {"BER": [], "SER": [], "BLER": [], "FER": []}

for snr_db in snr_db_range:
    # Calculate noise power from SNR
    noise_power = snr_to_noise_power(1.0, snr_db)

    # Create channel
    channel = AWGNChannel(avg_noise_power=noise_power)

    # Transmit through channel
    received_symbols = channel(symbols)

    # Demodulate
    received_bits = demodulator(received_symbols)

    # Calculate BER
    ber = ber_metric(received_bits, bits).item()
    metrics["BER"].append(ber)

    # Calculate SER
    # Reshape bits to calculate symbol errors
    usable_bits_ser = (n_bits // bits_per_symbol) * bits_per_symbol
    symbol_bits = bits[:, :usable_bits_ser].reshape(1, -1, bits_per_symbol)
    received_symbol_bits = received_bits[:, :usable_bits_ser].reshape(1, -1, bits_per_symbol)
    ser = ser_metric(received_symbol_bits, symbol_bits).item()
    metrics["SER"].append(ser)

    # Calculate BLER
    usable_bits_bler = (n_bits // block_size) * block_size
    blocks = bits[:, :usable_bits_bler].reshape(1, -1, block_size)
    received_blocks = received_bits[:, :usable_bits_bler].reshape(1, -1, block_size)
    bler = bler_metric(received_blocks, blocks).item()
    metrics["BLER"].append(bler)

    # Calculate FER
    usable_bits_fer = (n_bits // frame_size) * frame_size
    frames = bits[:, :usable_bits_fer].reshape(1, -1, frame_size)
    received_frames = received_bits[:, :usable_bits_fer].reshape(1, -1, frame_size)
    fer = fer_metric(received_frames, frames).item()
    metrics["FER"].append(fer)

# %%
# Plot all metrics together
plt.figure(figsize=(10, 6))

markers = ["o-", "s-", "^-", "x-"]
colors = ["b", "r", "g", "m"]
line_styles = ["-", "--", "-.", ":"]

for metric_name, style, color, marker in zip(metrics.keys(), line_styles, colors, markers):
    plt.semilogy(snr_db_range, metrics[metric_name], marker[0] + style, label=metric_name, color=color)

plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Error Rate")
plt.title("Error Rate Metrics vs SNR for 16-QAM")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Conclusion
# --------------------------------------------------------------
# This example demonstrated:
#
# 1. Implementation and usage of various error rate metrics in Kaira
# 2. The relationship between different error metrics (BER, SER, BLER, FER)
# 3. How modulation order affects error rates
# 4. The impact of block size on BLER
# 5. Performance evaluation of communication systems using multiple metrics
#
# Key observations:
#
# - Higher-order QAM schemes are more susceptible to noise, requiring higher SNR
# - SER is always equal to or higher than BER
# - Larger block sizes increase the probability of block errors at a given SNR
# - For coding and system design, different metrics are relevant at different stages
# - BLER and FER are critical for evaluating the performance of coded systems
