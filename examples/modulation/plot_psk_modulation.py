"""
=========================================
Phase-Shift Keying (PSK) Modulation
=========================================

This example demonstrates the usage of Phase-Shift Keying (PSK) modulation schemes
in the Kaira library, specifically focusing on BPSK and QPSK modulation.
We'll visualize constellation diagrams and analyze bit error rates.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# --------------------------------
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.metrics.signal import BER
from kaira.modulations import BPSKDemodulator, BPSKModulator, QPSKDemodulator, QPSKModulator
from kaira.modulations.utils import plot_constellation
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Generate Random Binary Data
# -------------------------------------------------
n_symbols = 1000
bits_bpsk = torch.randint(0, 2, (1, n_symbols))
bits_qpsk = torch.randint(0, 2, (1, 2 * n_symbols))  # QPSK uses 2 bits per symbol

print(f"Number of BPSK bits: {bits_bpsk.numel()}")
print(f"Number of QPSK bits: {bits_qpsk.numel()}")

# %%
# Create Modulators and Demodulators
# -----------------------------------------------------------------
bpsk_mod = BPSKModulator()
bpsk_demod = BPSKDemodulator()
qpsk_mod = QPSKModulator()
qpsk_demod = QPSKDemodulator()

# Modulate the data
bpsk_symbols = bpsk_mod(bits_bpsk)
qpsk_symbols = qpsk_mod(bits_qpsk)

# %%
# Plot Constellation Diagrams
# -------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# BPSK constellation
plot_constellation(bpsk_symbols.flatten(), title="BPSK Constellation", marker="o", ax=axs[0])
axs[0].grid(True)

# QPSK constellation
plot_constellation(qpsk_symbols.flatten(), title="QPSK Constellation", marker="o", ax=axs[1])
axs[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# Simulate Transmission over AWGN Channel
# ---------------------------------------------------------------------
# We'll transmit the modulated symbols through an AWGN channel at different SNR levels

snr_db_range = np.arange(0, 21, 2)
ber_bpsk = []
ber_qpsk = []

# Initialize BER metric
ber_metric = BER()

for snr_db in snr_db_range:
    # Calculate noise power and create AWGN channel
    noise_power = snr_to_noise_power(1.0, snr_db)  # Assuming unit signal power
    channel = AWGNChannel(avg_noise_power=noise_power)

    # BPSK transmission
    received_bpsk = channel(bpsk_symbols)
    demod_bits_bpsk = bpsk_demod(received_bpsk)
    ber_bpsk.append(ber_metric(demod_bits_bpsk, bits_bpsk).item())

    # QPSK transmission
    received_qpsk = channel(qpsk_symbols)
    demod_bits_qpsk = qpsk_demod(received_qpsk)
    ber_qpsk.append(ber_metric(demod_bits_qpsk, bits_qpsk).item())

# %%
# Plot BER vs SNR
# ----------------------
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_bpsk, "bo-", label="BPSK")
plt.semilogy(snr_db_range, ber_qpsk, "ro-", label="QPSK")

# Add theoretical BER curves
snr_lin = 10 ** (snr_db_range / 10)
theoretical_ber_bpsk = 0.5 * torch.erfc(torch.sqrt(torch.tensor(snr_lin)))
theoretical_ber_qpsk = torch.erfc(torch.sqrt(torch.tensor(snr_lin))) / 2

plt.semilogy(snr_db_range, theoretical_ber_bpsk, "b--", alpha=0.5, label="BPSK (Theoretical)")
plt.semilogy(snr_db_range, theoretical_ber_qpsk, "r--", alpha=0.5, label="QPSK (Theoretical)")

plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Performance of BPSK and QPSK")
plt.legend()
plt.show()

# %%
# Visualize Effect of Noise
# ---------------------------------------
# Let's visualize how noise affects the constellation diagrams
test_snr_db = [20, 10, 5]
n_test_symbols = 1000

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, snr_db in enumerate(test_snr_db):
    noise_power = snr_to_noise_power(1.0, snr_db)
    channel = AWGNChannel(avg_noise_power=noise_power)

    # Generate and modulate random QPSK symbols
    test_bits = torch.randint(0, 2, (1, 2 * n_test_symbols))
    qpsk_symbols = qpsk_mod(test_bits)
    received_symbols = channel(qpsk_symbols)

    plot_constellation(received_symbols.flatten(), title=f"QPSK at {snr_db} dB SNR", marker=".", ax=axs[i])
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. Implementation of BPSK and QPSK modulation using Kaira
# 2. Visualization of constellation diagrams
# 3. Effect of AWGN on modulated signals
# 4. BER performance analysis
#
# Key observations:
#
# - BPSK achieves better BER performance than QPSK at the same SNR
# - QPSK provides twice the spectral efficiency of BPSK
# - Constellation points become more scattered as SNR decreases
# - Practical results closely match theoretical predictions
# Test comment
