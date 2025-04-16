"""
=========================================
π/4-QPSK Modulation
=========================================
This example demonstrates π/4-QPSK (pi/4 shifted QPSK) modulation in Kaira.
π/4-QPSK is a variant of QPSK where the constellation is rotated by π/4 radians
on alternating symbols, providing improved envelope properties and phase transitions.

This modulation scheme is used in several digital mobile communications systems
including North American TDMA (IS-136) and Japanese Digital Cellular.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# --------------------------------
import numpy as np
import torch

from kaira.channels import AWGNChannel, FlatFadingChannel
from kaira.metrics import BER
from kaira.modulations import Pi4QPSKDemodulator, Pi4QPSKModulator, QPSKDemodulator, QPSKModulator
from kaira.modulations.utils import plot_constellation
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Generate Random Binary Data
# -------------------------------------------------
n_symbols = 1000
bits = torch.randint(0, 2, (1, 2 * n_symbols))  # Each symbol uses 2 bits
print(f"Number of bits: {bits.numel()}")

# %%
# Create π/4-QPSK Modulator and Demodulator
# -----------------------------------------------------------------
pi4qpsk_mod = Pi4QPSKModulator()
pi4qpsk_demod = Pi4QPSKDemodulator()

# Regular QPSK for comparison
qpsk_mod = QPSKModulator()
qpsk_demod = QPSKDemodulator()

# Modulate the data
pi4qpsk_symbols = pi4qpsk_mod(bits)
qpsk_symbols = qpsk_mod(bits)

# %%
# Visualize π/4-QPSK Constellation
# -------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# π/4-QPSK constellation
plot_constellation(pi4qpsk_symbols.flatten(), title="π/4-QPSK Constellation", marker="o", ax=axs[0])
axs[0].grid(True)

# Regular QPSK constellation for comparison
plot_constellation(qpsk_symbols.flatten(), title="Regular QPSK Constellation", marker="o", ax=axs[1])
axs[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# Visualize Phase Transitions
# -----------------------------------------------------------------
# Generate a simple bit sequence to visualize transitions
simple_bits = torch.tensor([[0, 0, 1, 1, 0, 1, 1, 0, 0, 1]])
# Reset modulator state
pi4qpsk_mod.reset_state()
qpsk_mod.reset_state()

# Modulate bits
pi4qpsk_simple = pi4qpsk_mod(simple_bits)
qpsk_simple = qpsk_mod(simple_bits)

# Plot transitions
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# π/4-QPSK transitions
axs[0].plot(pi4qpsk_simple[0].real, pi4qpsk_simple[0].imag, "bo-", linewidth=2)
axs[0].plot(pi4qpsk_simple[0, 0].real, pi4qpsk_simple[0, 0].imag, "ro", markersize=10, label="First symbol")
axs[0].grid(True)
axs[0].set_aspect("equal")
axs[0].set_title("π/4-QPSK Phase Transitions")
axs[0].set_xlabel("In-phase")
axs[0].set_ylabel("Quadrature")
axs[0].legend()

# QPSK transitions
axs[1].plot(qpsk_simple[0].real, qpsk_simple[0].imag, "bo-", linewidth=2)
axs[1].plot(qpsk_simple[0, 0].real, qpsk_simple[0, 0].imag, "ro", markersize=10, label="First symbol")
axs[1].grid(True)
axs[1].set_aspect("equal")
axs[1].set_title("Regular QPSK Phase Transitions")
axs[1].set_xlabel("In-phase")
axs[1].set_ylabel("Quadrature")
axs[1].legend()

plt.tight_layout()
plt.show()

# %%
# Performance in AWGN Channel
# ---------------------------------------------------------------------
# Compare π/4-QPSK with regular QPSK in AWGN
snr_db_range = np.arange(0, 21, 2)
ber_pi4qpsk = []
ber_qpsk = []

# Initialize BER metric
ber_metric = BER()

# Reset modulator states
pi4qpsk_mod.reset_state()
pi4qpsk_demod.reset_state()

for snr_db in snr_db_range:
    # Calculate noise power and create AWGN channel
    noise_power = snr_to_noise_power(1.0, snr_db)
    channel = AWGNChannel(avg_noise_power=noise_power)

    # π/4-QPSK transmission
    received_pi4qpsk = channel(pi4qpsk_symbols)
    demod_bits_pi4qpsk = pi4qpsk_demod(received_pi4qpsk)
    ber_pi4qpsk.append(ber_metric(demod_bits_pi4qpsk, bits).item())

    # QPSK transmission
    received_qpsk = channel(qpsk_symbols)
    demod_bits_qpsk = qpsk_demod(received_qpsk)
    ber_qpsk.append(ber_metric(demod_bits_qpsk, bits).item())

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_pi4qpsk, "bo-", label="π/4-QPSK")
plt.semilogy(snr_db_range, ber_qpsk, "ro-", label="QPSK")

# Theoretical QPSK BER
snr_lin = 10 ** (snr_db_range / 10)
theoretical_ber_qpsk = torch.erfc(torch.sqrt(torch.tensor(snr_lin))) / 2
plt.semilogy(snr_db_range, theoretical_ber_qpsk, "k--", alpha=0.5, label="Theoretical QPSK")

plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Performance in AWGN Channel")
plt.legend()
plt.show()

# %%
# Performance in Fading Channels
# ---------------------------------------------------------------------
# Compare π/4-QPSK with regular QPSK in Rayleigh fading
ber_pi4qpsk_fading = []
ber_qpsk_fading = []

# Reset modulator states
pi4qpsk_mod.reset_state()
pi4qpsk_demod.reset_state()

for snr_db in snr_db_range:
    # Calculate noise power and create Rayleigh fading channel
    noise_power = snr_to_noise_power(1.0, snr_db)
    channel = FlatFadingChannel(fading_type="rayleigh", coherence_time=n_symbols // 10, snr_db=snr_db)  # Introducing some temporal correlation

    # π/4-QPSK transmission
    received_pi4qpsk = channel(pi4qpsk_symbols)
    demod_bits_pi4qpsk = pi4qpsk_demod(received_pi4qpsk)
    ber_pi4qpsk_fading.append(ber_metric(demod_bits_pi4qpsk, bits).item())

    # QPSK transmission
    received_qpsk = channel(qpsk_symbols)
    demod_bits_qpsk = qpsk_demod(received_qpsk)
    ber_qpsk_fading.append(ber_metric(demod_bits_qpsk, bits).item())

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db_range, ber_pi4qpsk_fading, "bo-", label="π/4-QPSK")
plt.semilogy(snr_db_range, ber_qpsk_fading, "ro-", label="QPSK")
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Performance in Rayleigh Fading Channel")
plt.legend()
plt.show()

# %%
# Envelope Characteristics
# ---------------------------------------------------------------------
# Generate a longer sequence to demonstrate envelope characteristics
long_bits = torch.randint(0, 2, (1, 2 * 200))

# Reset modulator states
pi4qpsk_mod.reset_state()
qpsk_mod.reset_state()

# Modulate bits
pi4qpsk_long = pi4qpsk_mod(long_bits)
qpsk_long = qpsk_mod(long_bits)

# Calculate envelopes (magnitude)
pi4qpsk_envelope = torch.abs(pi4qpsk_long)
qpsk_envelope = torch.abs(qpsk_long)

# Plot envelope variations
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(pi4qpsk_envelope[0].numpy(), "b-", linewidth=1.5)
axs[0].grid(True)
axs[0].set_title("π/4-QPSK Envelope")
axs[0].set_xlabel("Symbol index")
axs[0].set_ylabel("Envelope magnitude")
axs[0].set_ylim(0, 1.5)

axs[1].plot(qpsk_envelope[0].numpy(), "r-", linewidth=1.5)
axs[1].grid(True)
axs[1].set_title("QPSK Envelope")
axs[1].set_xlabel("Symbol index")
axs[1].set_ylabel("Envelope magnitude")
axs[1].set_ylim(0, 1.5)

plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. Implementation of π/4-QPSK modulation using Kaira
# 2. Comparison with conventional QPSK
# 3. Visualization of constellation diagrams and phase transitions
# 4. Performance analysis in AWGN and Rayleigh fading channels
# 5. Envelope characteristics analysis
#
# Key observations:
#
# - π/4-QPSK has the same BER performance as QPSK in AWGN
# - The π/4 phase rotation creates smoother transitions between symbols
# - π/4-QPSK avoids transitions through the origin, improving envelope properties
# - This results in lower peak-to-average power ratio, beneficial for RF amplifiers
# - The constellation alternates between two sets rotated by 45° from each other
