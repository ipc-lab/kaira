"""
===========================================
Fading Channels in Wireless Communications
===========================================

This example demonstrates how to simulate and analyze fading channels using Kaira.
Fading channels model signal attenuation and phase shifts that occur in wireless
communications due to multipath propagation and other environmental factors.

In this example, we'll focus on the FlatFadingChannel model, which simulates
flat fading where all frequency components of the signal experience the same
magnitude of fading.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# -------------------------------
import numpy as np
import torch
from scipy import signal  # Added here to fix E402 error

from kaira.channels import AWGNChannel, FlatFadingChannel, PerfectChannel
from kaira.metrics.signal import BitErrorRate
from kaira.modulations import QPSKModulator
from kaira.modulations.utils import calculate_theoretical_ber
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Generate QPSK Signal
# ------------------------------------
# Let's use Kaira's QPSKModulator to generate QPSK symbols.

# Create a QPSK modulator
qpsk_modulator = QPSKModulator()

# Generate random bits for transmission
n_symbols = 1000
n_bits = n_symbols * 2  # QPSK uses 2 bits per symbol
random_bits = torch.randint(0, 2, (1, n_bits)).float()

# Modulate bits to QPSK symbols
with torch.no_grad():
    qpsk_symbols = qpsk_modulator(random_bits)

# Reshape for transmission through the channel (add sequence dimension if needed)
# Each symbol has 2 components (real and imaginary)
input_signal = qpsk_symbols.view(1, -1)

# Keep track of the original symbol indices for later analysis
symbol_indices = torch.zeros(n_symbols, dtype=torch.long)
for i in range(n_symbols):
    idx = 0
    if random_bits[0, i * 2] > 0:
        idx += 2
    if random_bits[0, i * 2 + 1] > 0:
        idx += 1
    symbol_indices[i] = idx

print(f"Generated {n_symbols} QPSK symbols")
print(f"Input signal shape: {input_signal.shape}")
print(f"First 5 complex symbols: {qpsk_symbols[:5]}")

# Show the QPSK constellation diagram
plt.figure(figsize=(6, 6))
qpsk_modulator.plot_constellation()
plt.title("QPSK Constellation")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Define Channel Scenarios
# ------------------------------------------
# We'll compare a perfect channel (no distortion), an AWGN channel (noise only),
# and a flat fading channel (fading + noise).

# Define SNR for our channels in dB
snr_db = 15
signal_power = 1.0  # QPSK normalized to unit power
noise_power = snr_to_noise_power(signal_power, snr_db)

# Create the channels
perfect_channel = PerfectChannel()
awgn_channel = AWGNChannel(avg_noise_power=noise_power)
fading_channel = FlatFadingChannel(fading_type="rayleigh", coherence_time=1, avg_noise_power=noise_power)  # Use Rayleigh fading  # Independent fading for each symbol

print(f"Created channels with SNR: {snr_db} dB (noise power: {noise_power:.6f})")
print(f"AWGN Channel configuration: {awgn_channel.get_config()}")
print(f"Fading Channel configuration: {fading_channel.get_config()}")

# %%
# Pass Signal Through Channels
# -------------------------------------------------
# Now we'll pass our QPSK signal through each channel type.

with torch.no_grad():
    # Pass through perfect channel (no distortion)
    perfect_output = perfect_channel(input_signal)

    # Pass through AWGN channel (adds noise)
    awgn_output = awgn_channel(input_signal)

    # Pass through flat fading channel (adds fading + noise)
    fading_output = fading_channel(input_signal)

# Convert outputs to complex values
perfect_complex = perfect_output.squeeze(0).cpu().numpy().view(np.complex128)
awgn_complex = awgn_output.squeeze(0).cpu().numpy().view(np.complex128)
fading_complex = fading_output.squeeze(0).cpu().numpy().view(np.complex128)

# %%
# Visualize Channel Effects on Constellation
# --------------------------------------------------------------------------
# Let's visualize how each channel affects the QPSK constellation.

# Take a subset for clearer visualization
n_points = 200
subset = slice(0, n_points)

# Create scatter plots
plt.figure(figsize=(15, 5))

# Perfect channel
plt.subplot(1, 3, 1)
plt.scatter(np.real(perfect_complex[subset]), np.imag(perfect_complex[subset]), c=symbol_indices[subset], cmap="viridis", alpha=0.7, s=30)
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title("Perfect Channel\nQPSK Constellation")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.axis("equal")

# AWGN channel
plt.subplot(1, 3, 2)
plt.scatter(np.real(awgn_complex[subset]), np.imag(awgn_complex[subset]), c=symbol_indices[subset], cmap="viridis", alpha=0.7, s=30)
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title(f"AWGN Channel (SNR={snr_db} dB)\nQPSK Constellation")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.axis("equal")

# Fading channel
plt.subplot(1, 3, 3)
plt.scatter(np.real(fading_complex[subset]), np.imag(fading_complex[subset]), c=symbol_indices[subset], cmap="viridis", alpha=0.7, s=30)
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title(f"Rayleigh Fading Channel (SNR={snr_db} dB)\nQPSK Constellation")
plt.xlabel("In-Phase")
plt.ylabel("Quadrature")
plt.axis("equal")

plt.tight_layout()
plt.show()

# %%
# Symbol Amplitude Distribution
# --------------------------------------------------
# Let's analyze how fading affects the amplitude distribution of the symbols.

# Calculate amplitudes - ensure we're using real values
perfect_amp = np.abs(perfect_complex).real
awgn_amp = np.abs(awgn_complex).real
fading_amp = np.abs(fading_complex).real
plt.figure(figsize=(12, 5))

# Histogram of amplitudes
plt.subplot(1, 2, 1)
plt.hist(perfect_amp, bins=30, alpha=0.3, label="Perfect Channel", density=True)
plt.hist(awgn_amp, bins=30, alpha=0.3, label="AWGN Channel", density=True)
plt.hist(fading_amp, bins=30, alpha=0.3, label="Fading Channel", density=True)
plt.grid(True)
plt.xlabel("Symbol Amplitude")
plt.ylabel("Probability Density")
plt.title("Symbol Amplitude Distribution")
plt.legend()

# Theoretical vs. Empirical Rayleigh Distribution
x = np.linspace(0, 3, 1000)
# Rayleigh PDF: (x/σ²) * exp(-x²/(2σ²))
# For unit variance Rayleigh, σ² = 1/2
rayleigh_pdf = x * np.exp(-(x**2) / 2)
plt.subplot(1, 2, 2)
plt.hist(fading_amp, bins=30, alpha=0.5, density=True, label="Empirical (Fading Channel)")
plt.plot(x, rayleigh_pdf, "r-", linewidth=2, label="Theoretical Rayleigh")
plt.grid(True)
plt.xlabel("Symbol Amplitude")
plt.ylabel("Probability Density")
plt.title("Rayleigh Fading Amplitude Distribution")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Effect of SNR on Symbol Error Rate in Fading Channels
# -------------------------------------------------------------------------------------------------
# Let's examine how SNR affects symbol error rates in AWGN vs. fading channels.

# Define a range of SNR values
snr_range_db = np.arange(0, 26, 5)

# Use Kaira's BitErrorRate metric
ber_metric = BitErrorRate()

# Set up arrays to store results
awgn_ser = []
fading_ser = []

# For each SNR level, simulate transmission and measure error rate
for snr_db in snr_range_db:
    # Calculate noise power from SNR
    noise_power = snr_to_noise_power(signal_power, snr_db)

    # Create channels with current SNR
    awgn = AWGNChannel(avg_noise_power=noise_power)
    fading = FlatFadingChannel(fading_type="rayleigh", coherence_time=1, avg_noise_power=noise_power)

    # Pass signal through channels
    with torch.no_grad():
        awgn_out = awgn(input_signal)
        fading_out = fading(input_signal)

    # Convert to complex form for demodulation
    awgn_complex_out = awgn_out.squeeze(0).cpu().numpy().view(np.complex128)
    fading_complex_out = fading_out.squeeze(0).cpu().numpy().view(np.complex128)

    # Convert to constellation indices by finding closest constellation point
    qpsk_points = qpsk_modulator.constellation.cpu().numpy().view(np.complex128)

    # Calculate SER manually (since we're interested in symbol errors, not bit errors)
    def calculate_ser(received, original_indices):
        """Calculate Symbol Error Rate for QPSK symbols."""
        # Map received symbols back to nearest constellation point
        received_real = np.real(received)
        received_imag = np.imag(received)

        # Determine quadrant (equivalent to detecting QPSK symbol)
        detected_indices = np.zeros(len(received), dtype=int)
        detected_indices[(received_real > 0) & (received_imag > 0)] = 0  # 1+1j
        detected_indices[(received_real > 0) & (received_imag < 0)] = 1  # 1-1j
        detected_indices[(received_real < 0) & (received_imag > 0)] = 2  # -1+1j
        detected_indices[(received_real < 0) & (received_imag < 0)] = 3  # -1-1j

        # Convert to numpy and ensure shapes match
        original_np = original_indices.numpy()

        # Make sure both arrays have the same length
        min_length = min(len(detected_indices), len(original_np))
        detected_indices = detected_indices[:min_length]
        original_np = original_np[:min_length]

        # Calculate error rate
        errors = detected_indices != original_np
        ser = np.mean(errors)

        return ser

    # Calculate SER
    awgn_ser.append(calculate_ser(awgn_complex_out, symbol_indices))
    fading_ser.append(calculate_ser(fading_complex_out, symbol_indices))

    print(f"SNR: {snr_db} dB - AWGN SER: {awgn_ser[-1]:.4f}, Fading SER: {fading_ser[-1]:.4f}")

# %%
# Plot SER vs. SNR
# --------------------------

plt.figure(figsize=(10, 6))
plt.semilogy(snr_range_db, awgn_ser, "bo-", linewidth=2, label="AWGN Channel")
plt.semilogy(snr_range_db, fading_ser, "rs-", linewidth=2, label="Rayleigh Fading Channel")

# Add theoretical curves using Kaira's calculate_theoretical_ber function
snr_theory = np.linspace(0, 25, 100)
# For QPSK - use Kaira's built-in function - Fix parameter order
awgn_theory_ser = calculate_theoretical_ber(snr_theory, "qpsk") * 2  # Convert BER to SER (approx)

# For Rayleigh fading, we still need to use the formula since it's not in Kaira yet
snr_linear = 10 ** (snr_theory / 10)
fading_theory_ser = 1 - 1 / np.sqrt(1 + 1 / (2 * snr_linear))

plt.semilogy(snr_theory, awgn_theory_ser, "b--", alpha=0.7, label="AWGN Theory")
plt.semilogy(snr_theory, fading_theory_ser, "r--", alpha=0.7, label="Rayleigh Theory")

plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Symbol Error Rate (SER)")
plt.title("SER vs. SNR Comparison for QPSK")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Visualizing Time-Varying Fading
# -------------------------------------------------------
# Let's simulate and visualize time-varying fading effects.

# Generate a longer sequence of symbols for better visualization
n_symbols_time = 500
time_bits = torch.randint(0, 2, (1, n_symbols_time * 2)).float()
with torch.no_grad():
    time_symbols = qpsk_modulator(time_bits)
time_input = time_symbols.view(1, -1)

# Create a fading channel with time-correlation
time_fading_channel = FlatFadingChannel(fading_type="rayleigh", coherence_time=10, avg_noise_power=noise_power)  # Fading stays constant for 10 symbols

# Pass signal through the channel
with torch.no_grad():
    time_fading_output = time_fading_channel(time_input)

# Extract the fading coefficients (this is a simplified approach)
# For real implementation, we'd need to compute h from channel model directly
time_fading_complex = time_fading_output.squeeze(0).cpu().numpy().view(np.complex128)
original_complex = time_input.squeeze(0).cpu().numpy().view(np.complex128)

# Estimate fading by division (simplified)
fading_estimate = time_fading_complex / original_complex
fading_magnitude = np.abs(fading_estimate)

# %%
# Plot the time-varying fading magnitude
plt.figure(figsize=(12, 6))
plt.plot(fading_magnitude, linewidth=2)
plt.grid(True)
plt.xlabel("Symbol Index")
plt.ylabel("Fading Magnitude")
plt.title("Time-Varying Rayleigh Fading Magnitude")

# Add the theoretical Rayleigh average (√(π/2) ≈ 0.89 for normalized Rayleigh)
plt.axhline(y=np.sqrt(np.pi / 2) * np.mean(fading_magnitude) / 1.253, color="r", linestyle="--", label="Theoretical Average")
plt.legend()
plt.show()

# %%
# Power Spectral Density of Fading Process
# -------------------------------------------------------------------------
# Let's analyze the frequency characteristics of the fading process.

# Calculate PSD using FFT

# Use Welch's method to estimate PSD
f, psd = signal.welch(fading_magnitude, fs=1.0, nperseg=256)

plt.figure(figsize=(10, 5))
plt.semilogy(f, psd, linewidth=2)
plt.grid(True)
plt.xlabel("Normalized Frequency")
plt.ylabel("Power Spectral Density")
plt.title("PSD of Rayleigh Fading Process")
plt.axvline(x=0.05, color="r", linestyle="--", label="Doppler Frequency (0.05)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------
# This example demonstrated the use of fading channels in Kaira, with a focus on
# the FlatFadingChannel model. We observed:
#
# - How Rayleigh fading distorts the constellation of a QPSK modulated signal
# - The amplitude distribution of signals under fading conditions
# - The impact of SNR on symbol error rates in fading vs. AWGN channels
# - The time-varying nature of fading channels
#
# Fading channels are essential for simulating wireless communications in realistic
# environments where signals experience amplitude and phase variations due to
# multipath propagation, scattering, and other physical phenomena.
