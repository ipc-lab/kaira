"""
===========================================
Simulating AWGN Channels with Kaira
===========================================

This example demonstrates the usage of Additive White Gaussian Noise (AWGN) channel
in the Kaira library. AWGN is one of the most common communication channel models,
which adds Gaussian noise to the input signal.

We'll visualize how different noise levels (SNR) affect signal transmission.
"""

# %%
# Imports and Setup
# -------------------------------
# We start by importing the necessary modules and setting up the environment.

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.metrics.image import PSNR
from kaira.metrics.signal import SNR
from kaira.utils import snr_to_noise_power
from kaira.utils.plotting import PlottingUtils

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
PlottingUtils.setup_plotting_style()

# %%
# Create Sample Signal
# ------------------------------------
# Let's create a sample signal to pass through our AWGN channels.
# We'll use a sine wave as our input signal.

# Generate time points
t = np.linspace(0, 1, 1000)
# Create a sine wave
frequency = 5  # Hz
amplitude = 1.0
signal = amplitude * np.sin(2 * np.pi * frequency * t)

# Convert to torch tensor
input_signal = torch.from_numpy(signal).float().reshape(1, -1)  # Add batch dimension
print(f"Input signal shape: {input_signal.shape}")

# %%
# Create AWGN Channels with Different SNR Levels
# --------------------------------------------------------------------------------------
# We'll create multiple AWGN channels with different Signal-to-Noise Ratio (SNR) levels
# to observe how noise affects our signal.

# Define SNR levels in dB
snr_levels_db = [20, 10, 5, 0, -5]

# Create AWGN channels
awgn_channels = []
for snr_db in snr_levels_db:
    # Convert SNR from dB to linear scale and calculate noise power
    signal_power = amplitude**2 / 2  # Average power of sine wave is A²/2
    noise_power = snr_to_noise_power(signal_power, snr_db)

    # Create AWGN channel with the calculated noise power
    channel = AWGNChannel(avg_noise_power=noise_power)
    awgn_channels.append((snr_db, channel))
    print(f"Created AWGN channel with SNR: {snr_db} dB (noise power: {noise_power:.6f})")

# %%
# Pass Signal Through AWGN Channels
# -------------------------------------------------------------
# Now we'll pass our input signal through each AWGN channel and collect the outputs.

# Initialize Kaira SNR and PSNR metrics
snr_metric = SNR()
psnr_metric = PSNR(dim=1)  # Specify dimension 1 (signal dimension) for calculation

# Process signal through each channel
outputs = []
measured_metrics = []

for snr_db, channel in awgn_channels:
    with torch.no_grad():
        # Pass signal through the channel
        output_signal = channel(input_signal)

    # Measure actual SNR and PSNR using Kaira metrics
    measured_snr = snr_metric(output_signal, input_signal).item()
    measured_psnr = psnr_metric(output_signal, input_signal).item()

    # Store the results
    outputs.append((snr_db, output_signal.numpy().flatten()))
    measured_metrics.append({"target_snr_db": snr_db, "measured_snr_db": measured_snr, "measured_psnr_db": measured_psnr})

    # Ensure we're using float values for string formatting
    # AWGN Channel Performance Analysis
    # ===============================
    # Target SNR: {snr_db:.1f} dB, Measured SNR: {measured_snr:.1f} dB, PSNR: {measured_psnr:.1f} dB

# %%
# Visualize the Results
# -------------------------------------
# Let's visualize how different SNR levels affect the transmitted signal.

# Visualization: Signal Degradation with Noise
# ============================================
# Compare the original clean signal with signals processed through
# AWGN channels at different SNR levels to observe noise effects.

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("AWGN Channel Effects on Signal Transmission", fontsize=16, fontweight="bold")

# Plot original and noisy signals
ax1 = axes[0, 0]
ax1.plot(t, signal, "b-", linewidth=2, label="Original Signal", alpha=0.8)
for i, (snr_db, output) in enumerate(outputs[:3]):  # Show first 3 for clarity
    color = PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)]
    ax1.plot(t, output, "--", color=color, linewidth=1.5, alpha=0.7, label=f"SNR: {snr_db} dB")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.set_title("Signal Comparison")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot SNR comparison
ax2 = axes[0, 1]
target_snrs = [metric["target_snr_db"] for metric in measured_metrics]
measured_snrs = [metric["measured_snr_db"] for metric in measured_metrics]
ax2.plot(target_snrs, measured_snrs, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8)
ax2.plot(target_snrs, target_snrs, "--", color="gray", alpha=0.7, label="Ideal (Target = Measured)")
ax2.set_xlabel("Target SNR (dB)")
ax2.set_ylabel("Measured SNR (dB)")
ax2.set_title("SNR Validation")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot PSNR values
ax3 = axes[1, 0]
psnr_values = [metric["measured_psnr_db"] for metric in measured_metrics]
ax3.plot(target_snrs, psnr_values, "s-", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, markersize=8)
ax3.set_xlabel("Target SNR (dB)")
ax3.set_ylabel("PSNR (dB)")
ax3.set_title("PSNR vs Target SNR")
ax3.grid(True, alpha=0.3)

# Plot noise effects on signal
ax4 = axes[1, 1]
for i, (snr_db, output) in enumerate(outputs):
    noise = output - signal
    noise_power = np.mean(noise**2)
    ax4.bar(i, noise_power, color=PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)], alpha=0.7)
ax4.set_xlabel("Channel Index")
ax4.set_ylabel("Noise Power")
ax4.set_title("Noise Power by Channel")
ax4.set_xticks(range(len(outputs)))
ax4.set_xticklabels([f"{snr}dB" for snr, _ in outputs], rotation=45)
ax4.grid(True, alpha=0.3)

fig.show()

# %%
# Compare Theoretical and Measured SNR Values
# ------------------------------------------------------------------------------
# Let's compare the target SNR values with what we actually measured.

# SNR and PSNR Analysis
# ====================
# Compare theoretical vs measured SNR values and examine PSNR behavior
# to validate the AWGN channel model performance.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig.suptitle("SNR and PSNR Validation", fontsize=16, fontweight="bold")

# SNR comparison
target_snrs = [metric["target_snr_db"] for metric in measured_metrics]
measured_snrs = [metric["measured_snr_db"] for metric in measured_metrics]
psnr_values = [metric["measured_psnr_db"] for metric in measured_metrics]

ax1.scatter(target_snrs, measured_snrs, color=PlottingUtils.MODERN_PALETTE[0], s=100, alpha=0.7, label="Measured SNR")
ax1.plot(target_snrs, target_snrs, "--", color="gray", alpha=0.7, label="Ideal (Target = Measured)")
ax1.set_xlabel("Target SNR (dB)")
ax1.set_ylabel("Measured SNR (dB)")
ax1.set_title("SNR Validation")
ax1.legend()
ax1.grid(True, alpha=0.3)

# PSNR vs Target SNR
ax2.plot(target_snrs, psnr_values, "o-", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, markersize=8)
ax2.set_xlabel("Target SNR (dB)")
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("PSNR vs Target SNR")
ax2.grid(True, alpha=0.3)

fig.show()

# %%
# Calculate Mean Squared Error (MSE)
# --------------------------------------------------------------
# Let's calculate the MSE between the original and the noisy signals.

mse_values = []
for snr_db, output in outputs:
    mse = np.mean((signal - output) ** 2)
    mse_values.append((snr_db, mse))
    # MSE Analysis Results
    # ===================
    # SNR: {snr_db} dB, MSE: {mse:.6f}

# %%
# Plot SNR vs MSE
# -------------------------
# Let's plot the relationship between SNR and MSE.

# MSE vs SNR Relationship Analysis
# ===============================
# Examine the theoretical and measured relationship between
# signal-to-noise ratio and mean squared error.

snr_levels = [snr for snr, _ in mse_values]
mse_vals = [mse for _, mse in mse_values]
signal_power = amplitude**2 / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig.suptitle("SNR vs Mean Squared Error Analysis", fontsize=16, fontweight="bold")

# Plot MSE vs SNR
ax1.semilogy(snr_levels, mse_vals, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8)
ax1.set_xlabel("SNR (dB)")
ax1.set_ylabel("MSE")
ax1.set_title("Measured MSE vs SNR")
ax1.grid(True, alpha=0.3)

# Plot theoretical MSE (noise power)
theoretical_mse = []
for snr_db in snr_levels:
    snr_linear = 10 ** (snr_db / 10)
    theoretical_noise_power = signal_power / snr_linear
    theoretical_mse.append(theoretical_noise_power)

ax2.semilogy(snr_levels, mse_vals, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8, label="Measured MSE")
ax2.semilogy(snr_levels, theoretical_mse, "--", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, label="Theoretical MSE")
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("MSE")
ax2.set_title("Measured vs Theoretical MSE")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.show()

# %%
# Conclusion
# ------------------
# This example demonstrated how to use the AWGNChannel in Kaira to simulate noisy
# signal transmission with different SNR levels. We also used Kaira's metrics to evaluate
# the actual SNR and PSNR of the transmitted signals.
#
# Key observations:
# - Lower SNR values result in higher noise levels and greater signal distortion
# - The AWGNChannel accurately models the specified SNR levels
# - The PSNR is higher than SNR, as expected for this type of signal
# - MSE follows the theoretical inverse relationship with SNR
#
# The AWGNChannel is widely used in communication system modeling because AWGN
# is a good first approximation for many real-world noisy channels.
