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

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    print(f"Target SNR: {snr_db:.1f} dB, Measured SNR: {measured_snr:.1f} dB, PSNR: {measured_psnr:.1f} dB")

# %%
# Visualize the Results
# -------------------------------------
# Let's visualize how different SNR levels affect the transmitted signal.

plt.figure(figsize=(10, 8))

# Plot the original signal
plt.subplot(len(snr_levels_db) + 1, 1, 1)
plt.plot(t, signal, "b-", linewidth=1.5)
plt.title("Original Signal")
plt.grid(True)
plt.ylabel("Amplitude")
plt.xlim([0, 1])

# Plot each noisy signal
for i, (snr_db, output) in enumerate(outputs):
    plt.subplot(len(snr_levels_db) + 1, 1, i + 2)
    plt.plot(t, output, "r-", alpha=0.8)
    measured_snr = measured_metrics[i]["measured_snr_db"]
    plt.title(f"AWGN Channel (Target SNR = {snr_db} dB, Measured SNR = {measured_snr:.1f} dB)")
    plt.grid(True)
    plt.ylabel("Amplitude")
    if i == len(outputs) - 1:
        plt.xlabel("Time (s)")
    plt.xlim([0, 1])

plt.tight_layout()
plt.show()

# %%
# Compare Theoretical and Measured SNR Values
# ------------------------------------------------------------------------------
# Let's compare the target SNR values with what we actually measured.

plt.figure(figsize=(10, 5))

target_snrs = [metric["target_snr_db"] for metric in measured_metrics]
measured_snrs = [metric["measured_snr_db"] for metric in measured_metrics]
measured_psnrs = [metric["measured_psnr_db"] for metric in measured_metrics]

# Plot SNR comparison
plt.subplot(1, 2, 1)
plt.plot(target_snrs, measured_snrs, "bo-", linewidth=2, label="Measured SNR")
plt.plot(target_snrs, target_snrs, "k--", linewidth=1, label="Theoretical (Target)")
plt.grid(True)
plt.xlabel("Target SNR (dB)")
plt.ylabel("Measured SNR (dB)")
plt.title("Theoretical vs. Measured SNR")
plt.legend()

# Plot PSNR values
plt.subplot(1, 2, 2)
plt.plot(target_snrs, measured_psnrs, "ro-", linewidth=2)
plt.grid(True)
plt.xlabel("Target SNR (dB)")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs. Target SNR")

plt.tight_layout()
plt.show()

# %%
# Calculate Mean Squared Error (MSE)
# --------------------------------------------------------------
# Let's calculate the MSE between the original and the noisy signals.

mse_values = []
for snr_db, output in outputs:
    mse = np.mean((signal - output) ** 2)
    mse_values.append((snr_db, mse))
    print(f"SNR: {snr_db} dB, MSE: {mse:.6f}")

# %%
# Plot SNR vs MSE
# -------------------------
# Let's plot the relationship between SNR and MSE.

plt.figure(figsize=(8, 5))
snr_levels = [snr for snr, _ in mse_values]
mse_vals = [mse for _, mse in mse_values]

plt.plot(snr_levels, mse_vals, "o-", linewidth=2)
plt.grid(True)
plt.xlabel("SNR (dB)")
plt.ylabel("Mean Squared Error")
plt.title("SNR vs. Mean Squared Error")
plt.yscale("log")  # Use logarithmic scale for MSE

# Add theoretical MSE curve: MSE = noise_power = signal_power / 10^(SNR/10)
snr_range = np.linspace(-6, 21, 100)
signal_power = amplitude**2 / 2
theoretical_mse = signal_power / np.power(10, snr_range / 10)
plt.plot(snr_range, theoretical_mse, "k--", linewidth=1, label="Theoretical")
plt.legend()

plt.show()

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
