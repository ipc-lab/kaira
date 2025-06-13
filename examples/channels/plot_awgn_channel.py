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

import numpy as np
import torch

from examples.utils.plotting import (
    setup_plotting_style,
    plot_signal_noise_comparison,
    plot_snr_psnr_comparison,
    plot_snr_vs_mse,
    plot_noise_level_analysis
)

from kaira.channels import AWGNChannel
from kaira.metrics.image import PSNR
from kaira.metrics.signal import SNR
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
setup_plotting_style()

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

fig = plot_signal_noise_comparison(t, signal, outputs, measured_metrics, 
                                 "AWGN Channel Effects on Signal Transmission")
fig.show()

# %%
# Compare Theoretical and Measured SNR Values
# ------------------------------------------------------------------------------
# Let's compare the target SNR values with what we actually measured.

# SNR and PSNR Analysis
# ====================
# Compare theoretical vs measured SNR values and examine PSNR behavior
# to validate the AWGN channel model performance.

fig = plot_snr_psnr_comparison(measured_metrics, "SNR and PSNR Validation")
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

fig = plot_snr_vs_mse(snr_levels, mse_vals, signal_power, "SNR vs Mean Squared Error Analysis")
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
