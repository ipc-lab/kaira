"""
===========================================
Digital Binary Channels in Kaira
===========================================

This example demonstrates the usage of binary channel models in Kaira.
Binary channels are fundamental in digital communications as they represent
the transmission of binary data (0s and 1s) through a noisy medium.

We'll explore the three main binary channel models:
1. Binary Symmetric Channel (BSC)
2. Binary Erasure Channel (BEC)
3. Binary Z-Channel
"""

# %%
# Imports and Setup
# -------------------------------
# We start by importing the necessary modules and setting up the environment.

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.channels import BinaryErasureChannel, BinarySymmetricChannel, BinaryZChannel
from kaira.utils.plotting import PlottingUtils

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
PlottingUtils.setup_plotting_style()

# %%
# Generate Binary Data
# ------------------------------------
# Let's generate a random binary sequence to transmit through our channels.

# Generate 1000 random binary values (0 or 1)
num_bits = 1000
binary_data = torch.randint(0, 2, (1, num_bits)).float()

# Binary Data Generation Results
# =============================
# Generated {num_bits} random bits
# First 20 bits: {binary_data[0, :20].int().tolist()}

# %%
# Binary Symmetric Channel (BSC)
# -------------------------------------------------------
# The BSC flips bits with probability p. Both 0→1 and 1→0 transitions occur with the same probability.

# Create BSC with different error probabilities
error_probs = [0.01, 0.05, 0.1, 0.2, 0.5]
bsc_outputs = []

for p in error_probs:
    # Create the channel
    bsc = BinarySymmetricChannel(crossover_prob=p)

    # Pass data through the channel
    with torch.no_grad():
        output = bsc(binary_data)

    # Calculate bit error rate
    errors = (output != binary_data).sum().item()
    error_rate = errors / num_bits

    bsc_outputs.append((p, output, error_rate))
    # BSC Channel Analysis Results
    # ===========================
    # BSC (p={p}): Errors: {errors}/{num_bits}, Error rate: {error_rate:.4f}

# %%
# Binary Erasure Channel (BEC)
# --------------------------------------------------
# The BEC erases bits with probability p, replacing them with a special "erasure" symbol (here represented by -1).

# Create BEC with different erasure probabilities
erasure_probs = [0.01, 0.05, 0.1, 0.2, 0.5]
bec_outputs = []

for p in erasure_probs:
    # Create the channel
    bec = BinaryErasureChannel(erasure_prob=p)

    # Pass data through the channel
    with torch.no_grad():
        output = bec(binary_data)

    # Count erasures
    erasures = (output == -1).sum().item()
    erasure_rate = erasures / num_bits

    bec_outputs.append((p, output, erasure_rate))
    print(f"BEC (p={p}): Erasures: {erasures}/{num_bits}, Erasure rate: {erasure_rate:.4f}")

# %%
# Binary Z-Channel
# --------------------------
# The Z-Channel has asymmetric error probabilities. Only 1→0 transitions occur with probability p.

# Create Z-Channel with different error probabilities
z_error_probs = [0.01, 0.05, 0.1, 0.2, 0.5]
z_outputs = []

for p in z_error_probs:
    # Create the channel
    z_channel = BinaryZChannel(error_prob=p)

    # Pass data through the channel
    with torch.no_grad():
        output = z_channel(binary_data)

    # Calculate errors (only 1→0 flips can occur)
    original_ones = binary_data == 1
    errors = ((output != binary_data) & original_ones).sum().item()
    ones_count = original_ones.sum().item()
    error_rate = errors / ones_count if ones_count > 0 else 0

    z_outputs.append((p, output, error_rate))
    print(f"Z-Channel (p={p}): 1→0 Errors: {errors}/{ones_count}, Error rate: {error_rate:.4f}")

# %%
# Visualizing Channel Effects
# -------------------------------------------------
# Let's visualize a small segment of the data to see how each channel affects the binary transmission.

# Take a small segment of the data for visualization
segment_start = 0
segment_length = 50

# Binary Channel Effects Visualization
# ====================================
# Compare how different binary channels affect the same data segment
# to understand the distinct characteristics of each channel type.

# Create channel outputs for visualization
channel_outputs = []

# BSC output (high error probability for visibility)
bsc_p = 0.2
bsc = BinarySymmetricChannel(crossover_prob=bsc_p)
with torch.no_grad():
    bsc_output = bsc(binary_data).numpy()[0]
channel_outputs.append(("BSC", bsc_output, bsc_p))

# BEC output (high erasure probability for visibility)
bec_p = 0.2
bec = BinaryErasureChannel(erasure_prob=bec_p)
with torch.no_grad():
    bec_output = bec(binary_data).numpy()[0]
channel_outputs.append(("BEC", bec_output, bec_p))

# Z-Channel output (high error probability for visibility)
z_p = 0.5  # Higher for visibility since it only affects 1→0 transitions
z_channel = BinaryZChannel(error_prob=z_p)
with torch.no_grad():
    z_output = z_channel(binary_data).numpy()[0]
channel_outputs.append(("Z-Channel", z_output, z_p))

# Visualize channel effects
original_data = binary_data[0].numpy()

# Create binary channel comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Binary Channel Effects Comparison", fontsize=16, fontweight="bold")

# Plot original data segment
segment_end = segment_start + segment_length
ax1 = axes[0, 0]
ax1.plot(range(segment_length), original_data[segment_start:segment_end], "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=6, label="Original")
ax1.set_title("Original Binary Data")
ax1.set_xlabel("Bit Index")
ax1.set_ylabel("Bit Value")
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3)

# Plot each channel output
for i, (channel_name, output, error_prob) in enumerate(channel_outputs[:3]):
    ax = axes.flat[i + 1]
    ax.plot(range(segment_length), output[segment_start:segment_end], "o-", color=PlottingUtils.MODERN_PALETTE[(i + 1) % len(PlottingUtils.MODERN_PALETTE)], linewidth=2, markersize=6, label=f"{channel_name} (p={error_prob:.2f})")
    ax.set_title(f"{channel_name} Output")
    ax.set_xlabel("Bit Index")
    ax.set_ylabel("Bit Value")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.show()

# %%
# Comparing Error Rates Across Channels
# -------------------------------------------------------------------
# Now let's compare the theoretical vs. observed error rates for each channel type.

# Channel Error Rate Analysis
# ===========================
# Compare theoretical vs observed error rates across different
# channel types to validate the implementation accuracy.

# Prepare BSC error rate data
theoretical_bsc = error_probs  # Theoretical error rate equals p
observed_bsc = [err_rate for _, _, err_rate in bsc_outputs]

# Plot BSC error rates
fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax1.plot(error_probs, theoretical_bsc, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8, label="Theoretical BSC")
ax1.plot(error_probs, observed_bsc, "s-", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, markersize=8, label="Observed BSC")
ax1.set_xlabel("Error Probability")
ax1.set_ylabel("Error Rate")
ax1.set_title("Binary Symmetric Channel Error Rates", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.show()

# Prepare BEC erasure rate data
theoretical_bec = erasure_probs  # Theoretical erasure rate equals p
observed_bec = [erasure_rate for _, _, erasure_rate in bec_outputs]

# Plot BEC erasure rates
fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax2.plot(erasure_probs, theoretical_bec, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8, label="Theoretical BEC")
ax2.plot(erasure_probs, observed_bec, "s-", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, markersize=8, label="Observed BEC")
ax2.set_xlabel("Erasure Probability")
ax2.set_ylabel("Erasure Rate")
ax2.set_title("Binary Erasure Channel Erasure Rates", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.show()

# Prepare Z-Channel error rate data
# Theoretical error rate for Z-channel is p * P(1), where P(1) is probability of input being 1
p_one = (binary_data == 1).sum().item() / num_bits
theoretical_z = [p * p_one for p in z_error_probs]
observed_z = [err_rate * p_one for _, _, err_rate in z_outputs]

# Plot Z-Channel error rates
fig3, ax3 = plt.subplots(figsize=(10, 6), constrained_layout=True)
ax3.plot(z_error_probs, theoretical_z, "o-", color=PlottingUtils.MODERN_PALETTE[0], linewidth=2, markersize=8, label="Theoretical Z-Channel")
ax3.plot(z_error_probs, observed_z, "s-", color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, markersize=8, label="Observed Z-Channel")
ax3.set_xlabel("Error Probability")
ax3.set_ylabel("Error Rate")
ax3.set_title("Z-Channel Error Rates", fontsize=14, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.show()

# %%
# Channel Transition Matrices
# ------------------------------------------------
# Visualize the transition matrices for each channel type.

# Channel Transition Matrix Visualization
# =======================================
# Create visual representations of how each binary channel
# transforms input bits to output bits based on transition probabilities

# Create transition matrices for each channel
p_bsc = 0.2
bsc_matrix = np.array([[1 - p_bsc, p_bsc], [p_bsc, 1 - p_bsc]])

p_bec = 0.2
# For BEC, we use -1 to represent erasure, but for visualization we'll use a 3x2 matrix
bec_matrix = np.array([[1 - p_bec, 0], [0, 1 - p_bec]])

p_z = 0.2
z_matrix = np.array([[1, 0], [p_z, 1 - p_z]])

# Plot transition matrices
matrices = [("Binary Symmetric Channel", bsc_matrix, p_bsc), ("Binary Erasure Channel", bec_matrix, p_bec), ("Z-Channel", z_matrix, p_z)]

fig4, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig4.suptitle("Binary Channel Transition Matrices", fontsize=16, fontweight="bold")

for i, (name, matrix, p) in enumerate(matrices):
    ax = axes[i]
    im = ax.imshow(matrix, cmap=PlottingUtils.MATRIX_CMAP, interpolation="nearest", aspect="auto")
    ax.set_title(f"{name}\n(p={p:.2f})")
    ax.set_xlabel("Output")
    ax.set_ylabel("Input")

    # Add text annotations
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > 0.5 else "black"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)

fig4.show()

# %%
# Channel Capacity Analysis
# ------------------------------------------------
# Analyze the information capacity of different binary channels.

# Channel Capacity Analysis
# ========================
# Calculate and compare the theoretical channel capacities for different
# binary channel types as their parameters vary

# Parameter ranges for capacity analysis
p_range = np.linspace(0, 1, 51)


# Calculate capacities for each channel type
def calculate_bsc_capacity(p):
    """Calculate BSC capacity: C = 1 - H(p)"""
    if p == 0 or p == 1:
        return 1.0 if p == 0 else 0.0
    return 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)


def calculate_bec_capacity(p):
    """Calculate BEC capacity: C = 1 - p"""
    return 1 - p


def calculate_z_capacity(p):
    """Calculate Z-channel capacity."""
    if p == 0:
        return 1.0
    if p == 1:
        return 0.0
    # Z-channel capacity formula
    return 1 + (1 - p) * np.log2(1 - p) + p * np.log2(p)


# Calculate capacities
bsc_capacities = np.array([calculate_bsc_capacity(p) for p in p_range])
bec_capacities = np.array([calculate_bec_capacity(p) for p in p_range])
z_capacities = np.array([calculate_z_capacity(p) for p in p_range])

# Plot capacity analysis
capacities = {"BSC": bsc_capacities, "BEC": bec_capacities, "Z-Channel": z_capacities}

fig5 = PlottingUtils.plot_capacity_analysis(p_range, capacities, "Binary Channel Capacity Analysis")
fig5.show()

# %%
# Conclusion
# ------------------
# This example demonstrated the use of different binary channel models in Kaira:
#
# - The **Binary Symmetric Channel** randomly flips bits with probability p,
#   affecting both 0→1 and 1→0 transitions equally.
#
# - The **Binary Erasure Channel** randomly erases bits with probability p,
#   converting them to an erasure symbol (often useful in coding theory).
#
# - The **Z-Channel** has asymmetric error probability, where only 1→0
#   transitions occur with probability p (common in some physical systems).
#
# These binary channels serve as building blocks for more complex digital communication
# systems and are fundamental in information theory for analyzing capacity and error rates.
