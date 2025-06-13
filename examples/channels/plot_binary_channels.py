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

import numpy as np
import torch

from examples.utils.plotting import (
    setup_plotting_style,
    plot_binary_channel_comparison,
    plot_channel_error_rates,
    plot_transition_matrices,
    plot_channel_capacity_analysis
)

from kaira.channels import BinaryErasureChannel, BinarySymmetricChannel, BinaryZChannel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
setup_plotting_style()

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
fig = plot_binary_channel_comparison(original_data, channel_outputs, 
                                   segment_start, segment_length,
                                   "Binary Channel Effects Comparison")
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
fig1 = plot_channel_error_rates(error_probs, theoretical_bsc, observed_bsc, 
                               ["BSC"], "Binary Symmetric Channel Error Rates")
fig1.show()

# Prepare BEC erasure rate data  
theoretical_bec = erasure_probs  # Theoretical erasure rate equals p
observed_bec = [erasure_rate for _, _, erasure_rate in bec_outputs]

# Plot BEC erasure rates
fig2 = plot_channel_error_rates(erasure_probs, theoretical_bec, observed_bec,
                               ["BEC"], "Binary Erasure Channel Erasure Rates") 
fig2.show()

# Prepare Z-Channel error rate data
# Theoretical error rate for Z-channel is p * P(1), where P(1) is probability of input being 1
p_one = (binary_data == 1).sum().item() / num_bits
theoretical_z = [p * p_one for p in z_error_probs]
observed_z = [err_rate * p_one for _, _, err_rate in z_outputs]

# Plot Z-Channel error rates
fig3 = plot_channel_error_rates(z_error_probs, theoretical_z, observed_z,
                               ["Z-Channel"], "Z-Channel Error Rates")
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
matrices = [
    ("Binary Symmetric Channel", bsc_matrix, p_bsc),
    ("Binary Erasure Channel", bec_matrix, p_bec),
    ("Z-Channel", z_matrix, p_z)
]

fig4 = plot_transition_matrices(matrices, "Binary Channel Transition Matrices")
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
    """Calculate Z-channel capacity"""
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
capacities = {
    "BSC": bsc_capacities,
    "BEC": bec_capacities,
    "Z-Channel": z_capacities
}

fig5 = plot_channel_capacity_analysis(p_range, capacities, 
                                    "Binary Channel Capacity Analysis")
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
