"""
====================================================================
LDPC Coding and Belief Propagation Decoding
====================================================================

This example demonstrates Low-Density Parity-Check (LDPC) codes and
belief propagation decoding :cite:`gallager1962low` :cite:`kschischang2001factor`. We'll simulate a complete communication
system using LDPC codes over an AWGN channel and analyze the error
performance at different SNR levels.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from kaira.channels.analog import AWGNChannel
from kaira.models.fec.decoders import BeliefPropagationDecoder
from kaira.models.fec.encoders import LDPCCodeEncoder

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# %%
# Setting up
# --------------------------------------
# LDPC Code Configuration and Reproducibility Setup
# =============================================
#
# First, we set a random seed to ensure reproducibility and
# configure our visualization settings.

torch.manual_seed(42)
np.random.seed(42)

# %%
# LDPC Code Fundamentals
# --------------------------------------
# LDPC Code Matrix Definition
# ===========================
#
# LDPC codes are defined by a sparse parity-check matrix H.
# Here we create a simple parity-check matrix for demonstration.

# Define a simple parity-check matrix
parity_check_matrix = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

# Display parity-check matrix information
# Parity-check matrix (H):
# tensor([[1., 0., 1., 1., 0., 0.],
#         [0., 1., 1., 0., 1., 0.],
#         [0., 0., 0., 1., 1., 1.]])

# %%
# Visualizing the Parity-Check Matrix
# -------------------------------------------------------------------
# Matrix Sparsity Visualization
# =============================
#
# We can visualize the parity-check matrix as a binary grid,
# which helps illustrate the sparsity pattern essential for LDPC codes.

# Create visualization using utility function
PlottingUtils.plot_ldpc_matrix_comparison([parity_check_matrix], ["LDPC Parity-Check Matrix"], "LDPC Code Matrix Structure")
plt.show()

# %%
# Communication System Setup
# -------------------------------------------------
# LDPC Encoder Configuration
# ===========================
#
# We'll set up a complete communication system with an LDPC encoder,
# an AWGN channel, and a belief propagation decoder.

# Initialize the encoder
encoder = LDPCCodeEncoder(parity_check_matrix)

# LDPC Code Dimensions Analysis
# ============================
# For LDPC codes, dimensions are determined from the parity check matrix
# The parity check matrix H has dimensions (n-k) x n, where:
# - n is the codeword length
# - k is the message length
# - (n-k) is the number of parity bits
parity_bits = parity_check_matrix.shape[0]
codeword_length = parity_check_matrix.shape[1]
message_length = codeword_length - parity_bits

# Code parameters:
# Message length: {message_length} bits
# Codeword length: {codeword_length} bits
# Code rate: {message_length/codeword_length:.3f}

# %%
# Simulating Communication at Different SNR Levels
# --------------------------------------------------------------------------------------
# Let's simulate the transmission of messages over an AWGN channel at various
# SNR levels and analyze the bit error rate performance.

snr_db_values = [0, 2, 4, 6, 8, 10]
iterations_values = [5, 10, 20]
num_messages = 1000
batch_size = 100
results = {}

for bp_iters in iterations_values:
    ber_values = []
    bler_values = []

    # Initialize decoder with specific iteration count
    decoder = BeliefPropagationDecoder(encoder, bp_iters=bp_iters)

    for snr_db in tqdm(snr_db_values, desc=f"BP Iterations: {bp_iters}"):
        # Initialize the channel for current SNR
        channel = AWGNChannel(snr_db=snr_db)

        # Counters for error statistics
        total_bits = 0
        error_bits = 0
        total_blocks = 0
        error_blocks = 0

        # Process messages in batches
        for i in range(0, num_messages, batch_size):
            # Generate random messages
            message = torch.randint(0, 2, (batch_size, message_length), dtype=torch.float32)

            # Encode the messages
            codeword = encoder(message)

            # Convert to bipolar format for AWGN channel
            bipolar_codeword = 1 - 2.0 * codeword

            # Transmit over AWGN channel
            received_soft = channel(bipolar_codeword)

            # Decode the received signals
            decoded_message = decoder(received_soft)

            # Calculate errors
            bit_errors = (message != decoded_message).to(torch.float32)
            block_errors = (torch.sum(bit_errors, dim=1) > 0).to(torch.float32)

            # Update statistics
            error_bits += torch.sum(bit_errors).item()
            total_bits += message.numel()
            error_blocks += torch.sum(block_errors).item()
            total_blocks += message.shape[0]

        # Calculate error rates
        ber = error_bits / total_bits
        bler = error_blocks / total_blocks

        ber_values.append(ber)
        bler_values.append(bler)

    results[bp_iters] = {"ber": ber_values, "bler": bler_values}

# %%
# Performance Analysis
# -----------------------------------
# Error Rate Performance Visualization
# ===================================
#
# Let's visualize the performance of our LDPC code with different numbers
# of belief propagation iterations across various SNR levels.

# Extract BER data for plotting
ber_curves = []
bler_curves = []
labels = []

for bp_iters, data in results.items():
    ber_curves.append(data["ber"])
    bler_curves.append(data["bler"])
    labels.append(f"BP Iterations = {bp_iters}")

# Plot BER performance
PlottingUtils.plot_ber_performance(np.array(snr_db_values), [np.array(curve) for curve in ber_curves], labels, "BER Performance of LDPC Code", "Bit Error Rate (BER)")
plt.show()

# Plot BLER performance
PlottingUtils.plot_ber_performance(np.array(snr_db_values), [np.array(curve) for curve in bler_curves], labels, "BLER Performance of LDPC Code", "Block Error Rate (BLER)")
plt.show()

# %%
# Single Message Example
# ------------------------------------
# Individual Message Processing Demonstration
# =========================================
#
# Let's walk through the encoding, transmission, and decoding process
# for a single message to better understand the flow.

# Generate a single random message
single_message = torch.randint(0, 2, (1, message_length), dtype=torch.float32)

# Encode the message
single_codeword = encoder(single_message)

# Set channel SNR
test_snr_db = 5.0
test_channel = AWGNChannel(snr_db=test_snr_db)

# Convert to bipolar format for AWGN channel
single_bipolar = 1 - 2.0 * single_codeword

# Transmit over AWGN channel
single_received = test_channel(single_bipolar)

# Initialize decoder with 10 iterations
test_decoder = BeliefPropagationDecoder(encoder, bp_iters=10)

# Decode the received signal
single_decoded = test_decoder(single_received)

# Check if successfully decoded
success = torch.all(single_message == single_decoded).item()

# Single Message Transmission Results (SNR = {test_snr_db} dB):
# ============================================================
# Original message: {single_message.squeeze().int().tolist()}
# Encoded codeword: {single_codeword.squeeze().int().tolist()}
# Decoded message: {single_decoded.squeeze().int().tolist()}
# Decoding result: {'successful' if success else 'failed'}

# %%
# Visualizing the Transmission Process
# ------------------------------------------------------------------
# Signal Flow Visualization
# =========================
#
# Let's visualize the transmission process for our single message example.

fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)

# Plot original and encoded messages
axes[0].step(range(message_length), single_message.squeeze(), "ro-", where="mid", label="Original Message")
axes[0].step(range(codeword_length), single_codeword.squeeze(), "bo-", where="mid", label="Encoded Codeword")
axes[0].grid(True)
axes[0].legend()
axes[0].set_title("Message Encoding")
axes[0].set_ylim(-0.1, 1.1)

# Plot channel input and output
axes[1].step(range(codeword_length), single_bipolar.squeeze(), "go-", where="mid", label="Channel Input (Bipolar)")
axes[1].step(range(codeword_length), single_received.squeeze(), "mo-", where="mid", label="Channel Output (with Noise)")
axes[1].grid(True)
axes[1].legend()
axes[1].set_title(f"AWGN Channel (SNR = {test_snr_db} dB)")

# Plot comparison of original and decoded messages
axes[2].step(range(message_length), single_message.squeeze(), "ro-", where="mid", label="Original Message")
axes[2].step(range(message_length), single_decoded.squeeze(), "bo-", where="mid", label="Decoded Message")
axes[2].grid(True)
axes[2].legend()
axes[2].set_title("Decoding Result")
axes[2].set_ylim(-0.1, 1.1)

plt.show()

# %%
# Conclusion
# ------------------
# LDPC Performance Summary
# ========================
#
# This example demonstrates how LDPC codes :cite:`gallager1962low` can effectively correct
# errors introduced by noisy channels. We've shown how the performance
# improves with increased SNR and more decoding iterations.
#
# Key Insights:
# - LDPC codes are widely used in modern communication systems due to
#   their excellent error-correcting capabilities that approach the
#   Shannon limit
# - The belief propagation algorithm :cite:`kschischang2001factor` provides an
#   efficient decoding method that works well for sparse parity-check
#   matrices
# - Performance scales with both SNR and the number of decoding iterations
# - Block error rates typically decrease faster than bit error rates
