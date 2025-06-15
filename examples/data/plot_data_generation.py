"""
==========================================
Data Generation Utilities
==========================================

This example demonstrates the data generation utilities in Kaira,
including binary and uniform tensor creation, as well as dataset
classes for batch processing. These utilities are particularly
useful for creating synthetic data for information theory and
communication systems experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kaira.data import (
    BinaryTensorDataset,
    UniformTensorDataset,
    create_binary_tensor,
    create_uniform_tensor,
)

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# %%
# Imports and Setup
# ---------------------------------------------------------
# Data Generation Configuration and Reproducibility Setup
# =======================================================

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# 1. Basic Tensor Generation
# ---------------------------------------------------------
# Binary and Uniform Tensor Creation
# ==================================
#
# Let's start with the basic tensor generation functions.
# These functions are useful for creating synthetic data with
# specific distributions.

# Create a binary tensor (values are 0 or 1)
binary_tensor = create_binary_tensor(size=[1, 1000], prob=0.3)

# Create a uniform tensor (values are uniformly distributed)
uniform_tensor = create_uniform_tensor(size=[1, 1000], low=-2.0, high=2.0)

# Tensor Generation Results:
# Binary tensor shape: {binary_tensor.shape}
# Average value in binary tensor: {binary_tensor.mean().item():.4f} (expected: 0.3)
# Uniform tensor shape: {uniform_tensor.shape}
# Uniform tensor range: [{uniform_tensor.min().item():.4f}, {uniform_tensor.max().item():.4f}]

# %%
# Visualizing the generated tensors
# ---------------------------------------------------------
# Tensor Distribution Visualization
# =================================
#
# Let's visualize the generated tensors to understand their distributions.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot binary tensor
axes[0].stem(binary_tensor[0, :100].numpy())
axes[0].set_title("Binary Tensor (first 100 values)")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)

# Plot uniform tensor
axes[1].hist(uniform_tensor.numpy().flatten(), bins=30, alpha=0.7)
axes[1].set_title("Uniform Tensor Distribution")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 2. Controlling the Probability in Binary Tensors
# ---------------------------------------------------------
# Binary Probability Control Demonstration
# ========================================
#
# We can control the probability of 1s in the binary tensor.
# This is useful for simulating different types of sources.

# Create binary tensors with different probabilities
probs = [0.1, 0.3, 0.5, 0.7, 0.9]
binary_tensors = [create_binary_tensor(size=[1, 5000], prob=p) for p in probs]

# Calculate the actual frequencies of 1s
actual_freqs = [tensor.mean().item() for tensor in binary_tensors]

# Visualize the results
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(probs, actual_freqs, width=0.05, alpha=0.7)
ax.plot([0, 1], [0, 1], "r--", label="Expected")
ax.scatter(probs, actual_freqs, s=100, c="red", zorder=3)

for p, f in zip(probs, actual_freqs):
    ax.annotate(f"{f:.3f}", (p, f), xytext=(0, 10), textcoords="offset points", ha="center")

ax.set_xlabel("Target Probability (p)")
ax.set_ylabel("Actual Frequency of 1s")
ax.set_title("Controlling Binary Tensor Distribution")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()

# %%
# 3. Using Dataset Classes for Batch Processing
# ---------------------------------------------------------
# Dataset Creation and Analysis
# =============================
#
# Kaira provides dataset classes that wrap the tensor generation
# for easier batch processing in training loops.

# Create datasets
n_samples = 1000
feature_dim = 10

# Binary dataset with 30% probability of 1s
binary_dataset = BinaryTensorDataset(size=[n_samples, feature_dim], prob=0.3)

# Uniform dataset with values between -1 and 1
uniform_dataset = UniformTensorDataset(size=[n_samples, feature_dim], low=-1.0, high=1.0)

# Dataset Information:
# Binary dataset size: {len(binary_dataset)}
# Feature dimension: {binary_dataset[0].shape}
# Average value in binary dataset: {binary_dataset.data.mean().item():.4f}
#
# Uniform dataset size: {len(uniform_dataset)}
# Feature dimension: {uniform_dataset[0].shape}
# Average value in uniform dataset: {uniform_dataset.data.mean().item():.4f}

# %%
# Visualizing dataset samples
# ---------------------------------------------------------
# Dataset Sample Visualization
# ============================
#
# Let's visualize some samples from our datasets.

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot binary dataset samples
for i in range(5):
    axes[0].plot(binary_dataset[i].numpy(), "o-", alpha=0.7, label=f"Sample {i+1}")
axes[0].set_title("Binary Dataset Samples")
axes[0].set_xlabel("Feature Index")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot uniform dataset samples
for i in range(5):
    axes[1].plot(uniform_dataset[i].numpy(), "o-", alpha=0.7, label=f"Sample {i+1}")
axes[1].set_title("Uniform Dataset Samples")
axes[1].set_xlabel("Feature Index")
axes[1].set_ylabel("Value")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# %%
# 4. Creating a Mini-Batch Loader
# ---------------------------------------------------------
# Mini-Batch Processing with DataLoader
# =====================================
#
# We can use the PyTorch DataLoader with our dataset classes
# to create mini-batches for training.

# Create a DataLoader for the binary dataset
batch_size = 32
binary_loader = DataLoader(binary_dataset, batch_size=batch_size, shuffle=True)

# Get a batch
batch = next(iter(binary_loader))

# Batch Information:
# Batch shape: {batch.shape}
# Number of batches: {len(binary_loader)}

# Visualize the batch
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(batch.numpy(), aspect="auto", cmap="binary")
axes[0].set_title(f"Binary Batch ({batch_size} samples)")
axes[0].set_xlabel("Feature Index")
axes[0].set_ylabel("Sample Index")

# Show the mean for each feature across the batch
feature_means = batch.mean(dim=0).numpy()
bars = axes[1].bar(np.arange(feature_dim), feature_means, alpha=0.7)
axes[1].axhline(y=0.3, color="r", linestyle="--", label="Expected Mean (p=0.3)")
axes[1].set_title("Feature Means Across Batch")
axes[1].set_xlabel("Feature Index")
axes[1].set_ylabel("Mean Value")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 5. Practical Use Case: Channel Coding Simulation
# ---------------------------------------------------------
# Channel Coding Simulation Example
# =================================
#
# Let's demonstrate a practical use case where we generate
# binary data for a simple channel coding simulation.

# Generate random binary data (message bits)
message_length = 4
batch_size = 8
messages = create_binary_tensor(size=[batch_size, message_length], prob=0.5)


# Simple repetition code: repeat each bit 3 times
def repetition_encoder(x, repeat=3):
    """Simple repetition encoder."""
    return x.repeat_interleave(repeat, dim=-1)


encoded = repetition_encoder(messages, repeat=3)


# Simulate a noisy channel (bit flipping with 20% probability)
def binary_symmetric_channel(x, flip_prob=0.2):
    """Binary symmetric channel with bit flipping."""
    noise = create_binary_tensor(size=x.shape, prob=flip_prob)
    return (x + noise) % 2  # XOR operation


received = binary_symmetric_channel(encoded, flip_prob=0.2)


# Simple majority vote decoder
def majority_decoder(x, repeat=3):
    """Majority vote decoder for repetition code."""
    x_reshaped = x.reshape(x.shape[0], -1, repeat)
    return (x_reshaped.sum(dim=-1) > repeat / 2).float()


decoded = majority_decoder(received, repeat=3)

# Calculate bit error rate
original_bits = messages.numel()
error_bits = (decoded != messages).sum().item()
bit_error_rate = error_bits / original_bits

# Channel Coding Results:
# Bit Error Rate (BER): {bit_error_rate:.4f}

# Visualize the coding process
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Plot original messages
im1 = axes[0].imshow(messages.numpy(), cmap="binary", aspect="auto")
axes[0].set_title("Original Messages")
axes[0].set_ylabel("Message")
plt.colorbar(im1, ax=axes[0], ticks=[0, 1], orientation="horizontal", pad=0.05)

# Plot encoded messages
im2 = axes[1].imshow(encoded.numpy(), cmap="binary", aspect="auto")
axes[1].set_title("Encoded Messages (3x Repetition)")
axes[1].set_ylabel("Message")
plt.colorbar(im2, ax=axes[1], ticks=[0, 1], orientation="horizontal", pad=0.05)

# Plot received messages
im3 = axes[2].imshow(received.numpy(), cmap="binary", aspect="auto")
axes[2].set_title("Received Messages (After Noisy Channel)")
axes[2].set_ylabel("Message")
plt.colorbar(im3, ax=axes[2], ticks=[0, 1], orientation="horizontal", pad=0.05)

# Plot decoded messages
im4 = axes[3].imshow(decoded.numpy(), cmap="binary", aspect="auto")
axes[3].set_title(f"Decoded Messages (BER: {bit_error_rate:.4f})")
axes[3].set_ylabel("Message")
axes[3].set_xlabel("Bit Position")
plt.colorbar(im4, ax=axes[3], ticks=[0, 1], orientation="horizontal", pad=0.05)

plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------
# Data Generation Summary
# ====================================
#
# This example demonstrated the data generation utilities in Kaira:
#
# Key Features:
# - Binary tensor generation with controllable probability
# - Uniform tensor generation with custom ranges
# - Dataset classes for batch processing
# - Integration with PyTorch DataLoader
# - Practical application in channel coding simulation
#
# These utilities provide a foundation for:
# • Information theory experiments
# • Communication system simulations
# • Machine learning data preparation
# • Statistical analysis and visualization
