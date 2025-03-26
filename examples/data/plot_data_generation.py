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

# %%
# Imports and Setup
# ---------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from kaira.data import (
    create_binary_tensor,
    create_uniform_tensor,
    BinaryTensorDataset,
    UniformTensorDataset
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# 1. Basic Tensor Generation
# ---------------------------------------------------------
# Let's start with the basic tensor generation functions.
# These functions are useful for creating synthetic data with
# specific distributions.

# Create a binary tensor (values are 0 or 1)
binary_tensor = create_binary_tensor(size=[1, 1000], prob=0.3)

# Create a uniform tensor (values are uniformly distributed)
uniform_tensor = create_uniform_tensor(size=[1, 1000], low=-2.0, high=2.0)

print(f"Binary tensor shape: {binary_tensor.shape}")
print(f"Average value in binary tensor: {binary_tensor.mean().item():.4f} (expected: 0.3)")
print(f"Uniform tensor shape: {uniform_tensor.shape}")
print(f"Uniform tensor range: [{uniform_tensor.min().item():.4f}, {uniform_tensor.max().item():.4f}]")

# %%
# Visualizing the generated tensors
# ---------------------------------------------------------
# Let's visualize the generated tensors to understand their distributions.

plt.figure(figsize=(12, 5))

# Plot binary tensor
plt.subplot(1, 2, 1)
plt.stem(binary_tensor[0, :100].numpy())
plt.title("Binary Tensor (first 100 values)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)

# Plot uniform tensor
plt.subplot(1, 2, 2)
plt.hist(uniform_tensor.numpy().flatten(), bins=30, alpha=0.7)
plt.title("Uniform Tensor Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 2. Controlling the Probability in Binary Tensors
# ---------------------------------------------------------
# We can control the probability of 1s in the binary tensor.
# This is useful for simulating different types of sources.

# Create binary tensors with different probabilities
probs = [0.1, 0.3, 0.5, 0.7, 0.9]
binary_tensors = [create_binary_tensor(size=[1, 5000], prob=p) for p in probs]

# Calculate the actual frequencies of 1s
actual_freqs = [tensor.mean().item() for tensor in binary_tensors]

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(probs, actual_freqs, width=0.05, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', label='Expected')
plt.scatter(probs, actual_freqs, s=100, c='red', zorder=3)

for p, f in zip(probs, actual_freqs):
    plt.annotate(f"{f:.3f}", (p, f), xytext=(0, 10), 
                 textcoords="offset points", ha='center')

plt.xlabel("Target Probability (p)")
plt.ylabel("Actual Frequency of 1s")
plt.title("Controlling Binary Tensor Distribution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

# %%
# 3. Using Dataset Classes for Batch Processing
# ---------------------------------------------------------
# Kaira provides dataset classes that wrap the tensor generation
# for easier batch processing in training loops.

# Create datasets
n_samples = 1000
feature_dim = 10

# Binary dataset with 30% probability of 1s
binary_dataset = BinaryTensorDataset(size=[n_samples, feature_dim], prob=0.3)

# Uniform dataset with values between -1 and 1
uniform_dataset = UniformTensorDataset(size=[n_samples, feature_dim], low=-1.0, high=1.0)

print(f"Binary dataset size: {len(binary_dataset)}")
print(f"Feature dimension: {binary_dataset[0].shape}")
print(f"Average value in binary dataset: {binary_dataset.data.mean().item():.4f}")

print(f"Uniform dataset size: {len(uniform_dataset)}")
print(f"Feature dimension: {uniform_dataset[0].shape}")
print(f"Average value in uniform dataset: {uniform_dataset.data.mean().item():.4f}")

# %%
# Visualizing dataset samples
# ---------------------------------------------------------
# Let's visualize some samples from our datasets.

plt.figure(figsize=(12, 8))

# Plot binary dataset samples
plt.subplot(2, 1, 1)
for i in range(5):
    plt.plot(binary_dataset[i].numpy(), 'o-', alpha=0.7, label=f"Sample {i+1}")
plt.title("Binary Dataset Samples")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot uniform dataset samples
plt.subplot(2, 1, 2)
for i in range(5):
    plt.plot(uniform_dataset[i].numpy(), 'o-', alpha=0.7, label=f"Sample {i+1}")
plt.title("Uniform Dataset Samples")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# %%
# 4. Creating a Mini-Batch Loader
# ---------------------------------------------------------
# We can use the PyTorch DataLoader with our dataset classes
# to create mini-batches for training.

from torch.utils.data import DataLoader

# Create a DataLoader for the binary dataset
batch_size = 32
binary_loader = DataLoader(binary_dataset, batch_size=batch_size, shuffle=True)

# Get a batch
batch = next(iter(binary_loader))
print(f"Batch shape: {batch.shape}")
print(f"Number of batches: {len(binary_loader)}")

# Visualize the batch
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(batch.numpy(), aspect='auto', cmap='binary')
plt.title(f"Binary Batch ({batch_size} samples)")
plt.xlabel("Feature Index")
plt.ylabel("Sample Index")
plt.colorbar(label="Value")

# Show the mean for each feature across the batch
plt.subplot(1, 2, 2)
feature_means = batch.mean(dim=0).numpy()
plt.bar(np.arange(feature_dim), feature_means, alpha=0.7)
plt.axhline(y=0.3, color='r', linestyle='--', label="Expected Mean (p=0.3)")
plt.title("Feature Means Across Batch")
plt.xlabel("Feature Index")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 5. Practical Use Case: Channel Coding Simulation
# ---------------------------------------------------------
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
    return (x_reshaped.sum(dim=-1) > repeat/2).float()

decoded = majority_decoder(received, repeat=3)

# Calculate bit error rate
original_bits = messages.numel()
error_bits = (decoded != messages).sum().item()
bit_error_rate = error_bits / original_bits

print(f"Bit Error Rate (BER): {bit_error_rate:.4f}")

# Visualize the coding process
plt.figure(figsize=(12, 10))

# Plot original messages
plt.subplot(4, 1, 1)
plt.imshow(messages.numpy(), cmap='binary', aspect='auto')
plt.title("Original Messages")
plt.ylabel("Message")
plt.colorbar(ticks=[0, 1], orientation="vertical", pad=0.05)

# Plot encoded messages
plt.subplot(4, 1, 2)
plt.imshow(encoded.numpy(), cmap='binary', aspect='auto')
plt.title("Encoded Messages (3x Repetition)")
plt.ylabel("Message")
plt.colorbar(ticks=[0, 1], orientation="vertical", pad=0.05)

# Plot received messages
plt.subplot(4, 1, 3)
plt.imshow(received.numpy(), cmap='binary', aspect='auto')
plt.title("Received Messages (After Noisy Channel)")
plt.ylabel("Message")
plt.colorbar(ticks=[0, 1], orientation="vertical", pad=0.05)

# Plot decoded messages
plt.subplot(4, 1, 4)
plt.imshow(decoded.numpy(), cmap='binary', aspect='auto')
plt.title(f"Decoded Messages (BER: {bit_error_rate:.4f})")
plt.ylabel("Message")
plt.xlabel("Bit Position")
plt.colorbar(ticks=[0, 1], orientation="vertical", pad=0.05)

plt.tight_layout()
plt.show()

# %%
# Conclusion
# -------------------------------------------------------------
# This example demonstrated the data generation utilities in Kaira:
#
# 1. Basic tensor generation with specific distributions
# 2. Controlling the probability parameters
# 3. Using dataset classes for batch processing
# 4. Practical application in a channel coding simulation
#
# These utilities provide a foundation for experiments in
# information theory, communications, and machine learning
# for communication systems.