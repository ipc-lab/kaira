"""
==========================================
Data Generation Utilities
==========================================

This example demonstrates the new HuggingFace-based data generation utilities in Kaira,
including binary and uniform datasets creation. These utilities are particularly
useful for creating synthetic data for information theory and communication systems experiments.
"""

import matplotlib.pyplot as plt
import numpy as np

from kaira.data import (
    BinaryTensorDataset,
    UniformTensorDataset,
)

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# %%
# 1. Basic Dataset Generation
# ---------------------------------------------------------
# Binary and Uniform Dataset Creation
# ==================================
#
# Let's start with the basic dataset generation functions using HuggingFace datasets.
# These functions create datasets with specific distributions.

# Create a binary dataset (values are 0 or 1)
binary_dataset = BinaryTensorDataset(n_samples=1000, feature_shape=(10,), prob=0.3)  # Each sample has shape (10,)

# Create a uniform dataset (values are uniformly distributed)
uniform_dataset = UniformTensorDataset(n_samples=1000, feature_shape=(10,), low=-2.0, high=2.0)  # Each sample has shape (10,)

print(f"Binary dataset size: {len(binary_dataset)}")
print(f"Uniform dataset size: {len(uniform_dataset)}")
print(f"Binary sample shape: {np.array(binary_dataset[0]['data']).shape}")
print(f"Uniform sample shape: {np.array(uniform_dataset[0]['data']).shape}")

# Get first few samples for analysis
binary_samples = [binary_dataset[i]["data"] for i in range(100)]
binary_data = np.array(binary_samples).flatten()

uniform_samples = [uniform_dataset[i]["data"] for i in range(100)]
uniform_data = np.array(uniform_samples).flatten()

print(f"Average value in binary data: {binary_data.mean():.4f} (expected: 0.3)")
print(f"Uniform data range: [{uniform_data.min():.4f}, {uniform_data.max():.4f}]")

# %%
# Visualizing the generated data
# ---------------------------------------------------------
# Data Distribution Visualization
# =================================
#
# Let's visualize the generated data to understand their distributions.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot binary data
axes[0].stem(binary_data[:100])
axes[0].set_title("Binary Data (first 100 values)")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)

# Plot uniform data
axes[1].hist(uniform_data, bins=30, alpha=0.7)
axes[1].set_title("Uniform Data Distribution")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 2. Controlling the Probability in Binary Datasets
# ---------------------------------------------------------
# Binary Probability Control Demonstration
# ========================================
#
# We can control the probability of 1s in the binary dataset.
# This is useful for simulating different types of sources.

# Create binary datasets with different probabilities
probs = [0.1, 0.3, 0.5, 0.7, 0.9]
binary_datasets = [BinaryTensorDataset(n_samples=1000, feature_shape=(50,), prob=p) for p in probs]

# Calculate the actual frequencies of 1s
actual_freqs = []
for dataset in binary_datasets:
    # Get all data from the dataset
    all_data: list[float] = []
    for i in range(len(dataset)):
        all_data.extend(dataset[i]["data"])
    actual_freqs.append(np.mean(all_data))

# Visualize the results
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(probs, actual_freqs, width=0.05, alpha=0.7)
ax.plot([0, 1], [0, 1], "r--", label="Expected")
ax.scatter(probs, actual_freqs, s=100, c="red", zorder=3)

for p, f in zip(probs, actual_freqs):
    ax.annotate(f"{f:.3f}", (p, float(f)), xytext=(0, 10), textcoords="offset points", ha="center")

ax.set_xlabel("Target Probability (p)")
ax.set_ylabel("Actual Frequency of 1s")
ax.set_title("Controlling Binary Dataset Distribution")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.show()

# %%
# 3. Using Dataset Classes for Streaming Data
# ---------------------------------------------------------
# HuggingFace Dataset Usage
# =========================
#
# The new HuggingFace-based datasets support streaming and
# efficient data processing.

# Create datasets using the dataset classes directly
binary_hf_dataset = BinaryTensorDataset(n_samples=500, feature_shape=(20,), prob=0.4)

uniform_hf_dataset = UniformTensorDataset(n_samples=500, feature_shape=(20,), low=0.0, high=1.0)

print(f"Binary HF dataset length: {len(binary_hf_dataset)}")
print(f"Uniform HF dataset length: {len(uniform_hf_dataset)}")


# Demonstrate batch processing with HF datasets
def process_batch(dataset, batch_size=10):
    """Process dataset in batches."""
    batched_data = []
    for i in range(0, len(dataset), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(dataset))):
            batch.append(dataset[j]["data"])
        batched_data.append(np.array(batch))
    return batched_data


# Process data in batches
binary_batches = process_batch(binary_hf_dataset, batch_size=50)
uniform_batches = process_batch(uniform_hf_dataset, batch_size=50)

print(f"Number of binary batches: {len(binary_batches)}")
print(f"Binary batch shape: {binary_batches[0].shape}")
print(f"Number of uniform batches: {len(uniform_batches)}")
print(f"Uniform batch shape: {uniform_batches[0].shape}")

# %%
# 4. Communication Channel Simulation
# ---------------------------------------------------------
# Binary Symmetric Channel Example
# ===============================
#
# Let's simulate a simple binary symmetric channel using
# the generated binary data.

# Parameters for the channel simulation
message_length = 100
batch_size = 20
flip_prob = 0.1  # Channel error probability

# Generate random binary messages using HF datasets
message_dataset = BinaryTensorDataset(n_samples=batch_size, feature_shape=(message_length,), prob=0.5)


# Function to simulate channel noise
def add_channel_noise(messages, flip_prob):
    """Add binary symmetric channel noise."""
    noisy_messages = []
    for msg in messages:
        msg_array = np.array(msg)
        # Generate noise mask
        noise_mask = np.random.binomial(1, flip_prob, size=msg_array.shape)
        # Apply XOR operation (flip bits where noise is 1)
        noisy_msg = (msg_array + noise_mask) % 2
        noisy_messages.append(noisy_msg.tolist())
    return noisy_messages


# Get messages and add noise
messages = [message_dataset[i]["data"] for i in range(batch_size)]
noisy_messages = add_channel_noise(messages, flip_prob)


# Calculate bit error rate
def calculate_ber(original, noisy):
    """Calculate bit error rate."""
    total_bits = 0
    error_bits = 0
    for orig, noise in zip(original, noisy):
        orig_array = np.array(orig)
        noise_array = np.array(noise)
        total_bits += len(orig_array)
        error_bits += np.sum(orig_array != noise_array)
    return error_bits / total_bits


ber = calculate_ber(messages, noisy_messages)
print(f"Channel flip probability: {flip_prob}")
print(f"Measured bit error rate: {ber:.4f}")
print(f"Expected bit error rate: {flip_prob:.4f}")

# Visualize the first message and its noisy version
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Original message
axes[0].stem(range(len(messages[0])), messages[0])
axes[0].set_title("Original Binary Message")
axes[0].set_ylabel("Bit Value")
axes[0].grid(True, alpha=0.3)

# Noisy message
axes[1].stem(range(len(noisy_messages[0])), noisy_messages[0])
axes[1].set_title(f"Noisy Message (flip prob = {flip_prob})")
axes[1].set_xlabel("Bit Index")
axes[1].set_ylabel("Bit Value")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 5. Advanced Dataset Features
# ---------------------------------------------------------
# Dataset Filtering and Mapping
# =============================
#
# HuggingFace datasets support advanced operations like filtering and mapping.

# Create a larger dataset for demonstration
large_dataset = BinaryTensorDataset(n_samples=1000, feature_shape=(10,), prob=0.3)


# Filter samples based on sum (number of 1s)
def filter_by_sum(example):
    """Keep only samples with sum >= 3."""
    return sum(example["data"]) >= 3


filtered_dataset = large_dataset.filter(filter_by_sum)

print(f"Original dataset size: {len(large_dataset)}")
print(f"Filtered dataset size: {len(filtered_dataset)}")


# Map function to calculate features
def add_features(example):
    """Add sum and mean as features."""
    data = example["data"]
    example["sum"] = sum(data)
    example["mean"] = sum(data) / len(data)
    return example


enhanced_dataset = filtered_dataset.map(add_features)

# Show some examples
print("\\nSample enhanced data:")
for i in range(min(5, len(enhanced_dataset))):
    sample = enhanced_dataset[i]
    print(f"Sample {i}: sum={sample['sum']}, mean={sample['mean']:.3f}")

print("\\nData generation examples completed!")
print("The new HuggingFace-based datasets provide:")
print("- Native streaming support")
print("- Advanced filtering and mapping operations")
print("- Better integration with modern ML frameworks")
print("- Efficient data processing for large datasets")
