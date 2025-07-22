"""
Data Generation with Modern Datasets
====================================

This example demonstrates how to use the new Kaira data generation classes
for creating various types of synthetic data useful in communication
systems research.

We'll explore binary, uniform, Gaussian, and function-based datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.data import BinaryDataset, FunctionDataset, GaussianDataset, UniformDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

###############################################################################
# Binary Data Generation
# ======================
#
# Generate binary data for digital communication experiments

# Create a binary dataset with different probabilities
n_samples = 1000
seq_length = 100

# Different bias levels
probabilities = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, prob in enumerate(probabilities):
    binary_dataset = BinaryDataset(length=n_samples, shape=(seq_length,), prob=prob, seed=42)

    # Get a sample sequence
    sample = binary_dataset[0].numpy()

    # Plot the binary sequence
    axes[i].plot(sample[:50], "o-", linewidth=1, markersize=4)
    axes[i].set_title(f"Binary Sequence (p = {prob})")
    axes[i].set_xlabel("Sample Index")
    axes[i].set_ylabel("Bit Value")
    axes[i].set_ylim(-0.1, 1.1)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("Binary Data with Different Probabilities", y=1.02)
plt.show()

###############################################################################
# Uniform and Gaussian Distributions
# ==================================
#
# Compare uniform and Gaussian noise generation

# Create datasets
uniform_dataset = UniformDataset(length=1000, shape=(256,), low=-1.0, high=1.0, seed=42)

gaussian_dataset = GaussianDataset(length=1000, shape=(256,), mean=0.0, std=0.5, seed=42)

# Generate samples and create histograms
uniform_samples = []
gaussian_samples = []

for i in range(100):
    uniform_samples.append(uniform_dataset[i].numpy())
    gaussian_samples.append(gaussian_dataset[i].numpy())

# Combine all samples
all_uniform = np.concatenate(uniform_samples)
all_gaussian = np.concatenate(gaussian_samples)

# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(all_uniform, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")
ax1.set_title("Uniform Distribution")
ax1.set_xlabel("Value")
ax1.set_ylabel("Density")
ax1.grid(True, alpha=0.3)

ax2.hist(all_gaussian, bins=50, density=True, alpha=0.7, color="red", edgecolor="black")
ax2.set_title("Gaussian Distribution")
ax2.set_xlabel("Value")
ax2.set_ylabel("Density")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Custom Function-Based Generation
# ================================
#
# Use FunctionDataset for complex signal generation


def generate_sine_wave(idx):
    """Generate a sine wave with varying frequency."""
    t = np.linspace(0, 1, 128)
    frequency = 1 + idx * 0.1  # Frequency increases with index
    signal = np.sin(2 * np.pi * frequency * t)
    return torch.from_numpy(signal.astype(np.float32))


def generate_chirp(idx):
    """Generate a linear frequency chirp."""
    t = np.linspace(0, 1, 128)
    # Frequency sweep from 1 Hz to 10 Hz
    signal = np.sin(2 * np.pi * (1 + 9 * t) * t)
    # Add some noise based on index
    noise_level = idx * 0.01
    noise = np.random.normal(0, noise_level, len(signal))
    return torch.from_numpy((signal + noise).astype(np.float32))


# Create function-based datasets
sine_dataset = FunctionDataset(length=50, generator_fn=generate_sine_wave, seed=42)
chirp_dataset = FunctionDataset(length=50, generator_fn=generate_chirp, seed=42)

# Visualize generated signals
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Sine waves with different frequencies
for i in range(2):
    signal = sine_dataset[i * 10].numpy()  # Every 10th sample
    axes[0, i].plot(signal)
    axes[0, i].set_title(f"Sine Wave (Sample {i * 10})")
    axes[0, i].set_xlabel("Time Sample")
    axes[0, i].set_ylabel("Amplitude")
    axes[0, i].grid(True, alpha=0.3)

# Chirp signals with increasing noise
for i in range(2):
    signal = chirp_dataset[i * 20].numpy()  # Every 20th sample
    axes[1, i].plot(signal)
    axes[1, i].set_title(f"Chirp Signal (Sample {i * 20})")
    axes[1, i].set_xlabel("Time Sample")
    axes[1, i].set_ylabel("Amplitude")
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Performance and Memory Efficiency
# =================================
#
# Demonstrate on-demand generation efficiency

print("Dataset Performance Comparison:")
print("==============================")

# Test dataset sizes
sizes = [1000, 10000, 100000]

for size in sizes:
    # Create a large Gaussian dataset
    dataset = GaussianDataset(length=size, shape=(512,), seed=42)

    # Measure time to access random samples
    import time

    start_time = time.time()

    # Access 100 random samples
    indices = np.random.choice(size, 100, replace=False)
    samples = [dataset[int(idx)] for idx in indices]

    end_time = time.time()

    print(f"Size {size:6d}: {(end_time - start_time)*1000:.2f} ms for 100 samples")

print("\nMemory Usage:")
print("Dataset objects are lightweight - data is generated on-demand!")
print("No large arrays stored in memory until accessed.")

###############################################################################
# Combining Multiple Data Types
# =============================
#
# Show how to combine different data sources


# Create mixed signal: binary modulation + Gaussian noise
def generate_mixed_signal(idx):
    """Generate BPSK signal with noise."""
    # Generate random binary sequence
    np.random.seed(idx + 42)  # Deterministic per index
    bits = np.random.randint(0, 2, 64)

    # BPSK modulation: 0 -> -1, 1 -> +1
    bpsk_signal = 2 * bits - 1

    # Add Gaussian noise
    noise = np.random.normal(0, 0.2, len(bpsk_signal))

    return torch.from_numpy((bpsk_signal + noise).astype(np.float32))


# Create the mixed dataset
mixed_dataset = FunctionDataset(length=100, generator_fn=generate_mixed_signal, seed=42)

# Visualize a few samples
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for i in range(4):
    signal = mixed_dataset[i * 10].numpy()
    axes[i].plot(signal, "o-", markersize=3, linewidth=1)
    axes[i].set_title(f"BPSK + Noise (Sample {i * 10})")
    axes[i].set_xlabel("Symbol Index")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=1, color="r", linestyle="--", alpha=0.5, label="+1")
    axes[i].axhline(y=-1, color="r", linestyle="--", alpha=0.5, label="-1")
    if i == 0:
        axes[i].legend()

plt.tight_layout()
plt.suptitle("Combined Binary Modulation and Gaussian Noise", y=1.02)
plt.show()
