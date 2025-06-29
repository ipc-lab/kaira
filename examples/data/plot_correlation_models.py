"""
Correlation Models for Wyner-Ziv Coding
========================================

This example demonstrates various correlation models used in distributed
source coding and Wyner-Ziv compression using the new CorrelatedDataset.

We explore different correlation coefficients and visualize the relationship
between source and side information signals.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.data import CorrelatedDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

###############################################################################
# Generate Correlated Data
# ========================
#
# Create datasets with different correlation coefficients

# Define correlation levels to test
correlations = [0.2, 0.5, 0.8, 0.95]
n_samples = 1000
signal_length = 128

# Generate correlated data for each correlation level
datasets = {}
for corr in correlations:
    datasets[corr] = CorrelatedDataset(length=n_samples, shape=(signal_length,), correlation=corr, noise_std=0.1, seed=42)

###############################################################################
# Visualize Signal Correlation
# ============================
#
# Plot source vs side information for different correlation levels

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, corr in enumerate(correlations):
    # Get a sample from the dataset
    source, side_info = datasets[corr][0]

    # Convert to numpy for plotting
    source_np = source.numpy()
    side_info_np = side_info.numpy()

    # Scatter plot of first 100 samples
    axes[i].scatter(source_np[:100], side_info_np[:100], alpha=0.6, s=10)
    axes[i].set_title(f"Correlation = {corr}")
    axes[i].set_xlabel("Source Signal")
    axes[i].set_ylabel("Side Information")
    axes[i].grid(True, alpha=0.3)

    # Add correlation line
    x_range = np.linspace(source_np.min(), source_np.max(), 100)
    y_range = corr * x_range
    axes[i].plot(x_range, y_range, "r--", alpha=0.8, label=f"y = {corr}x")
    axes[i].legend()

plt.tight_layout()
plt.suptitle("Source-Side Information Correlation", y=1.02, fontsize=14)
plt.show()

###############################################################################
# Measure Empirical Correlation
# =============================
#
# Calculate actual correlation coefficients for validation

print("Empirical vs Theoretical Correlation:")
print("=====================================")

for corr in correlations:
    # Generate multiple samples and calculate correlation
    sources = []
    side_infos = []

    for i in range(100):  # Use 100 samples for statistics
        source, side_info = datasets[corr][i]
        sources.append(source.numpy().flatten())
        side_infos.append(side_info.numpy().flatten())

    # Combine all samples
    all_sources = np.concatenate(sources)
    all_side_infos = np.concatenate(side_infos)

    # Calculate empirical correlation
    empirical_corr = np.corrcoef(all_sources, all_side_infos)[0, 1]

    print(f"Theoretical: {corr:.2f}, Empirical: {empirical_corr:.3f}")

###############################################################################
# Time Series Visualization
# =========================
#
# Show how correlated signals evolve over time

plt.figure(figsize=(15, 8))

# Use high correlation for clearer visualization
high_corr_dataset = CorrelatedDataset(length=1, shape=(200,), correlation=0.85, noise_std=0.1, seed=42)

source, side_info = high_corr_dataset[0]
time_steps = np.arange(len(source))

plt.subplot(2, 1, 1)
plt.plot(time_steps, source.numpy(), "b-", label="Source Signal", linewidth=1.5)
plt.plot(time_steps, side_info.numpy(), "r--", label="Side Information", linewidth=1.5)
plt.title("Correlated Signals Over Time (ρ = 0.85)")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

# Show the difference signal
plt.subplot(2, 1, 2)
difference = source.numpy() - side_info.numpy()
plt.plot(time_steps, difference, "g-", linewidth=1.5)
plt.title("Difference Signal (Source - Side Information)")
plt.xlabel("Time Step")
plt.ylabel("Difference")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Statistical Analysis
# ====================
#
# Analyze the statistical properties of the correlation model

print("\nStatistical Properties:")
print("======================")

for corr in [0.5, 0.8]:
    dataset = CorrelatedDataset(length=1000, shape=(64,), correlation=corr, noise_std=0.1, seed=42)

    # Collect statistics
    source_vars = []
    side_info_vars = []
    correlations_empirical = []

    for i in range(100):
        source, side_info = dataset[i]
        source_np = source.numpy()
        side_info_np = side_info.numpy()

        source_vars.append(np.var(source_np))
        side_info_vars.append(np.var(side_info_np))
        correlations_empirical.append(np.corrcoef(source_np, side_info_np)[0, 1])

    print(f"\nCorrelation {corr}:")
    print(f"  Source variance: {np.mean(source_vars):.3f} ± {np.std(source_vars):.3f}")
    print(f"  Side info variance: {np.mean(side_info_vars):.3f} ± {np.std(side_info_vars):.3f}")
    print(f"  Empirical correlation: {np.mean(correlations_empirical):.3f} ± {np.std(correlations_empirical):.3f}")
