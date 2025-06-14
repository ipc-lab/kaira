"""
==========================
Channel Comparison
==========================

This example demonstrates how to create a visually appealing comparison of different
communication channels in Kaira. We'll visualize the effects of various channels
on transmitted signals and compare their characteristics.
"""

# %%
# Imports and Setup
# --------------------------
#
# First, let's import the necessary libraries and set up our environment.

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

from kaira.channels import (
    AWGNChannel,
    BinaryErasureChannel,
    BinarySymmetricChannel,
    FlatFadingChannel,
    RayleighFadingChannel,
)
from kaira.data import create_binary_tensor, create_uniform_tensor
from kaira.utils import seed_everything

# Set seeds for reproducibility
seed_everything(42)

# Set a visually appealing style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Create a custom colormap for attractive visualizations
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
cmap = LinearSegmentedColormap.from_list("kaira_cmap", colors)

# %%
# Creating Input Data
# -------------------------------------
#
# We'll create both binary and continuous input data to test with our channels.

# Create binary data
binary_data = create_binary_tensor(size=(1000, 1))
binary_data_torch = binary_data.clone().detach()  # Properly clone the tensor

# Create continuous data (uniform distribution between -1 and 1)
continuous_data = create_uniform_tensor(size=(1000, 1), low=-1, high=1)
continuous_data_torch = continuous_data.clone().detach()  # Properly clone the tensor

# %%
# Channel Setup
# ----------------------
#
# Initialize different channel models with various parameters

# AWGN Channel with different noise levels
awgn_low = AWGNChannel(snr_db=20)  # High SNR (low noise)
awgn_med = AWGNChannel(snr_db=10)  # Medium SNR
awgn_high = AWGNChannel(snr_db=0)  # Low SNR (high noise)

# Binary Symmetric Channel with different crossover probabilities
bsc_low = BinarySymmetricChannel(crossover_prob=0.05)  # Low error probability
bsc_high = BinarySymmetricChannel(crossover_prob=0.2)  # High error probability

# Binary Erasure Channel
bec = BinaryErasureChannel(erasure_prob=0.15)  # Correct parameter name

# Fading Channels
fading = FlatFadingChannel(snr_db=15, fading_type="rayleigh", coherence_time=100)
rayleigh = RayleighFadingChannel(snr_db=15)

# %%
# Visualizing Channel Effects
# --------------------------------------------
#
# Let's see how each channel affects the transmitted signals.


# First, let's set up a function to apply channels and collect results
def apply_channels(data, channels, channel_names):
    """Apply multiple communication channels to input data and collect results.

    This function takes input data and passes it through each of the provided
    channels, collecting the output along with the channel's name.

    Parameters
    -------------------
    data : torch.Tensor
        The input data to be transmitted through the channels.
    channels : list
        A list of channel objects that implement the __call__ method.
    channel_names : list of str
        A list of names corresponding to each channel for labeling.

    Returns
    -------
    list
        A list of tuples, where each tuple contains the channel output
        and the corresponding channel name.
    """
    results = []
    for channel, name in zip(channels, channel_names):
        received = channel(data)
        results.append((received, name))
    return results


# Channels for continuous data
continuous_channels = [awgn_low, awgn_med, awgn_high, fading, rayleigh]
continuous_names = ["AWGN (SNR=20dB)", "AWGN (SNR=10dB)", "AWGN (SNR=0dB)", "Flat Fading (SNR=15dB)", "Rayleigh Fading (SNR=15dB)"]

# Channels for binary data
binary_channels = [awgn_low, bsc_low, bsc_high, bec]
binary_names = ["AWGN (SNR=20dB)", "BSC (p=0.05)", "BSC (p=0.2)", "BEC (p=0.15)"]

# Apply channels to data
continuous_results = apply_channels(continuous_data_torch, continuous_channels, continuous_names)
binary_results = apply_channels(binary_data_torch, binary_channels, binary_names)

# %%
# Visualizing Continuous Data Results
# -------------------------------------------------------------
#
# Let's create scatter plots to see how each channel affects continuous data.

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Add an "Original" plot
axes[0].scatter(np.arange(100), continuous_data[:100], color=colors[0], alpha=0.7, s=40)
axes[0].set_title("Original Signal", fontsize=14)
axes[0].set_xlabel("Sample Index", fontsize=12)
axes[0].set_ylabel("Signal Value", fontsize=12)

# Plot each channel result
for i, (received, name) in enumerate(continuous_results):
    # Convert to numpy for plotting
    received_np = received.numpy() if isinstance(received, torch.Tensor) else received

    # Handle complex data by taking only the real part for plotting
    if np.iscomplexobj(received_np):
        received_np = received_np.real

    # Plot
    axes[i + 1].scatter(np.arange(100), received_np[:100], color=colors[i + 1], alpha=0.7, s=40)
    axes[i + 1].set_title(f"{name}", fontsize=14)
    axes[i + 1].set_xlabel("Sample Index", fontsize=12)
    axes[i + 1].set_ylabel("Signal Value", fontsize=12)

plt.tight_layout()
plt.suptitle("Effects of Different Channels on Continuous Data (First 100 Samples)", fontsize=16, y=1.02)
plt.show()

# %%
# Visualizing Binary Data Results
# ---------------------------------------------------------
#
# Now, let's create visualizations to show how each channel affects binary data.

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()


# Function to create an effective visualization for binary data
def plot_binary_transmission(ax, original, received, title, color_idx=0):
    """Plot binary data transmission with error highlighting."""
    samples = 100  # Number of samples to display

    # Convert to numpy if needed
    original_np = original[:samples].numpy().flatten() if isinstance(original, torch.Tensor) else original[:samples].flatten()
    received_np = received[:samples].numpy().flatten() if isinstance(received, torch.Tensor) else received[:samples].flatten()

    # Create a plot showing the original and received bits
    x = np.arange(samples)
    ax.step(x, original_np, where="mid", label="Original", color=colors[0], linewidth=2)
    ax.step(x, received_np, where="mid", label="Received", color=colors[color_idx + 1], linewidth=2, alpha=0.7)

    # Highlight errors or erasures
    for i in range(samples):
        if received_np[i] != original_np[i]:
            if received_np[i] == 0.5:  # Erasure (0.5 is used for erasure in BEC)
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color="orange")
            else:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color="red")

    # Adjust the plot aesthetics
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Bit Index", fontsize=12)
    ax.set_ylabel("Bit Value", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)


# Plot original data
plot_binary_transmission(axes[0], binary_data, binary_data, "Original Binary Signal", 0)

# Plot results for each channel
for i, (received, name) in enumerate(binary_results):
    plot_binary_transmission(axes[i + 1], binary_data_torch, received, name, i)

# Remove any unused axes
for i in range(len(binary_results) + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.suptitle("Effects of Different Channels on Binary Data (First 100 Bits)", fontsize=16, y=1.02)
plt.show()

# %%
# Creating a Heatmap Visualization of Channel Reliability
# --------------------------------------------------------------------------------------------------
#
# Let's create a heatmap that shows the reliability of different channels
# under various conditions.

# Create a matrix of bit error rates for different channels and SNR values
snr_values = np.linspace(0, 20, 11)  # SNR from 0 to 20 dB

# Initialize channels for each SNR value
awgn_channels = [AWGNChannel(snr_db=snr) for snr in snr_values]
bsc_channels = [BinarySymmetricChannel(crossover_prob=0.5 * np.exp(-snr / 10)) for snr in snr_values]
bec_channels = [BinaryErasureChannel(erasure_prob=0.5 * np.exp(-snr / 10)) for snr in snr_values]

# Create test data
test_data = create_binary_tensor(size=(10000, 1))
test_data_torch = test_data.clone().detach()  # Properly clone the tensor


# Calculate error rates
def calculate_error_rate(original, received):
    """Calculate the error rate between original and received signals.

    This function computes the bit error rate (BER) between the original transmitted
    data and the received data after passing through a channel. It handles both
    PyTorch tensors and NumPy arrays, and properly accounts for erasures in the
    Binary Erasure Channel (BEC) by treating them as errors.

    Parameters
    -------------------
    original : torch.Tensor or numpy.ndarray
        The original transmitted data.
    received : torch.Tensor or numpy.ndarray
        The received data after passing through a channel.

    Returns
    -------
    float
        The proportion of bits that differ between the original and received data,
        i.e., the bit error rate.
    """
    original_np = original.numpy() if isinstance(original, torch.Tensor) else original
    received_np = received.numpy() if isinstance(received, torch.Tensor) else received
    # For BEC, treat erasures (0.5) as errors
    received_np = np.where(np.isclose(received_np, 0.5), 2, received_np)
    return np.mean(original_np != received_np)


# Store error rates for each channel and SNR value
error_rates: Dict[str, List[float]] = {"AWGN": [], "BSC": [], "BEC": []}

for awgn, bsc, bec in zip(awgn_channels, bsc_channels, bec_channels):
    # Apply channels
    awgn_received = awgn(test_data_torch)
    bsc_received = bsc(test_data_torch)
    bec_received = bec(test_data_torch)

    # Calculate error rates
    error_rates["AWGN"].append(calculate_error_rate(test_data, awgn_received))
    error_rates["BSC"].append(calculate_error_rate(test_data, bsc_received))
    error_rates["BEC"].append(calculate_error_rate(test_data, bec_received))

# Create a heatmap of error rates
plt.figure(figsize=(12, 8))

# Prepare data for heatmap
channels = list(error_rates.keys())
error_matrix = np.array([error_rates[channel] for channel in channels])

# Create heatmap with a custom colormap
sns.heatmap(error_matrix, annot=True, fmt=".3f", cmap="viridis_r", xticklabels=[f"{snr:.1f}" for snr in snr_values], yticklabels=channels)

plt.title("Bit Error Rate Comparison Across Channels and SNR Values", fontsize=16)
plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14)
plt.ylabel("Channel Type", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# 3D Visualization of Channel Characteristics
# ------------------------------------------------------------------------------
#
# Let's create a 3D plot to visualize how the noise variance and signal power
# affect the performance of an AWGN channel.


# Define parameter ranges
signal_powers = np.linspace(0.1, 2, 10)  # Signal power levels
noise_vars = np.linspace(0.1, 2, 10)  # Noise variance levels

# Create a mesh grid
signal_grid, noise_grid = np.meshgrid(signal_powers, noise_vars)

# Calculate SNR for each combination
snr_grid = 10 * np.log10(signal_grid / noise_grid)

# Calculate theoretical bit error rate for BPSK in AWGN
# (Using approximate formula: Q(sqrt(2*SNR)) ≈ 0.5*exp(-SNR))
ber_grid = 0.5 * np.exp(-np.power(10, snr_grid / 10))

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(signal_grid, noise_grid, ber_grid, cmap=cmap, edgecolor="none", alpha=0.8)  # type: ignore[attr-defined]

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Bit Error Rate")

# Set labels and title
ax.set_xlabel("Signal Power", fontsize=12)
ax.set_ylabel("Noise Variance", fontsize=12)
ax.set_zlabel("Bit Error Rate", fontsize=12)  # type: ignore[attr-defined]
ax.set_title("Theoretical Bit Error Rate for AWGN Channel\nwith varying Signal Power and Noise", fontsize=16)

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)  # type: ignore[attr-defined]

plt.tight_layout()
plt.show()

# %%
# Constellation Visualization
# --------------------------------------------
#
# Let's create a visualization of signal constellations before and after
# passing through different channels.


# Create QPSK constellation (4-QAM)
def create_qpsk_symbols(n_symbols=1000):
    """Generate random QPSK (Quadrature Phase-Shift Keying) symbols.

    This function creates a sequence of random QPSK symbols from the normalized
    constellation points [(1+1j), (1-1j), (-1+1j), (-1-1j)]/√2. These points
    are uniformly distributed on a circle with unit energy.

    Parameters
    -------------------
    n_symbols : int, optional
        The number of symbols to generate, by default 1000

    Returns
    -------
    numpy.ndarray
        An array of complex values representing the generated QPSK symbols.
    """
    # QPSK symbols at (±1±1j)/√2
    symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    # Randomly select from the constellation
    indices = np.random.choice(len(symbols), size=n_symbols)
    return symbols[indices]


# Create 16-QAM constellation
def create_16qam_symbols(n_symbols=1000):
    """Generate random 16-QAM (Quadrature Amplitude Modulation) symbols.

    This function creates a sequence of random 16-QAM symbols from a constellation
    with 16 points. The constellation consists of symbols with real and imaginary
    parts taking values from {-3, -1, 1, 3}, normalized to have unit average energy.

    Parameters
    -------------------
    n_symbols : int, optional
        The number of symbols to generate, by default 1000

    Returns
    -------
    numpy.ndarray
        An array of complex values representing the generated 16-QAM symbols.
    """
    # 16-QAM symbols
    real_parts = np.array([-3, -1, 1, 3])
    imag_parts = np.array([-3, -1, 1, 3])
    symbols = np.array([complex(r, i) for r in real_parts for i in imag_parts]) / np.sqrt(10)
    # Randomly select from the constellation
    indices = np.random.choice(len(symbols), size=n_symbols)
    return symbols[indices]


# Create symbols
qpsk_symbols = create_qpsk_symbols(1000)
qam_symbols = create_16qam_symbols(1000)


# Convert to torch tensors (complex numbers as 2D real tensors)
# Convert complex symbols to real-valued 2D representation
def complex_to_real(x):
    """Convert complex-valued symbols to real-valued 2D representation.

    This function transforms complex numbers into a 2D real-valued representation
    by stacking the real and imaginary parts as columns in a matrix. This is
    useful for processing complex constellation points in machine learning
    frameworks that primarily work with real-valued data.

    Parameters
    -------------------
    x : array_like
        An array of complex numbers to be converted.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array where each row contains [real_part, imaginary_part]
        of the corresponding complex number in the input.
    """
    return np.column_stack((np.real(x), np.imag(x)))


qpsk_data = torch.tensor(complex_to_real(qpsk_symbols), dtype=torch.float32)
qam_data = torch.tensor(complex_to_real(qam_symbols), dtype=torch.float32)

# Apply channels to constellations
awgn_5db = AWGNChannel(snr_db=5)
awgn_15db = AWGNChannel(snr_db=15)
fading_10db = FlatFadingChannel(fading_type="rayleigh", coherence_time=1, snr_db=10)

# Process data through channels
qpsk_awgn_5db = awgn_5db(qpsk_data).numpy()
qpsk_awgn_15db = awgn_15db(qpsk_data).numpy()
qpsk_fading = fading_10db(qpsk_data).numpy()

qam_awgn_5db = awgn_5db(qam_data).numpy()
qam_awgn_15db = awgn_15db(qam_data).numpy()
qam_fading = fading_10db(qam_data).numpy()

# Create constellation plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# QPSK plots
axes[0, 0].scatter(qpsk_data[:, 0], qpsk_data[:, 1], c=colors[0], alpha=0.7, s=30, label="Original")
axes[0, 0].set_title("QPSK Original", fontsize=14)

axes[0, 1].scatter(qpsk_awgn_5db[:, 0], qpsk_awgn_5db[:, 1], c=colors[1], alpha=0.7, s=30, label="AWGN 5dB")
axes[0, 1].set_title("QPSK + AWGN (SNR=5dB)", fontsize=14)

axes[0, 2].scatter(qpsk_awgn_15db[:, 0], qpsk_awgn_15db[:, 1], c=colors[2], alpha=0.7, s=30, label="AWGN 15dB")
axes[0, 2].set_title("QPSK + AWGN (SNR=15dB)", fontsize=14)

axes[0, 3].scatter(qpsk_fading[:, 0], qpsk_fading[:, 1], c=colors[3], alpha=0.7, s=30, label="Fading")
axes[0, 3].set_title("QPSK + Flat Fading (SNR=10dB)", fontsize=14)

# 16-QAM plots
axes[1, 0].scatter(qam_data[:, 0], qam_data[:, 1], c=colors[0], alpha=0.7, s=30, label="Original")
axes[1, 0].set_title("16-QAM Original", fontsize=14)

axes[1, 1].scatter(qam_awgn_5db[:, 0], qam_awgn_5db[:, 1], c=colors[1], alpha=0.7, s=30, label="AWGN 5dB")
axes[1, 1].set_title("16-QAM + AWGN (SNR=5dB)", fontsize=14)

axes[1, 2].scatter(qam_awgn_15db[:, 0], qam_awgn_15db[:, 1], c=colors[2], alpha=0.7, s=30, label="AWGN 15dB")
axes[1, 2].set_title("16-QAM + AWGN (SNR=15dB)", fontsize=14)

axes[1, 3].scatter(qam_fading[:, 0], qam_fading[:, 1], c=colors[3], alpha=0.7, s=30, label="Fading")
axes[1, 3].set_title("16-QAM + Flat Fading (SNR=10dB)", fontsize=14)

# Add grid and labels to all axes
for row in axes:
    for ax in row:
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlabel("In-phase Component", fontsize=12)
        ax.set_ylabel("Quadrature Component", fontsize=12)
        # Set equal aspect ratio to maintain the shape of the constellation
        ax.set_aspect("equal")
        # Set axis limits to better view the constellation
        lim = 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

plt.tight_layout()
plt.suptitle("Signal Constellations Before and After Channel Transmission", fontsize=18, y=1.02)
plt.show()

# %%
# Conclusion
# -------------------
#
# In this visualization example, we have demonstrated how to create attractive
# and informative visualizations of different communication channels in Kaira.
# These visualizations help in understanding the characteristics and effects of
# various channels on transmitted signals.
#
# The key takeaways from these visualizations are:
#
# 1. AWGN channels add Gaussian noise that increases with decreasing SNR.
# 2. Binary channels like BSC and BEC introduce different types of errors.
# 3. Fading channels introduce both attenuation and phase shift to the signal.
# 4. Signal constellations are useful for visualizing modulation schemes and
#    channel effects.
#
# These visualizations can help in designing robust communication systems by
# understanding the behavior of different channels and their impact on signal
# transmission.
