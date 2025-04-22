"""
=========================================================
Complex Projections for Wireless Communications
=========================================================

This example demonstrates the usage of complex-valued projections in Kaira
for dimensionality reduction in wireless communication systems. Complex projections
are essential for efficiently representing signals with in-phase (I) and
quadrature (Q) components commonly found in wireless communications.

We'll visualize and compare:
1. Real-valued projections (Rademacher, Gaussian, Orthogonal)
2. Complex-valued projections (Complex Gaussian, Complex Orthogonal)
3. Applications to wireless channel modeling and signal compression
"""

# %%
# Imports and Setup
# ----------------------------------
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

from kaira.models.components import Projection, ProjectionType
from kaira.utils import seed_everything

# Set seeds for reproducibility
seed_everything(42)

# Set plotting style for better visualization
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Define custom color palette for visualizations
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
sns.set_palette(sns.color_palette(colors))

# %%
# Understanding Complex Projections in Communications
# ----------------------------------------------------
# Complex projections map high-dimensional complex data to lower-dimensional complex spaces
# while preserving important properties of the data. In wireless communications, complex
# projections are useful for:
#
# 1. Dimensionality reduction of I/Q signals
# 2. Efficient representation of channel state information
# 3. Feature extraction for wireless signals
# 4. Compressive sensing of sparse spectrum
#
# Let's create and visualize different types of projections

# Create input dimensions
dim = 32
output_dim = 32

# Create different projection types
proj_radem = Projection(dim, output_dim, projection_type=ProjectionType.RADEMACHER, seed=42)
proj_gauss = Projection(dim, output_dim, projection_type=ProjectionType.GAUSSIAN, seed=42)
proj_ortho = Projection(dim, output_dim, projection_type=ProjectionType.ORTHOGONAL, seed=42)
proj_complex_gauss = Projection(dim, output_dim, projection_type=ProjectionType.COMPLEX_GAUSSIAN, seed=42)
proj_complex_ortho = Projection(dim, output_dim, projection_type=ProjectionType.COMPLEX_ORTHOGONAL, seed=42)

# Extract projection matrices
radem_matrix = proj_radem.projection.detach().cpu().numpy()
gauss_matrix = proj_gauss.projection.detach().cpu().numpy()
ortho_matrix = proj_ortho.projection.detach().cpu().numpy()
complex_gauss_matrix = proj_complex_gauss.projection.detach().cpu().numpy()
complex_ortho_matrix = proj_complex_ortho.projection.detach().cpu().numpy()

# %%
# Visualizing Projection Matrices
# -------------------------------------
# Let's visualize the real and complex projection matrices

# Create a larger figure with more space for subplots and colorbars
fig = plt.figure(figsize=(18, 16))  # Increase figure size even more
gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1], hspace=0.6, wspace=0.6)  # Increase spacing further

# Plot real-valued projections (magnitude)
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(np.abs(radem_matrix), cmap="viridis")
ax1.set_title("Rademacher Projection\n(Magnitude)", fontsize=12)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(np.abs(gauss_matrix), cmap="viridis")
ax2.set_title("Gaussian Projection\n(Magnitude)", fontsize=12)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(np.abs(ortho_matrix), cmap="viridis")
ax3.set_title("Orthogonal Projection\n(Magnitude)", fontsize=12)
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

ax4 = fig.add_subplot(gs[0, 3])
im4 = ax4.imshow(np.abs(complex_gauss_matrix), cmap="viridis")
ax4.set_title("Complex Gaussian\n(Magnitude)", fontsize=12)
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

ax5 = fig.add_subplot(gs[0, 4])
im5 = ax5.imshow(np.abs(complex_ortho_matrix), cmap="viridis")
ax5.set_title("Complex Orthogonal\n(Magnitude)", fontsize=12)
fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

# For complex projections, also plot the phase
ax6 = fig.add_subplot(gs[1, 3])
im6 = ax6.imshow(np.angle(complex_gauss_matrix), cmap="hsv")
ax6.set_title("Complex Gaussian\n(Phase)", fontsize=12)
fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

ax7 = fig.add_subplot(gs[1, 4])
im7 = ax7.imshow(np.angle(complex_ortho_matrix), cmap="hsv")
ax7.set_title("Complex Orthogonal\n(Phase)", fontsize=12)
fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

# Plot real projections value distributions
ax8 = fig.add_subplot(gs[1, 0])
sns.histplot(radem_matrix.flatten(), bins=3, kde=False, ax=ax8)
ax8.set_title("Rademacher Values", fontsize=12)
ax8.set_xlim(-1.5, 1.5)

ax9 = fig.add_subplot(gs[1, 1])
sns.histplot(gauss_matrix.flatten(), bins=30, kde=True, ax=ax9)
ax9.set_title("Gaussian Values", fontsize=12)

ax10 = fig.add_subplot(gs[1, 2])
sns.histplot(ortho_matrix.flatten(), bins=30, kde=True, ax=ax10)
ax10.set_title("Orthogonal Values", fontsize=12)

# Add the title with more space above the plots
fig.suptitle("Comparison of Real and Complex Projection Matrices", fontsize=16, y=0.95)

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.6, hspace=0.6)

plt.show()

# %%
# Column Orthogonality Analysis
# -------------------------------------
# A key property of projections is column orthogonality.
# Let's verify the orthogonality properties for both real and complex projections.


def compute_gram_matrix(matrix):
    """Compute Gram matrix (normalized dot products between columns)."""
    if np.iscomplexobj(matrix):
        # For complex matrices, use conjugate transpose
        normalized_cols = matrix / np.sqrt(np.sum(np.abs(matrix) ** 2, axis=0))
        gram = np.abs(normalized_cols.conjugate().T @ normalized_cols)
    else:
        # For real matrices
        normalized_cols = matrix / np.sqrt(np.sum(matrix**2, axis=0))
        gram = np.abs(normalized_cols.T @ normalized_cols)

    return gram


# Compute Gram matrices
gram_radem = compute_gram_matrix(radem_matrix)
gram_gauss = compute_gram_matrix(gauss_matrix)
gram_ortho = compute_gram_matrix(ortho_matrix)
gram_complex_gauss = compute_gram_matrix(complex_gauss_matrix)
gram_complex_ortho = compute_gram_matrix(complex_ortho_matrix)


# Calculate off-diagonal means (lower is better for orthogonality)
def off_diagonal_mean(matrix):
    """Calculate the mean of off-diagonal elements in a matrix.

    This function computes the mean value of all off-diagonal elements in the input matrix,
    which is useful for measuring orthogonality in projection matrices. Lower values indicate
    better orthogonality between columns.

    Args:
        matrix (numpy.ndarray): A square matrix for which to calculate the mean of
            off-diagonal elements.

    Returns:
        float: The mean value of the off-diagonal elements.
    """
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)
    return np.mean(off_diag)


off_means = {"Rademacher": off_diagonal_mean(gram_radem), "Gaussian": off_diagonal_mean(gram_gauss), "Orthogonal": off_diagonal_mean(gram_ortho), "Complex Gaussian": off_diagonal_mean(gram_complex_gauss), "Complex Orthogonal": off_diagonal_mean(gram_complex_ortho)}

# Visualize Gram matrices
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ["Rademacher", "Gaussian", "Orthogonal", "Complex Gaussian", "Complex Orthogonal"]
matrices = [gram_radem, gram_gauss, gram_ortho, gram_complex_gauss, gram_complex_ortho]

for i, (title, matrix) in enumerate(zip(titles, matrices)):
    im = axes[i].imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    axes[i].set_title(f"{title}\nMean Off-Diagonal: {off_means[title]:.4f}", fontsize=10)
    plt.colorbar(im, ax=axes[i])
    axes[i].set_xlabel("Column Index")
    axes[i].set_ylabel("Column Index")

plt.suptitle("Gram Matrices (Column Orthogonality)", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Application: Complex Signal Projection and Reconstruction
# ------------------------------------------------------------
# Let's demonstrate how complex projections can be used for wireless signal processing.
# We'll create a complex OFDM-like signal and compare the performance of
# different projection methods for dimensionality reduction.

# Parameters
signal_length = 1024  # Signal length
n_subcarriers = 64  # Number of subcarriers in our OFDM-like signal
compression_ratios = [0.75, 0.5, 0.25, 0.1]  # Compression ratios to test

# Generate a complex OFDM-like signal
# First, create complex symbols (QPSK-like constellation)
np.random.seed(42)
constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
symbols = np.random.choice(constellation, size=n_subcarriers)

# Zero-pad to create frequency-domain signal
freq_domain = np.zeros(signal_length, dtype=complex)
freq_domain[:n_subcarriers] = symbols

# Convert to time domain (using IFFT)
time_domain = np.fft.ifft(freq_domain) * np.sqrt(signal_length)

# Create Tensor from our signal (batched)
signal = torch.tensor(time_domain, dtype=torch.complex64).reshape(1, -1)


# Set up a function for projection and reconstruction with error calculation
def project_and_reconstruct(signal, projection_type, compression_ratio):
    """Project and reconstruct a signal using the specified projection type and ratio."""
    in_dim = signal.shape[1]
    out_dim = int(in_dim * compression_ratio)

    # Create projection
    proj = Projection(in_dim, out_dim, projection_type=projection_type, seed=42)

    # Project signal
    # Handle complex vs real compatibility
    is_complex_proj = proj.is_complex
    is_complex_signal = torch.is_complex(signal)

    if is_complex_proj and not is_complex_signal:
        # Convert real signal to complex for complex projection
        signal_to_project = torch.complex(signal, torch.zeros_like(signal))
    elif not is_complex_proj and is_complex_signal:
        # Use only real part of complex signal for real projection
        signal_to_project = signal.real
    else:
        # Types already match
        signal_to_project = signal

    projected = proj(signal_to_project)

    # Create reconstruction projection (transpose)
    if torch.is_complex(proj.projection):
        recon_matrix = torch.nn.Parameter(proj.projection.conj().t())
    else:
        recon_matrix = torch.nn.Parameter(proj.projection.t())

    # Reconstruct
    reconstructed = projected @ recon_matrix

    # Calculate error metrics
    if is_complex_signal:
        # For complex signals, use complex comparison
        mse = torch.mean(torch.abs(signal - reconstructed) ** 2).item()
    else:
        # For real signals with complex projection, compare with real part
        if torch.is_complex(reconstructed) and not is_complex_signal:
            mse = torch.mean(torch.abs(signal - reconstructed.real) ** 2).item()
        else:
            mse = torch.mean(torch.abs(signal - reconstructed) ** 2).item()

    psnr = 10 * np.log10(1 / mse) if mse > 0 else float("inf")

    return {"projected": projected.detach().cpu().numpy(), "reconstructed": reconstructed.detach().cpu().numpy(), "mse": mse, "psnr": psnr}


# %%
# Visualize Signal Compression Performance
# -------------------------------------------
# Let's visualize how well each projection type preserves the signal
# at different compression ratios

# Projection types to test
proj_types = [ProjectionType.RADEMACHER, ProjectionType.GAUSSIAN, ProjectionType.ORTHOGONAL, ProjectionType.COMPLEX_GAUSSIAN, ProjectionType.COMPLEX_ORTHOGONAL]

proj_type_names = ["Rademacher", "Gaussian", "Orthogonal", "Complex Gaussian", "Complex Orthogonal"]

# Set up a figure for plotting results
plt.figure(figsize=(10, 6))

# For each projection type, calculate PSNR across compression ratios
for i, proj_type in enumerate(proj_types):
    psnr_values = []
    for ratio in compression_ratios:
        result = project_and_reconstruct(signal, proj_type, ratio)
        psnr_values.append(result["psnr"])

    plt.plot(compression_ratios, psnr_values, "o-", label=proj_type_names[i], color=colors[i], linewidth=2)

plt.xlabel("Compression Ratio", fontsize=12)
plt.ylabel("PSNR (dB)", fontsize=12)
plt.title("Signal Reconstruction Quality vs. Compression Ratio", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(compression_ratios)

# Add annotations
for i, proj_type in enumerate(proj_types):
    for j, ratio in enumerate(compression_ratios):
        result = project_and_reconstruct(signal, proj_type, ratio)
        plt.annotate(f"{result['psnr']:.1f}dB", (ratio, result["psnr"]), xytext=(0, 5 + (i * 3)), textcoords="offset points", ha="center", color=colors[i], fontsize=8, fontweight="bold")

plt.tight_layout()
plt.show()

# %%
# Visualizing Signal Reconstruction
# -----------------------------------
# Let's visualize the original signal and reconstructed signals at 25% compression

# Set the compression ratio
ratio = 0.25

# Collect results for each projection type
results = {}
for proj_type, name in zip(proj_types, proj_type_names):
    results[name] = project_and_reconstruct(signal, proj_type, ratio)

# Create figure to visualize the original and reconstructed signals
fig, axes = plt.subplots(len(proj_types) + 1, 2, figsize=(14, 3 * (len(proj_types) + 1)))

# Plot original signal
original_signal = signal[0].cpu().numpy()
axes[0, 0].plot(np.real(original_signal), label="Real", color="blue", alpha=0.8)
axes[0, 0].plot(np.imag(original_signal), label="Imaginary", color="red", alpha=0.8)
axes[0, 0].set_title("Original Signal (Time Domain)")
axes[0, 0].set_xlabel("Sample Index")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot original signal frequency domain
freq_original = np.fft.fft(original_signal) / np.sqrt(len(original_signal))
freq_mag = np.abs(freq_original)
axes[0, 1].plot(freq_mag, color="green")
axes[0, 1].set_title("Original Signal (Frequency Domain)")
axes[0, 1].set_xlabel("Frequency Bin")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, n_subcarriers * 2)  # Focus on the relevant part of the spectrum

# Plot reconstructed signals
for i, name in enumerate(proj_type_names):
    recon_signal = results[name]["reconstructed"][0]

    # Time domain plot
    axes[i + 1, 0].plot(np.real(recon_signal), label="Real", color="blue", alpha=0.8)
    axes[i + 1, 0].plot(np.imag(recon_signal), label="Imaginary", color="red", alpha=0.8)
    axes[i + 1, 0].set_title(f"{name} Reconstruction (PSNR: {results[name]['psnr']:.1f}dB)")
    axes[i + 1, 0].set_xlabel("Sample Index")
    axes[i + 1, 0].set_ylabel("Amplitude")
    axes[i + 1, 0].legend()
    axes[i + 1, 0].grid(True, alpha=0.3)

    # Frequency domain plot
    freq_recon = np.fft.fft(recon_signal) / np.sqrt(len(recon_signal))
    freq_mag_recon = np.abs(freq_recon)
    axes[i + 1, 1].plot(freq_mag_recon, color="green")
    axes[i + 1, 1].plot(freq_mag, color="gray", linestyle="--", alpha=0.5, label="Original")
    axes[i + 1, 1].set_title(f"{name} Reconstruction (Frequency Domain)")
    axes[i + 1, 1].set_xlabel("Frequency Bin")
    axes[i + 1, 1].set_ylabel("Magnitude")
    axes[i + 1, 1].grid(True, alpha=0.3)
    axes[i + 1, 1].set_xlim(0, n_subcarriers * 2)  # Focus on the relevant part of the spectrum

plt.tight_layout()
plt.suptitle(f"Signal Reconstruction with {ratio:.0%} Compression Ratio", fontsize=16, y=1.02)
plt.show()

# %%
# Conclusion
# --------------------------------------------
# This example has demonstrated:
#
# 1. **Complex Projections in Kaira**: Implementation of complex Gaussian and orthogonal
#    projections for wireless communications
#
# 2. **Column Orthogonality**: Analysis of how well each projection preserves orthogonality,
#    which is critical for signal reconstruction
#
# 3. **Signal Compression Performance**: Comparison of projection methods for compressing
#    complex communication signals at various ratios
#
# 4. **Time and Frequency Domain Visualization**: Visual comparison of original and
#    reconstructed signals to understand fidelity
#
# Key observations:
#
# - Complex orthogonal projections provide the best reconstruction quality for complex signals
# - As expected, all projection methods show decreasing performance with higher compression
# - For wireless signals, complex projections maintain better phase information
# - Complex projections allow direct manipulation of complex signals without converting to real representations
#
# These projection methods can be applied to various wireless communication applications:
# - Channel state information compression
# - MIMO system dimensionality reduction
# - Feature extraction for ML-based wireless systems
# - Efficient signal representation for limited-capacity channels
