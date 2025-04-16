"""
=========================
Modulation Visualization
=========================

This example demonstrates how to create visually appealing visualizations of
different modulation schemes using the Kaira library. We'll visualize signal
constellations, decision boundaries, and performance characteristics.
"""

# %%
# Imports and Setup
# --------------------------
#
# First, let's import the necessary libraries and set up our environment.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import special

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
# Signal Constellation Visualization
# ------------------------------------------------------------
#
# Let's create beautiful visualizations of common digital modulation schemes
# by plotting their constellations.


def generate_constellation_points(modulation_type, M=None):
    """Generate constellation points for different modulation schemes."""
    if modulation_type == "BPSK":
        # BPSK: {0, 1} -> {-1, 1}
        symbols = np.array([-1, 1])
        # Convert to complex form for consistent representation
        return symbols + 0j

    elif modulation_type == "QPSK":
        # QPSK with Gray coding
        symbols = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        return symbols

    elif modulation_type == "QAM":
        # QAM constellation with M = k^2 points
        if M is None:
            M = 16  # Default to 16-QAM

        k = int(np.sqrt(M))
        if k * k != M:
            raise ValueError(f"M={M} is not a perfect square for QAM")

        # Generate k-PAM for in-phase and quadrature components
        amp_levels = np.arange(-(k - 1), k, 2)

        # Create QAM symbols from k-PAM symbols
        symbols = np.array([complex(i, q) for i in amp_levels for q in amp_levels])

        # Normalize to unit average energy
        return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    elif modulation_type == "PSK":
        # M-PSK constellation
        if M is None:
            M = 8  # Default to 8-PSK

        # Generate M points equally spaced on the unit circle
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / M)
        symbols = np.exp(1j * angles)

        return symbols

    elif modulation_type == "APSK":
        # Approximate APSK with 2 rings (16-APSK as example)
        if M is None:
            M = 16  # Default to 16-APSK

        # For 16-APSK: 4 points on inner ring, 12 on outer ring
        inner_count = 4
        outer_count = M - inner_count

        # Inner ring
        inner_angles = np.arange(0, 2 * np.pi, 2 * np.pi / inner_count)
        inner_radius = 1.0
        inner_symbols = inner_radius * np.exp(1j * inner_angles)

        # Outer ring
        outer_angles = np.arange(0, 2 * np.pi, 2 * np.pi / outer_count)
        outer_radius = 2.5  # Typical ratio between outer and inner
        outer_symbols = outer_radius * np.exp(1j * outer_angles)

        # Combine symbols
        symbols = np.concatenate([inner_symbols, outer_symbols])

        # Normalize to unit average energy
        return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    else:
        raise ValueError(f"Unknown modulation type: {modulation_type}")


def plot_constellation(ax, symbols, title, color="blue", marker="o", add_labels=False, snr_db=None):
    """Plot constellation points with optional noise cloud."""
    # Extract real and imaginary parts
    real = np.real(symbols)
    imag = np.imag(symbols)

    # Plot constellation points
    ax.scatter(real, imag, c=color, marker=marker, s=100, alpha=0.8, label="Constellation Points")

    # Add labels if requested
    if add_labels:
        for i, (r, im) in enumerate(zip(real, imag)):
            # Binary representation of the symbol index
            bin_label = format(i, f"0{int(np.log2(len(symbols)))}b")
            ax.text(r + 0.1, im + 0.1, bin_label, fontsize=10, ha="center")

    # Add noise cloud if SNR is provided
    if snr_db is not None:
        # Generate random samples around each constellation point
        samples_per_point = 200
        noise_std = 10 ** (-snr_db / 20)  # Convert SNR (dB) to noise std dev

        # For each constellation point, generate noisy samples
        for point in symbols:
            # Generate complex Gaussian noise
            noise = np.random.normal(0, noise_std / np.sqrt(2), samples_per_point) + 1j * np.random.normal(0, noise_std / np.sqrt(2), samples_per_point)

            # Add noise to the constellation point
            noisy_points = point + noise

            # Plot noisy points
            ax.scatter(np.real(noisy_points), np.imag(noisy_points), c=color, alpha=0.05, s=10, marker=".")

    # Plot decision boundaries for some common constellations
    if len(symbols) == 2:  # BPSK
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Decision Boundary")
    elif len(symbols) == 4 and np.allclose(np.abs(symbols), np.abs(symbols[0])):  # QPSK
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Decision Boundaries")

    # Add unit circle for PSK modulations
    if np.allclose(np.abs(symbols), 1.0) and len(symbols) > 2:  # PSK constellations
        circle = plt.Circle((0, 0), 1, fill=False, color="green", linestyle="-", alpha=0.7, label="Unit Circle")
        ax.add_patch(circle)

    # Add rings for APSK
    if "APSK" in title:
        # Detect the distinct radii
        radii = np.unique(np.round(np.abs(symbols), 2))
        for i, radius in enumerate(radii):
            circle = plt.Circle((0, 0), radius, fill=False, color=colors[i % len(colors)], linestyle="-", alpha=0.7, label=f"Ring {i+1}")
            ax.add_patch(circle)

    # Set axis limits (with some padding)
    max_val = np.max(np.abs(symbols)) * 1.5 if snr_db is None else np.max(np.abs(symbols)) * 2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add grid and labels
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("In-phase Component (I)", fontsize=12)
    ax.set_ylabel("Quadrature Component (Q)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")

    # Set equal aspect ratio to maintain the shape of the constellation
    ax.set_aspect("equal")


# Create constellations for different modulation schemes
bpsk_const = generate_constellation_points("BPSK")
qpsk_const = generate_constellation_points("QPSK")
qam16_const = generate_constellation_points("QAM", 16)
qam64_const = generate_constellation_points("QAM", 64)
psk8_const = generate_constellation_points("PSK", 8)
apsk16_const = generate_constellation_points("APSK", 16)

# Create figure for constellation visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot each constellation
plot_constellation(axes[0], bpsk_const, "BPSK Constellation", color=colors[0], add_labels=True)
plot_constellation(axes[1], qpsk_const, "QPSK Constellation", color=colors[1], add_labels=True)
plot_constellation(axes[2], psk8_const, "8-PSK Constellation", color=colors[2], add_labels=False)
plot_constellation(axes[3], qam16_const, "16-QAM Constellation", color=colors[3], add_labels=False)
plot_constellation(axes[4], qam64_const, "64-QAM Constellation", color=colors[4], add_labels=False)
plot_constellation(axes[5], apsk16_const, "16-APSK Constellation", color=colors[5], add_labels=False)

plt.tight_layout()
plt.suptitle("Digital Modulation Constellations", fontsize=18, y=1.02)
plt.show()

# %%
# Constellation with Noise Visualization
# -------------------------------------------------------------------------
#
# Let's visualize how noise affects the constellations at different SNR levels.

# Set up figure for noise visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# QPSK with different SNR levels
plot_constellation(axes[0], qpsk_const, "QPSK, SNR = 20 dB", color=colors[0], snr_db=20)
plot_constellation(axes[1], qpsk_const, "QPSK, SNR = 10 dB", color=colors[0], snr_db=10)
plot_constellation(axes[2], qpsk_const, "QPSK, SNR = 5 dB", color=colors[0], snr_db=5)

# 16-QAM with different SNR levels
plot_constellation(axes[3], qam16_const, "16-QAM, SNR = 25 dB", color=colors[2], snr_db=25)
plot_constellation(axes[4], qam16_const, "16-QAM, SNR = 15 dB", color=colors[2], snr_db=15)
plot_constellation(axes[5], qam16_const, "16-QAM, SNR = 10 dB", color=colors[2], snr_db=10)

plt.tight_layout()
plt.suptitle("Effect of Noise on Modulation Constellations", fontsize=18, y=1.02)
plt.show()

# %%
# 3D Constellation Visualization
# -------------------------------------------------------
#
# Let's create 3D visualizations of signal constellations to show the
# probability density of received signals.


# Create a 3D plot of constellation with noise distribution
def plot_3d_constellation(modulation_type, M=None, snr_db=15):
    """Create a 3D visualization of constellation with noise PDF."""
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")

    # Generate constellation points
    symbols = generate_constellation_points(modulation_type, M)

    # Calculate noise standard deviation
    noise_std = 10 ** (-snr_db / 20)

    # Create a grid for PDF calculation
    max_val = np.max(np.abs(symbols)) * 1.5
    x = np.linspace(-max_val, max_val, 100)
    y = np.linspace(-max_val, max_val, 100)
    X, Y = np.meshgrid(x, y)

    # Calculate PDF for each constellation point and sum them
    Z = np.zeros_like(X)
    for symbol in symbols:
        # 2D Gaussian PDF centered at the constellation point
        pdf = np.exp(-((X - np.real(symbol)) ** 2 + (Y - np.imag(symbol)) ** 2) / (2 * noise_std**2)) / (2 * np.pi * noise_std**2)
        Z += pdf / len(symbols)  # Assuming equiprobable symbols

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8, antialiased=True)

    # Plot constellation points
    real = np.real(symbols)
    imag = np.imag(symbols)
    ax.scatter(real, imag, np.zeros_like(real), c="red", s=100, marker="o")

    # Add vertical lines from each point to surface
    for i, (r, im) in enumerate(zip(real, imag)):
        pdf_height = np.exp(-(0) / (2 * noise_std**2)) / (2 * np.pi * noise_std**2)
        ax.plot([r, r], [im, im], [0, pdf_height / len(symbols)], "r-", linewidth=2, alpha=0.7)

    # Set labels and title
    ax.set_xlabel("In-phase Component (I)", fontsize=12)
    ax.set_ylabel("Quadrature Component (Q)", fontsize=12)
    ax.set_zlabel("Probability Density", fontsize=12)
    title = f"3D Visualization of {modulation_type}"
    if M is not None:
        title += f"-{M}"
    title += f" (SNR = {snr_db} dB)"
    ax.set_title(title, fontsize=16)

    # Add a color bar
    fig = plt.gcf()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Adjust the viewing angle
    ax.view_init(elev=25, azim=-40)

    plt.tight_layout()
    plt.show()


# Create 3D visualizations for different modulations
plot_3d_constellation("QPSK", snr_db=10)
plot_3d_constellation("QAM", M=16, snr_db=15)

# %%
# Bit Mapping Visualization
# -----------------------------------------
#
# Let's visualize how bits are mapped to symbols in different modulation schemes
# with a focus on Gray coding.


# Create a visualization of Gray-coded bit mapping
def plot_bit_mapping(modulation_type, M=None):
    """Create a visualization of bit mapping for different modulation schemes."""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Generate constellation points
    symbols = generate_constellation_points(modulation_type, M)
    num_bits = int(np.log2(len(symbols)))

    # Generate Gray-coded bit mappings for the constellation
    gray_codes = [format(i ^ (i >> 1), f"0{num_bits}b") for i in range(len(symbols))]

    # Plot constellation points with bit labels
    real = np.real(symbols)
    imag = np.imag(symbols)

    # Plot constellation points
    ax.scatter(real, imag, c=colors[0], marker="o", s=200, alpha=0.8)

    # Add bit labels
    for i, (r, im, bits) in enumerate(zip(real, imag, gray_codes)):
        ax.text(r, im, bits, fontsize=12, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Connect neighboring points (ones that differ by only 1 bit)
    for i, code_i in enumerate(gray_codes):
        for j, code_j in enumerate(gray_codes):
            # Count bit differences between codes
            bit_diff = sum(bit_i != bit_j for bit_i, bit_j in zip(code_i, code_j))
            if bit_diff == 1:  # If they differ by exactly 1 bit
                ax.plot([real[i], real[j]], [imag[i], imag[j]], "k-", alpha=0.3)

    # Set axis limits (with some padding)
    max_val = np.max(np.abs(symbols)) * 1.2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Add grid and labels
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("In-phase Component (I)", fontsize=12)
    ax.set_ylabel("Quadrature Component (Q)", fontsize=12)

    title = f"Gray-Coded Bit Mapping for {modulation_type}"
    if M is not None:
        title += f"-{M}"
    ax.set_title(title, fontsize=16)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# Create bit mapping visualizations
plot_bit_mapping("QPSK")
plot_bit_mapping("QAM", 16)

# %%
# Decision Regions Visualization
# -------------------------------------------------------
#
# Let's create visualizations of decision regions for different modulation schemes.


def plot_decision_regions(modulation_type, M=None, snr_db=None):
    """Create a visualization of decision regions for different modulation schemes."""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Generate constellation points
    symbols = generate_constellation_points(modulation_type, M)
    num_bits = int(np.log2(len(symbols)))

    # Create a grid for decision regions
    max_val = np.max(np.abs(symbols)) * 1.5
    grid_points = 200
    x = np.linspace(-max_val, max_val, grid_points)
    y = np.linspace(-max_val, max_val, grid_points)
    X, Y = np.meshgrid(x, y)
    grid_complex = X + 1j * Y

    # Find closest constellation point for each grid point
    distances = np.abs(grid_complex.reshape(-1, 1) - symbols.reshape(1, -1))
    closest_idx = np.argmin(distances, axis=1).reshape(grid_points, grid_points)

    # Create a colormap for decision regions
    region_cmap = plt.colormaps["tab20"].resampled(len(symbols))

    # Plot decision regions
    im = ax.pcolormesh(X, Y, closest_idx, cmap=region_cmap, alpha=0.5, shading="auto")

    # Add some noise to constellation points if SNR is provided
    if snr_db is not None:
        # Generate random samples around each constellation point
        samples_per_point = 200
        noise_std = 10 ** (-snr_db / 20)

        # For each constellation point, generate noisy samples
        noisy_points = []
        point_indices = []

        for i, point in enumerate(symbols):
            # Generate complex Gaussian noise
            noise = np.random.normal(0, noise_std / np.sqrt(2), samples_per_point) + 1j * np.random.normal(0, noise_std / np.sqrt(2), samples_per_point)

            # Add noise to the constellation point
            noisy = point + noise
            noisy_points.extend(noisy)
            point_indices.extend([i] * samples_per_point)

        # Plot noisy points
        noisy_real = np.real(noisy_points)
        noisy_imag = np.imag(noisy_points)
        ax.scatter(noisy_real, noisy_imag, c="black", alpha=0.2, s=10, marker=".")

    # Plot constellation points
    real = np.real(symbols)
    imag = np.imag(symbols)
    ax.scatter(real, imag, c=range(len(symbols)), cmap=region_cmap, marker="o", s=100, alpha=1, edgecolors="black")

    # Generate Gray-coded bit mappings for the constellation
    gray_codes = [format(i ^ (i >> 1), f"0{num_bits}b") for i in range(len(symbols))]

    # Add bit labels
    for i, (r, im, bits) in enumerate(zip(real, imag, gray_codes)):
        ax.text(r, im, bits, fontsize=10, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    # Add grid and labels
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("In-phase Component (I)", fontsize=12)
    ax.set_ylabel("Quadrature Component (Q)", fontsize=12)

    title = f"Decision Regions for {modulation_type}"
    if M is not None:
        title += f"-{M}"
    if snr_db is not None:
        title += f" (SNR = {snr_db} dB)"
    ax.set_title(title, fontsize=16)

    # Set equal aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# Create decision region visualizations
plot_decision_regions("QPSK", snr_db=10)
plot_decision_regions("QAM", 16, snr_db=15)

# %%
# Eye Diagram Visualization
# -----------------------------------------
#
# Let's create an eye diagram visualization to show signal quality and timing issues.


def plot_eye_diagram(modulation_type, snr_db=20, samples_per_symbol=8, num_symbols=100):
    """Create an eye diagram visualization."""
    plt.figure(figsize=(12, 8))

    # Generate random symbols based on modulation type
    if modulation_type == "BPSK":
        symbols = np.random.choice([-1, 1], size=num_symbols)
    elif modulation_type == "QPSK":
        # We'll use the in-phase component for the eye diagram
        symbols = np.random.choice([-1 / np.sqrt(2), 1 / np.sqrt(2)], size=num_symbols)
    else:
        # For simplicity, we'll use the in-phase component of 16-QAM
        symbol_values = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        symbols = np.random.choice(symbol_values, size=num_symbols)

    # Upsample to create the signal
    signal = np.zeros(num_symbols * samples_per_symbol)
    for i, symbol in enumerate(symbols):
        signal[i * samples_per_symbol : (i + 1) * samples_per_symbol] = symbol

    # Apply pulse shaping (simple rectangular pulse for demonstration)
    # For a more realistic eye diagram, use a pulse shaping filter (e.g., raised cosine)

    # Add noise
    noise_std = 10 ** (-snr_db / 20)
    noise = np.random.normal(0, noise_std, size=len(signal))
    noisy_signal = signal + noise

    # Create the eye diagram by overlaying segments of the signal
    # Each segment is 2 symbols long (2 * samples_per_symbol)
    segment_length = 2 * samples_per_symbol

    # Extract segments and plot them
    for i in range(num_symbols - 2):
        start_idx = i * samples_per_symbol
        segment = noisy_signal[start_idx : start_idx + segment_length]
        plt.plot(np.arange(segment_length), segment, "b-", alpha=0.3)

    # Add vertical lines to indicate symbol boundaries
    plt.axvline(x=samples_per_symbol, color="r", linestyle="--", alpha=0.7, label="Symbol Boundary")

    # Add horizontal lines to indicate decision thresholds
    if modulation_type == "BPSK":
        plt.axhline(y=0, color="g", linestyle="--", alpha=0.7, label="Decision Threshold")
    elif modulation_type == "QPSK":
        plt.axhline(y=0, color="g", linestyle="--", alpha=0.7, label="Decision Threshold")
    else:  # 16-QAM
        thresholds = np.array([-2, 0, 2]) / np.sqrt(10)
        for i, threshold in enumerate(thresholds):
            if i == 0:
                plt.axhline(y=threshold, color="g", linestyle="--", alpha=0.7, label="Decision Thresholds")
            else:
                plt.axhline(y=threshold, color="g", linestyle="--", alpha=0.7)

    # Set axis labels and title
    plt.xlabel("Time (samples)", fontsize=12)
    plt.ylabel("Signal Amplitude", fontsize=12)
    plt.title(f"Eye Diagram for {modulation_type} (SNR = {snr_db} dB)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Create eye diagram visualizations
plot_eye_diagram("BPSK", snr_db=10)
plot_eye_diagram("QPSK", snr_db=10)
plot_eye_diagram("16-QAM", snr_db=15)

# %%
# Symbol Error Rate Visualization
# --------------------------------------------------------
#
# Let's visualize the theoretical Symbol Error Rate (SER) for different modulation
# schemes as a function of SNR.


def theoretical_ser(modulation_type, M, snr_db):
    """Calculate theoretical SER for different modulation schemes."""
    snr_linear = 10 ** (snr_db / 10)

    if modulation_type == "PSK":
        if M == 2:  # BPSK
            return special.erfc(np.sqrt(snr_linear)) / 2
        else:  # M-PSK
            return special.erfc(np.sqrt(snr_linear) * np.sin(np.pi / M)) / 2

    elif modulation_type == "QAM":
        # Approximation for square QAM
        k = int(np.sqrt(M))
        if k * k != M:
            raise ValueError(f"M={M} is not a perfect square for QAM")

        ser_pam = 2 * (1 - 1 / k) * special.erfc(np.sqrt(3 * snr_linear / (2 * (k**2 - 1))))
        return 1 - (1 - ser_pam) ** 2

    elif modulation_type == "FSK":
        if M == 2:  # Binary FSK
            return special.erfc(np.sqrt(snr_linear / 2)) / 2
        else:  # M-FSK
            return (M - 1) * special.erfc(np.sqrt(snr_linear)) / (2 * M)

    else:
        raise ValueError(f"Unknown modulation type: {modulation_type}")


# Create SER vs SNR plot for different modulation schemes
plt.figure(figsize=(12, 8))

# SNR range
snr_db_range = np.linspace(0, 20, 100)

# Calculate SER for different modulation schemes
ser_bpsk = [theoretical_ser("PSK", 2, snr) for snr in snr_db_range]
ser_qpsk = [theoretical_ser("PSK", 4, snr) for snr in snr_db_range]
ser_8psk = [theoretical_ser("PSK", 8, snr) for snr in snr_db_range]
ser_16qam = [theoretical_ser("QAM", 16, snr) for snr in snr_db_range]
ser_64qam = [theoretical_ser("QAM", 64, snr) for snr in snr_db_range]
ser_2fsk = [theoretical_ser("FSK", 2, snr) for snr in snr_db_range]

# Plot SER curves
plt.semilogy(snr_db_range, ser_bpsk, linewidth=2, label="BPSK")
plt.semilogy(snr_db_range, ser_qpsk, linewidth=2, label="QPSK")
plt.semilogy(snr_db_range, ser_8psk, linewidth=2, label="8-PSK")
plt.semilogy(snr_db_range, ser_16qam, linewidth=2, label="16-QAM")
plt.semilogy(snr_db_range, ser_64qam, linewidth=2, label="64-QAM")
plt.semilogy(snr_db_range, ser_2fsk, linewidth=2, label="Binary FSK")

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Symbol Error Rate (SER)", fontsize=14)
plt.title("Theoretical Symbol Error Rate vs. SNR for Various Modulation Schemes", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 20)
plt.ylim(1e-6, 1)

plt.tight_layout()
plt.show()

# %%
# Spectral Efficiency Visualization
# ----------------------------------------------------------
#
# Let's visualize the spectral efficiency of different modulation schemes.

# Create a bar chart of spectral efficiency
plt.figure(figsize=(12, 8))

# Modulation schemes and their spectral efficiency (bits/s/Hz)
modulations = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "256-QAM"]
spectral_efficiency = [1, 2, 3, 4, 6, 8]  # bits/s/Hz

# Bar chart
plt.bar(modulations, spectral_efficiency, color=colors)

# Add value labels on top of the bars
for i, val in enumerate(spectral_efficiency):
    plt.text(i, val + 0.1, str(val), ha="center", fontsize=12)

plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.xlabel("Modulation Scheme", fontsize=14)
plt.ylabel("Spectral Efficiency (bits/s/Hz)", fontsize=14)
plt.title("Spectral Efficiency of Various Modulation Schemes", fontsize=16)

plt.tight_layout()
plt.show()

# %%
# Combined Performance Comparison
# --------------------------------------------------------
#
# Let's create a visualization that compares both SER and spectral efficiency.

# Create a scatter plot of SER vs. spectral efficiency
plt.figure(figsize=(12, 8))

# Modulation schemes
modulations = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM", "256-QAM"]
spectral_efficiency = [1, 2, 3, 4, 6, 8]  # bits/s/Hz

# SER at a specific SNR (e.g., SNR = 15 dB)
snr_fixed = 15  # dB
ser_values = []

# Calculate SER for each modulation scheme at the fixed SNR
ser_values.append(theoretical_ser("PSK", 2, snr_fixed))  # BPSK
ser_values.append(theoretical_ser("PSK", 4, snr_fixed))  # QPSK
ser_values.append(theoretical_ser("PSK", 8, snr_fixed))  # 8-PSK
ser_values.append(theoretical_ser("QAM", 16, snr_fixed))  # 16-QAM
ser_values.append(theoretical_ser("QAM", 64, snr_fixed))  # 64-QAM
ser_values.append(theoretical_ser("QAM", 256, snr_fixed))  # 256-QAM

# Scatter plot with different sizes based on SNR requirement
plt.scatter(spectral_efficiency, ser_values, c=colors[: len(modulations)], s=[100 * (1 + 2 * i) for i in range(len(modulations))], alpha=0.7, edgecolors="black")

# Add labels for each point
for i, (mod, x, y) in enumerate(zip(modulations, spectral_efficiency, ser_values)):
    plt.annotate(mod, (x, y), xytext=(10, 10), textcoords="offset points", fontsize=12, arrowprops=dict(arrowstyle="->", color="black"))

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("Spectral Efficiency (bits/s/Hz)", fontsize=14)
plt.ylabel("Symbol Error Rate (SER)", fontsize=14)
plt.title(f"Symbol Error Rate vs. Spectral Efficiency (SNR = {snr_fixed} dB)", fontsize=16)
plt.yscale("log")
plt.ylim(1e-6, 1)

plt.tight_layout()
plt.show()

# %%
# Conclusion
# -------------------
#
# In this visualization example, we have created various attractive and informative
# visualizations of digital modulation schemes using the Kaira library. These
# visualizations help in understanding the characteristics, performance, and trade-offs
# of different modulation techniques.
#
# The key insights from these visualizations are:
#
# 1. Different modulation schemes have varying constellation patterns, with higher-order
#    modulations (like 64-QAM) packing more bits per symbol at the cost of increased
#    sensitivity to noise.
#
# 2. Noise affects the received constellation by creating clouds of possible received
#    points around each constellation point, with the size of these clouds inversely
#    related to the SNR.
#
# 3. Decision regions show how the receiver makes decisions based on the received signal,
#    with higher-order modulations having more complex decision boundaries.
#
# 4. Gray coding in bit mapping ensures that adjacent constellation points differ by only
#    one bit, minimizing the bit error rate when symbol errors occur.
#
# 5. There is a fundamental trade-off between spectral efficiency and error performance,
#    with higher-order modulations providing better spectral efficiency at the cost of
#    decreased error performance.
#
# These visualizations serve as valuable tools for understanding and designing
# modulation schemes for various communication systems using the Kaira library.
