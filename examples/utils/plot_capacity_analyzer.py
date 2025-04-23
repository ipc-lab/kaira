"""
=========================================
Channel Capacity Analysis with Kaira
=========================================

This example demonstrates how to use the :class:`kaira.utils.CapacityAnalyzer` to analyze
the capacity of various modulation schemes and channel models.

Channel capacity is a fundamental concept in information theory that represents the
maximum rate at which information can be reliably transmitted over a communication channel.
It was first introduced by Claude Shannon :cite:`shannon1948mathematical`.

.. note::
   This example requires matplotlib for visualization and seaborn for enhanced styling.

   To run faster, set the FAST_MODE flag to True below.
"""

# %%
# Import necessary libraries
# --------------------------
# We'll start by importing the required modules.

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

from kaira.channels import AWGNChannel, RayleighFadingChannel, RicianFadingChannel
from kaira.modulations import BPSKModulator, PSKModulator, QAMModulator, QPSKModulator
from kaira.utils import CapacityAnalyzer

# User configuration for runtime performance
# -----------------------------------------
# Set to True for quick execution, False for detailed but slower analysis
FAST_MODE = True
VISUALIZATION_QUALITY = "low" if FAST_MODE else "high"  # 'low', 'medium', or 'high'

# Set the number of examples to run based on mode
if FAST_MODE:
    EXAMPLES_TO_RUN = [1, 3]  # Only run examples 1 and 3 in fast mode
    print("Running in FAST_MODE - only examples 1 and 3 will be executed with reduced computation")
else:
    EXAMPLES_TO_RUN = [1, 2, 3, 4]  # Run all examples in normal mode

# Configure sample sizes based on visualization quality
if VISUALIZATION_QUALITY == "low":
    NUM_SYMBOLS = 1000
    NUM_SNR_POINTS = 8
    NUM_MONTE_CARLO = 500
elif VISUALIZATION_QUALITY == "medium":
    NUM_SYMBOLS = 2000
    NUM_SNR_POINTS = 16
    NUM_MONTE_CARLO = 1000
else:  # high
    NUM_SYMBOLS = 5000
    NUM_SNR_POINTS = 31
    NUM_MONTE_CARLO = 5000

# Set seaborn style for prettier plots
sns.set(style="whitegrid", context="talk", palette="colorblind")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "figure.figsize": (12, 8),
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.linewidth": 1.5,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelweight": "bold",
        "axes.labelsize": 14,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.alpha": 0.3,
    }
)

# Create more vibrant custom colormaps for prettier visualizations
vibrant_colors = ["#4B0082", "#0000FF", "#00BFFF", "#00FF7F", "#FFFF00", "#FF7F00", "#FF0000"]
cmap_name = "capacity_spectrum"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, vibrant_colors)

# Additional colormaps for different visualization types
cool_warm = LinearSegmentedColormap.from_list("cool_warm", ["#00287F", "#7022BF", "#FF2970", "#FFBF00"])
forest_fire = LinearSegmentedColormap.from_list("forest_fire", ["#1E4620", "#54A24B", "#EACE3F", "#F57C00", "#B30000"])
ocean_depth = LinearSegmentedColormap.from_list("ocean_depth", ["#000033", "#001E6C", "#0057B8", "#009FFD", "#FFFFFF"])

# %%
# Initialize the Capacity Analyzer
# --------------------------------
# Create a capacity analyzer with parallel processing and fast mode for quicker execution.

# Start timing the execution
start_time = time.time()

analyzer = CapacityAnalyzer(num_processes=-1, fast_mode=True)  # Use all available CPU cores with fast approximations

# %%
# Define SNR range and create channel models
# ------------------------------------------
# We'll analyze capacity over an SNR range from -10 dB to 20 dB with granularity appropriate for our mode.

# Reduce the number of SNR points for faster execution
snr_db_range = np.linspace(-10, 20, NUM_SNR_POINTS)

# Create various channel models for comparison
awgn_channel = AWGNChannel(snr_db=10)  # SNR will be updated during analysis
rayleigh_channel = RayleighFadingChannel(coherence_time=10, snr_db=10)
rician_channel = RicianFadingChannel(k_factor=5, coherence_time=10, snr_db=10)

# Create modulation schemes of varying orders
bpsk = BPSKModulator()
qpsk = QPSKModulator()
psk8 = PSKModulator(order=8)
qam16 = QAMModulator(order=16)
qam64 = QAMModulator(order=64)

if 1 in EXAMPLES_TO_RUN:
    # %%
    # Example 1: Compare Modulation Schemes over AWGN
    # ----------------------------------------------
    # The Shannon-Hartley theorem :cite:`hartley1928transmission` gives the maximum
    # capacity of an AWGN channel, but different modulation schemes approach this
    # limit with varying efficiency.

    print(f"\nRunning Example 1: Modulation Schemes Comparison ({time.time() - start_time:.1f}s elapsed)")

    # Use only a subset of modulators in fast mode to reduce computation time
    if FAST_MODE:
        modulators = [bpsk, qpsk, qam16]
        labels = ["BPSK", "QPSK", "16-QAM"]
    else:
        modulators = [bpsk, qpsk, psk8, qam16, qam64]
        labels = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]

    # Create a custom color palette for the different modulation schemes
    modulation_colors = sns.color_palette("viridis", len(modulators))

    snr, capacities, fig1 = analyzer.compare_modulation_schemes(modulators, awgn_channel, snr_db_range, labels=labels, num_symbols=NUM_SYMBOLS, estimation_method="histogram")  # Explicitly use histogram method as it's faster than KNN

    # Enhance the plot with custom styling
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Shannon capacity limit with special styling
    shannon_capacity = analyzer.awgn_capacity(torch.tensor(snr_db_range)).cpu().numpy()
    ax1.plot(snr_db_range, shannon_capacity, "k--", linewidth=3, label="Shannon Limit")

    # Plot each modulation scheme with custom colors and markers
    for i, (label, capacity) in enumerate(capacities.items()):
        if isinstance(capacity, torch.Tensor):
            capacity = capacity.cpu().numpy()
        ax1.plot(snr_db_range, capacity, "-", color=modulation_colors[i], linewidth=2.5, marker="o", markersize=8, label=label)

    # Only add inset in non-fast mode due to computational requirements
    if not FAST_MODE:
        # Add an inset axes for zoomed view at high SNR region
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        axins = inset_axes(ax1, width="40%", height="30%", loc="lower right", bbox_to_anchor=(0.95, 0.15, 0.5, 0.5), bbox_transform=ax1.transAxes)

        # Plot the same data in the inset
        axins.plot(snr_db_range, shannon_capacity, "k--", linewidth=2, label="Shannon Limit")
        for i, (label, capacity) in enumerate(capacities.items()):
            if isinstance(capacity, torch.Tensor):
                capacity = capacity.cpu().numpy()
            axins.plot(snr_db_range, capacity, "-", color=modulation_colors[i], linewidth=2, marker="o", markersize=4)

        # Set the limits for the inset
        axins.set_xlim(10, 20)  # Zoom to high SNR region
        axins.set_ylim(3, 7)  # Zoom to the capacity range of interest
        # Apply zoomed area indicator
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # Add a text annotation explaining the inset
        ax1.text(10, 2, "Zoom in on high SNR region\nshowing modulation capacity limits", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"))

    # Add grid, labels, and legend with styling
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlabel("SNR (dB)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Capacity (bits/channel use)", fontsize=14, fontweight="bold")
    ax1.set_title("Modulation Capacity Comparison over AWGN Channel", fontsize=16, fontweight="bold")

    # Create a more visually appealing legend with two columns
    legend = ax1.legend(loc="upper left", fontsize=12, framealpha=0.9, fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")

    # Add annotations for key points
    if not FAST_MODE:
        ax1.annotate("Low SNR Region:\nBPSK optimal", xy=(-5, 0.3), xytext=(-9, 0.7), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax1.annotate("High SNR Region:\nHigher-order modulations approach capacity", xy=(15, 5), xytext=(5, 5.5), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Enhance axes appearance
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Add colorbar to show modulation order
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=len(modulators)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation="vertical", pad=0.01, shrink=0.8)
    cbar.set_label("Modulation Order", fontsize=12, fontweight="bold")
    cbar.set_ticks([1 + i * (len(modulators) - 1) / (len(modulators) - 1) for i in range(len(modulators))])
    cbar.set_ticklabels([mod.__class__.__name__.replace("Modulator", "") for mod in modulators])

    plt.tight_layout()

    # %%
    # The plot demonstrates how different modulation schemes approach the Shannon capacity limit.
    # The enhanced visualization includes:
    #
    # * A colored gradient representing modulation complexity
    # * An inset subplot zooming on the high SNR region (in non-fast mode)
    # * Annotations highlighting key insights (in non-fast mode)
    # * Custom styling for better readability
    #
    # As expected, higher-order modulations achieve higher capacities at high SNR,
    # while simpler schemes like BPSK are more robust at low SNR :cite:`proakis2007digital`.

if 2 in EXAMPLES_TO_RUN:
    # %%
    # Example 2: Compare Channels with 3D Visualization
    # ------------------------------------------------
    # Next, we'll create a 3D visualization to compare how different channel models
    # affect the capacity across multiple modulation schemes.

    print(f"\nRunning Example 2: 3D Channel Comparison ({time.time() - start_time:.1f}s elapsed)")

    # Create a figure for 3D plotting
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(111, projection="3d")

    # Define channel models and labels
    channels = [awgn_channel, rayleigh_channel, rician_channel]
    channel_labels = ["AWGN", "Rayleigh Fading", "Rician Fading (K=5)"]

    # Define a subset of modulation schemes - use fewer in fast mode
    if FAST_MODE:
        mod_subset = [bpsk, qpsk]
        mod_labels = ["BPSK", "QPSK"]
    else:
        mod_subset = [bpsk, qpsk, qam16]
        mod_labels = ["BPSK", "QPSK", "16-QAM"]

    # Calculate capacity for each combination
    z_data = np.zeros((len(mod_subset), len(channels), len(snr_db_range)))

    for i, modulator in enumerate(mod_subset):
        for j, channel in enumerate(channels):
            _, capacity = analyzer.modulation_capacity(modulator, channel, snr_db_range, num_symbols=NUM_SYMBOLS, estimation_method="histogram")  # Explicitly use histogram method
            z_data[i, j, :] = capacity.cpu().numpy()

    # Create a mesh grid for 3D surface plotting
    mod_indices, channel_indices = np.meshgrid(np.arange(len(mod_subset)), np.arange(len(channels)))

    # Plot 3D surfaces for each SNR point
    # We'll select representative SNR points - fewer in fast mode
    if FAST_MODE:
        snr_indices = [0, len(snr_db_range) // 2, -1]  # Beginning, middle, end
    else:
        snr_indices = [0, len(snr_db_range) // 3, 2 * len(snr_db_range) // 3, -1]  # More granular

    snr_labels = [f"{snr_db_range[i]:.0f} dB" for i in snr_indices]
    cmap = plt.cm.viridis

    for k, snr_idx in enumerate(snr_indices):
        offset = k * 2  # Offset each surface for visibility
        surf = ax2.plot_surface(mod_indices, channel_indices + offset, z_data[:, :, snr_idx].T, cmap=cmap, alpha=0.8, label=snr_labels[k])
        # Add wire frame for better visibility
        ax2.plot_wireframe(mod_indices, channel_indices + offset, z_data[:, :, snr_idx].T, color="black", alpha=0.1, linewidth=0.5)
        # Add text annotation for SNR value
        ax2.text(len(mod_subset) - 1, offset, np.max(z_data[:, :, snr_idx]) + 0.2, snr_labels[k], fontsize=12, fontweight="bold")

    # Customize the 3D plot
    ax2.set_xlabel("Modulation Scheme", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Channel Type", fontsize=14, fontweight="bold")
    ax2.set_zlabel("Capacity (bits/channel use)", fontsize=14, fontweight="bold")
    ax2.set_title("3D Comparison of Capacity across Channels and Modulations", fontsize=16, fontweight="bold")

    # Set custom tick labels
    ax2.set_xticks(np.arange(len(mod_subset)))
    ax2.set_xticklabels(mod_labels)
    ax2.set_yticks(np.arange(len(channels)) + 1.5)  # Center ticks in offset groups
    ax2.set_yticklabels(channel_labels)

    # Create custom proxy artists for the legend
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color=cmap(0.2), lw=4, label=snr_labels[0]), Line2D([0], [0], color=cmap(0.4), lw=4, label=snr_labels[1]), Line2D([0], [0], color=cmap(0.7), lw=4, label=snr_labels[-1])]

    # Add the legend
    ax2.legend(handles=legend_elements, loc="upper left", title="SNR Values", fontsize=10, title_fontsize=12)

    # Set the viewing angle for best visibility
    ax2.view_init(elev=25, azim=-35)

    # Add annotations explaining key insights
    ax2.text2D(
        0.05,
        0.05,
        "Observations:\n" + "1. AWGN channels achieve highest capacity\n" + "2. Rayleigh fading has lowest capacity due to absence of LOS\n" + "3. Higher order modulations benefit more from better channels\n" + "4. Capacity gap widens at higher SNR values",
        transform=ax2.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()

    # %%
    # This 3D visualization offers a comprehensive view of how both channel type and modulation
    # scheme affect the achievable capacity at different SNR levels. Rayleigh fading represents
    # scenarios with no line-of-sight path, resulting in lower capacity, while Rician fading
    # (with K=5) models a stronger direct path component, achieving capacity closer to the
    # AWGN case :cite:`rappaport2024wireless`.

if 3 in EXAMPLES_TO_RUN:
    # %%
    # Example 3: MIMO Capacity Analysis with Heatmap
    # ----------------------------------------------
    # Multiple-Input Multiple-Output (MIMO) systems can significantly increase capacity
    # by exploiting spatial multiplexing :cite:`telatar1999capacity`. Let's compare
    # different MIMO configurations with a heatmap visualization.

    print(f"\nRunning Example 3: MIMO Capacity Analysis ({time.time() - start_time:.1f}s elapsed)")

    # Use smaller antenna arrays in fast mode
    if FAST_MODE:
        tx_antennas = [1, 2, 4]
        rx_antennas = [1, 2, 4]
    else:
        tx_antennas = [1, 2, 4, 8]
        rx_antennas = [1, 2, 4, 8]

    mimo_capacities = np.zeros((len(tx_antennas), len(rx_antennas), len(snr_db_range)))

    # Calculate MIMO capacity for different antenna configurations
    for i, tx in enumerate(tx_antennas):
        for j, rx in enumerate(rx_antennas):
            mimo_capacities[i, j] = analyzer.mimo_capacity(snr_db_range, tx_antennas=tx, rx_antennas=rx, channel_knowledge="perfect", num_realizations=NUM_MONTE_CARLO // 2).cpu().numpy()  # Use fewer simulations for MIMO

    # Create a figure with subplots - one heatmap for each SNR value
    fig3 = plt.figure(figsize=(16, 12))
    fig3.suptitle("MIMO Capacity (bits/channel use) vs. Antenna Configuration", fontsize=20, fontweight="bold", y=0.98)

    # Select representative SNR points - fewer in fast mode
    if FAST_MODE:
        snr_indices = [0, len(snr_db_range) // 2, -1]  # Beginning, middle, end
    else:
        snr_indices = [0, len(snr_db_range) // 3, 2 * len(snr_db_range) // 3, -1]

    snr_values = [snr_db_range[i] for i in snr_indices]
    num_plots = len(snr_indices)

    # Calculate rows and columns for the subplot grid
    grid_rows = int(np.ceil(num_plots / 2))
    grid_cols = min(2, num_plots)

    # Create a grid of heatmaps for different SNR values
    for k, (snr_idx, snr_val) in enumerate(zip(snr_indices, snr_values)):
        ax = fig3.add_subplot(grid_rows, grid_cols, k + 1)

        # Extract capacity data for this SNR
        capacity_data = mimo_capacities[:, :, snr_idx]

        # Create heatmap
        im = ax.imshow(capacity_data, cmap="plasma", interpolation="nearest")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Capacity (bits/channel use)", fontsize=12)

        # Configure axes
        ax.set_xticks(np.arange(len(rx_antennas)))
        ax.set_yticks(np.arange(len(tx_antennas)))
        ax.set_xticklabels(rx_antennas)
        ax.set_yticklabels(tx_antennas)
        ax.set_xlabel("Receive Antennas", fontsize=12)
        ax.set_ylabel("Transmit Antennas", fontsize=12)
        ax.set_title(f"SNR = {snr_val:.1f} dB", fontsize=14)

        # Add text annotations with capacity values
        for i in range(len(tx_antennas)):
            for j in range(len(rx_antennas)):
                text = ax.text(j, i, f"{capacity_data[i, j]:.1f}", ha="center", va="center", color="w", fontweight="bold")

        # Highlight symmetric configurations (nTx = nRx)
        for i in range(len(tx_antennas)):
            if i < len(rx_antennas):
                rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2)
                ax.add_patch(rect)

    # Add an explanatory text box
    textbox = fig3.add_axes([0.1, 0.01, 0.8, 0.05])
    textbox.axis("off")
    textbox.text(
        0.5,
        0.5,
        "Key observations:\n" + "1. Capacity scales approximately linearly with min(nTx, nRx) at high SNR\n" + "2. Adding more transmit antennas beyond the number of receive antennas offers diminishing returns\n" + "3. The capacity gain from MIMO is more pronounced at higher SNR values",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8),
    )

    # Adjust subplot spacing instead of using tight_layout with rect
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3)

    # %%
    # The heatmap visualization shows how MIMO capacity scales with the number of antennas.
    # At high SNR, capacity grows approximately linearly with min(nTx, nRx), demonstrating
    # the significant multiplexing gain offered by MIMO systems :cite:`goldsmith2003capacity`.
    # The visualization also highlights the diminishing returns when adding transmit antennas
    # beyond the number of receive antennas, which is important for system design considerations.

if 4 in EXAMPLES_TO_RUN:
    # %%
    # Example 4: Capacity CDF for Fading Channels
    # -------------------------------------------
    # For fading channels, the cumulative distribution function (CDF) of capacity
    # helps characterize the probability of achieving a certain rate :cite:`goldsmith2005wireless`.

    print(f"\nRunning Example 4: Capacity CDF Analysis ({time.time() - start_time:.1f}s elapsed)")

    # Calculate capacity CDFs for different channel types at multiple SNR values
    # Use fewer SNR points in fast mode
    if FAST_MODE:
        snr_points = [0, 10, 20]
    else:
        snr_points = [0, 5, 10, 15, 20]

    channel_types = [rayleigh_channel, rician_channel]
    channel_names = ["Rayleigh Fading", "Rician Fading (K=5)"]
    channel_colors = ["#3498db", "#e74c3c"]  # Blue for Rayleigh, Red for Rician

    fig4, ax4 = plt.subplots(figsize=(12, 8))

    # Plot CDFs for each channel type and SNR value
    legend_elements = []

    for i, (channel, name, color) in enumerate(zip(channel_types, channel_names, channel_colors)):
        for j, snr in enumerate(snr_points):
            # Calculate CDF with reduced number of realizations for faster execution
            capacity_values, cdf = analyzer.capacity_cdf(channel, snr_db=snr, num_realizations=NUM_MONTE_CARLO)

            # Plot with varying color intensity based on SNR
            alpha = 0.4 + 0.6 * (j / len(snr_points))
            (line,) = ax4.plot(capacity_values.cpu().numpy(), cdf.cpu().numpy(), color=color, alpha=alpha, linewidth=2.5)

            # Add to legend only for the first channel (to avoid cluttering)
            if i == 0:
                legend_elements.append(line)

        # Add channel type to legend
        from matplotlib.patches import Patch

        legend_elements.append(Patch(facecolor=color, label=name))

    # Add median lines for better visualization
    for i, (channel, name, color) in enumerate(zip(channel_types, channel_names, channel_colors)):
        for j, snr in enumerate(snr_points):
            capacity_values, cdf = analyzer.capacity_cdf(channel, snr_db=snr, num_realizations=NUM_MONTE_CARLO)

            # Find median capacity (where CDF = 0.5)
            idx = np.argmin(np.abs(cdf.cpu().numpy() - 0.5))
            median_capacity = capacity_values[idx].item()

            # Add vertical line at median
            ax4.axvline(x=median_capacity, color=color, linestyle="--", alpha=0.4 + 0.6 * (j / len(snr_points)), linewidth=1)

            # Annotate median for highest SNR
            if j == len(snr_points) - 1:
                ax4.text(median_capacity + 0.2, 0.5, f"Median\n{median_capacity:.2f}", color=color, fontsize=10, ha="left", va="center")

    # Customize the plot
    ax4.set_xlabel("Capacity (bits/channel use)", fontsize=14, fontweight="bold")
    ax4.set_ylabel("Probability (Capacity ≤ x)", fontsize=14, fontweight="bold")
    ax4.set_title("Capacity CDF for Fading Channels at Different SNR Values", fontsize=16, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--")

    # Add SNR values to legend
    snr_patches = [Line2D([0], [0], color="gray", alpha=0.4 + 0.6 * (j / len(snr_points)), linewidth=2.5, label=f"SNR = {snr} dB") for j, snr in enumerate(snr_points)]

    # Create a nice legend with two columns
    ax4.legend(handles=snr_patches + legend_elements, loc="lower right", fontsize=11, framealpha=0.9, fancybox=True, shadow=True, ncol=2)

    # Add annotations
    ax4.annotate("Steeper CDF slopes\nindicate less variability", xy=(3.5, 0.2), xytext=(4.5, 0.2), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax4.annotate("Rician fading has higher\ncapacity than Rayleigh\ndue to LOS component", xy=(6, 0.8), xytext=(4, 0.8), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Add outage capacity indicators
    outage_prob = 0.1  # 10% outage probability
    for i, (channel, name, color) in enumerate(zip(channel_types, channel_names, channel_colors)):
        capacity_values, cdf = analyzer.capacity_cdf(channel, snr_db=snr_points[-1], num_realizations=NUM_MONTE_CARLO)

        # Find capacity at outage probability
        idx = np.argmin(np.abs(cdf.cpu().numpy() - outage_prob))
        outage_capacity = capacity_values[idx].item()

        # Plot outage capacity point
        ax4.plot(outage_capacity, outage_prob, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=1.5)

        # Add annotation for outage capacity
        ax4.annotate(
            f"{name}\nOutage Capacity\n{outage_capacity:.2f} at 10%",
            xy=(outage_capacity, outage_prob),
            xytext=(outage_capacity + (-3 if i == 0 else 1), outage_prob - 0.1),
            arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    # Add horizontal line at outage probability
    ax4.axhline(y=outage_prob, color="black", linestyle=":", alpha=0.5)
    ax4.text(0.5, outage_prob + 0.02, "10% Outage Probability", fontsize=10, va="bottom")

    plt.tight_layout()

    # %%
    # This enhanced CDF visualization provides a comprehensive view of capacity variability
    # across different fading channels and SNR values. The plot highlights:
    #
    # * The statistical nature of capacity in fading channels
    # * How Rician fading offers higher capacity than Rayleigh due to the line-of-sight component
    # * The concept of outage capacity, which is critical for reliable communication system design
    # * How the capacity distribution changes with SNR
    #
    # The steep slope indicates significant capacity variations due to channel fluctuations,
    # which affects reliability for fixed-rate transmission schemes.

# %%
# Print total execution time
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.1f} seconds")
