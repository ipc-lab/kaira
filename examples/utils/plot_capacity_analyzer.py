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

# NEW: More granular performance control
PERFORMANCE_MODE = "ultra_fast"  # Options: "ultra_fast", "fast", "balanced", "detailed"

# Configure visualization and computation based on performance mode
if PERFORMANCE_MODE == "ultra_fast":
    VISUALIZATION_QUALITY = "minimal"
    EXAMPLES_TO_RUN = [1]  # Run only the essential example
    NUM_SYMBOLS = 100
    NUM_SNR_POINTS = 3
    NUM_MONTE_CARLO = 50
    PARALLEL_PROCESSES = 1  # Limit parallelism to reduce overhead on smaller problems
elif PERFORMANCE_MODE == "fast":
    VISUALIZATION_QUALITY = "low"
    EXAMPLES_TO_RUN = [1, 3]  # Only run examples 1 and 3
    NUM_SYMBOLS = 1000
    NUM_SNR_POINTS = 8
    NUM_MONTE_CARLO = 500
    PARALLEL_PROCESSES = -1  # Use all available cores
elif PERFORMANCE_MODE == "balanced":
    VISUALIZATION_QUALITY = "medium"
    EXAMPLES_TO_RUN = [1, 2, 3, 4]  # Run examples 1, 2, and 3
    NUM_SYMBOLS = 1000
    NUM_SNR_POINTS = 8
    NUM_MONTE_CARLO = 1000
    PARALLEL_PROCESSES = -1  # Use all available cores
else:  # detailed
    VISUALIZATION_QUALITY = "high"
    EXAMPLES_TO_RUN = [1, 2, 3, 4]  # Run all examples
    NUM_SYMBOLS = 5000
    NUM_SNR_POINTS = 31
    NUM_MONTE_CARLO = 5000
    PARALLEL_PROCESSES = -1  # Use all available cores

# For backward compatibility
FAST_MODE = PERFORMANCE_MODE in ["ultra_fast", "fast"]

# Print performance settings
print(f"Running in {PERFORMANCE_MODE.upper()} mode")
print(f"Examples to run: {EXAMPLES_TO_RUN}")
print(f"Visualization quality: {VISUALIZATION_QUALITY}")
print(f"Using {NUM_SYMBOLS} symbols, {NUM_SNR_POINTS} SNR points, {NUM_MONTE_CARLO} Monte Carlo trials")

# Use PyTorch's accelerated operations and optimize memory usage
if torch.cuda.is_available() and PERFORMANCE_MODE in ["detailed"]:
    device = torch.device("cuda")
    print("Using GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU execution")

# Use just-in-time compilation for core functions if supported
USE_JIT = PERFORMANCE_MODE != "ultra_fast"  # JIT compilation has overhead for very small workloads

if USE_JIT and hasattr(torch, "jit"):
    print("Using PyTorch JIT compilation for acceleration")

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
# Create a capacity analyzer with optimal performance settings based on mode

# Start timing the execution
start_time = time.time()

# %%
# Define SNR range and create channel models
# ------------------------------------------
# We'll analyze capacity over an SNR range from -10 dB to 20 dB with granularity appropriate for our mode.

# Define the SNR range first
snr_db_range = np.linspace(-10, 20, NUM_SNR_POINTS)

# Configure analyzer with optimal settings for the selected performance mode
analyzer = CapacityAnalyzer(num_processes=PARALLEL_PROCESSES, fast_mode=FAST_MODE, device=device)  # Parallel processing based on mode  # Use fast approximations  # Use GPU when available and appropriate

# Set default tensor type based on performance mode to control precision
if PERFORMANCE_MODE in ["ultra_fast", "fast", "balanced"]:
    # Use float32 for faster computation in fast modes
    torch.set_default_dtype(torch.float32)
    print("Using 32-bit float precision for faster computation")
else:
    # Use float64 for more accurate results in detailed modes
    torch.set_default_dtype(torch.float64)
    print("Using 64-bit float precision for higher accuracy")

# Pre-compute and cache Shannon capacity limits to reuse across examples
shannon_capacity = analyzer.awgn_capacity(torch.tensor(snr_db_range, device=device))

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

# Pre-allocate tensors and arrays for results to avoid reallocations
# This improves memory usage and reduces garbage collection overhead
if 1 in EXAMPLES_TO_RUN:
    # Pre-allocate for example 1
    if FAST_MODE:
        ex1_modulators = [bpsk, qpsk, qam16]
        ex1_modulator_count = 3
    else:
        ex1_modulators = [bpsk, qpsk, psk8, qam16, qam64]
        ex1_modulator_count = 5

    # Pre-allocate results dictionary
    ex1_capacities: dict[str, torch.Tensor] = {}

if 1 in EXAMPLES_TO_RUN:
    # %%
    # Example 1: Compare Modulation Schemes over AWGN
    # ----------------------------------------------
    # The Shannon-Hartley theorem :cite:`hartley1928transmission` gives the maximum
    # capacity of an AWGN channel, but different modulation schemes approach this
    # limit with varying efficiency.

    print(f"\nRunning Example 1: Modulation Schemes Comparison ({time.time() - start_time:.1f}s elapsed)")
    example_start_time = time.time()

    # Use only a subset of modulators in fast mode to reduce computation time
    if FAST_MODE:
        modulators = [bpsk, qpsk, qam16]
        labels = ["BPSK", "QPSK", "16-QAM"]
    else:
        modulators = [bpsk, qpsk, psk8, qam16, qam64]
        labels = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]

    # Create a custom color palette for the different modulation schemes
    modulation_colors = sns.color_palette("viridis", len(modulators))

    # Convert SNR to tensor once for efficiency
    snr_tensor = torch.tensor(snr_db_range, device=device)

    # Calculate Shannon capacity using vectorized operation
    shannon_capacity_tensor = analyzer.awgn_capacity(snr_tensor)
    shannon_capacity_numpy = shannon_capacity_tensor.cpu().numpy()

    # Pre-allocate dictionary for capacities
    capacities = {}

    # Use more efficient batch computation when possible
    if hasattr(analyzer, "compare_modulation_schemes_batch"):
        # Batch computation is more efficient
        snr, capacities, _ = analyzer.compare_modulation_schemes_batch(modulators, awgn_channel, snr_tensor, labels=labels, num_symbols=NUM_SYMBOLS, estimation_method="histogram")  # Histogram method is faster than KNN
    else:
        # Regular computation if batch method not available
        for i, (modulator, label) in enumerate(zip(modulators, labels)):
            print(f"  Computing capacity for {label}...")
            _, capacity = analyzer.modulation_capacity(modulator, awgn_channel, snr_tensor, num_symbols=NUM_SYMBOLS, estimation_method="histogram")
            capacities[label] = capacity

    # Create the visualization (with performance optimizations)
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Shannon capacity limit with special styling
    ax1.plot(snr_db_range, shannon_capacity_numpy, "k--", linewidth=3, label="Shannon Limit")

    # Plot each modulation scheme with custom colors and markers
    # Optimize by plotting fewer markers in fast mode
    marker_step = 2 if PERFORMANCE_MODE == "ultra_fast" else 1
    for i, (label, capacity) in enumerate(capacities.items()):
        if isinstance(capacity, torch.Tensor):
            capacity = capacity.cpu().numpy()

        # Plot with optimized markers
        ax1.plot(
            snr_db_range, capacity, "-", color=modulation_colors[i], linewidth=2.5, marker="o" if i % marker_step == 0 else None, markersize=8 if PERFORMANCE_MODE != "ultra_fast" else 6, markevery=max(1, len(snr_db_range) // 8), label=label  # Use markers selectively  # Reduce number of markers
        )

    # Only add inset in non-fast mode due to computational requirements
    if PERFORMANCE_MODE not in ["ultra_fast", "fast"]:
        # Add an inset axes for zoomed view at high SNR region
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

        axins = inset_axes(ax1, width="40%", height="30%", loc="lower right", bbox_to_anchor=(0.95, 0.15, 0.5, 0.5), bbox_transform=ax1.transAxes)

        # Plot the same data in the inset
        axins.plot(snr_db_range, shannon_capacity_numpy, "k--", linewidth=2, label="Shannon Limit")
        for i, (label, capacity) in enumerate(capacities.items()):
            if isinstance(capacity, torch.Tensor):
                capacity = capacity.cpu().numpy()
            axins.plot(snr_db_range, capacity, "-", color=modulation_colors[i], linewidth=2, marker="o", markersize=4, markevery=2)

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

    # Add annotations for key points (only in detailed modes)
    if PERFORMANCE_MODE not in ["ultra_fast", "fast"]:
        ax1.annotate("Low SNR Region:\nBPSK optimal", xy=(-5, 0.3), xytext=(-9, 0.7), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax1.annotate("High SNR Region:\nHigher-order modulations approach capacity", xy=(15, 5), xytext=(5, 5.5), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Add colorbar only if not in ultra_fast mode (reduces render time)
    if PERFORMANCE_MODE != "ultra_fast":
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

    # Use subplots_adjust instead of tight_layout to avoid warnings
    fig1.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.12)

    # Report timing for this example
    example_time = time.time() - example_start_time
    print(f"  Example 1 completed in {example_time:.1f} seconds")

if 2 in EXAMPLES_TO_RUN:
    # %%
    # Example 2: Compare Channels with 3D Visualization
    # ------------------------------------------------
    # Next, we'll create a 3D visualization to compare how different channel models
    # affect the capacity across multiple modulation schemes.

    print(f"\nRunning Example 2: 3D Channel Comparison ({time.time() - start_time:.1f}s elapsed)")

    # Create a figure for 3D plotting
    # Increase figure size to allow more space for labels
    fig2 = plt.figure(figsize=(18, 12))
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

    # Skip tight_layout for 3D plot and use alternative spacing approach
    # This avoids the "left and right margins cannot be made large enough" warning
    fig2.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

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
    example_start_time = time.time()

    # Use smaller antenna arrays in fast mode
    if FAST_MODE:
        tx_antennas = [1, 2, 4]
        rx_antennas = [1, 2, 4]
    else:
        tx_antennas = [1, 2, 4, 8]
        rx_antennas = [1, 2, 4, 8]

    # Pre-allocate array for results
    mimo_capacities = np.zeros((len(tx_antennas), len(rx_antennas), len(snr_db_range)))

    # For MIMO analysis, convert SNR to tensor once
    snr_tensor = torch.tensor(snr_db_range, device=device)

    # Calculate MIMO capacity for different antenna configurations
    # Use vectorized calculation if available for better performance
    if hasattr(analyzer, "mimo_capacity_batch"):
        print("  Using batch MIMO capacity calculation for improved performance")
        # Prepare configurations for batch processing
        configurations = []
        for i, tx in enumerate(tx_antennas):
            for j, rx in enumerate(rx_antennas):
                configurations.append((tx, rx))

        # Batch calculate all configurations at once
        batch_results = analyzer.mimo_capacity_batch(snr_tensor, configurations, channel_knowledge="perfect", num_realizations=NUM_MONTE_CARLO // 2)

        # Reshape results back into our grid
        for idx, (i, j) in enumerate([(i, j) for i in range(len(tx_antennas)) for j in range(len(rx_antennas))]):
            mimo_capacities[i, j] = batch_results[idx].cpu().numpy()
    else:
        # Traditional loop-based calculation
        print("  Using individual MIMO capacity calculations")
        # Use a more efficient nested loop structure
        for i, tx in enumerate(tx_antennas):
            for j, rx in enumerate(rx_antennas):
                print(f"  Computing MIMO capacity for {tx}x{rx} configuration...")
                mimo_capacities[i, j] = analyzer.mimo_capacity(snr_tensor, tx_antennas=tx, rx_antennas=rx, channel_knowledge="perfect", num_realizations=NUM_MONTE_CARLO // 2).cpu().numpy()  # Use fewer simulations for MIMO

    # Create a figure with subplots - one heatmap for each SNR value
    # Use a more efficient figure size based on the performance mode
    if PERFORMANCE_MODE in ["ultra_fast", "fast"]:
        fig_size = (12, 8)
    else:
        fig_size = (16, 12)

    fig3 = plt.figure(figsize=fig_size)
    fig3.suptitle("MIMO Capacity (bits/channel use) vs. Antenna Configuration", fontsize=20, fontweight="bold", y=0.98)

    # Select representative SNR points - fewer in fast mode
    if PERFORMANCE_MODE == "ultra_fast":
        snr_indices = [0, -1]  # Just beginning and end
    elif PERFORMANCE_MODE == "fast":
        snr_indices = [0, len(snr_db_range) // 2, -1]  # Beginning, middle, end
    else:
        snr_indices = [0, len(snr_db_range) // 3, 2 * len(snr_db_range) // 3, -1]

    snr_values = [snr_db_range[i] for i in snr_indices]
    num_plots = len(snr_indices)

    # Calculate optimal rows and columns for the subplot grid
    if num_plots <= 2:
        grid_rows, grid_cols = 1, num_plots
    else:
        grid_rows = int(np.ceil(num_plots / 2))
        grid_cols = min(2, num_plots)

    # Create a grid of heatmaps for different SNR values
    for k, (snr_idx, snr_val) in enumerate(zip(snr_indices, snr_values)):
        ax = fig3.add_subplot(grid_rows, grid_cols, k + 1)

        # Extract capacity data for this SNR
        capacity_data = mimo_capacities[:, :, snr_idx]

        # Create heatmap with optimized settings
        im = ax.imshow(capacity_data, cmap="plasma", interpolation="nearest")

        # Add colorbar (simplified in ultra_fast mode)
        if PERFORMANCE_MODE == "ultra_fast":
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Capacity", fontsize=10)
        else:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Capacity (bits/channel use)", fontsize=12)

        # Configure axes - simplified in ultra_fast mode
        ax.set_xticks(np.arange(len(rx_antennas)))
        ax.set_yticks(np.arange(len(tx_antennas)))
        ax.set_xticklabels(rx_antennas)
        ax.set_yticklabels(tx_antennas)

        # Adjust font sizes based on performance mode
        label_fontsize = 10 if PERFORMANCE_MODE == "ultra_fast" else 12
        title_fontsize = 12 if PERFORMANCE_MODE == "ultra_fast" else 14

        ax.set_xlabel("Receive Antennas", fontsize=label_fontsize)
        ax.set_ylabel("Transmit Antennas", fontsize=label_fontsize)
        ax.set_title(f"SNR = {snr_val:.1f} dB", fontsize=title_fontsize)

        # Add text annotations with capacity values - skip in ultra_fast mode
        if PERFORMANCE_MODE != "ultra_fast":
            for i in range(len(tx_antennas)):
                for j in range(len(rx_antennas)):
                    text = ax.text(j, i, f"{capacity_data[i, j]:.1f}", ha="center", va="center", color="w", fontweight="bold")

        # Highlight symmetric configurations (nTx = nRx) - skip in ultra_fast mode
        if PERFORMANCE_MODE != "ultra_fast":
            for i in range(len(tx_antennas)):
                if i < len(rx_antennas):
                    rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2)
                    ax.add_patch(rect)

    # Add an explanatory text box - simplified in ultra_fast mode
    if PERFORMANCE_MODE == "ultra_fast":
        plt.figtext(0.5, 0.01, "Key observation: Capacity scales with min(nTx, nRx)", ha="center", fontsize=10)
    else:
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

    # Replace tight_layout with explicit subplots_adjust to avoid warnings
    # Apply spacing adjustments directly instead of using tight_layout
    fig3.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.15, wspace=0.25, hspace=0.30)

    # Report timing for this example
    example_time = time.time() - example_start_time
    print(f"  Example 3 completed in {example_time:.1f} seconds")

if 4 in EXAMPLES_TO_RUN:
    # %%
    # Example 4: Capacity CDF for Fading Channels
    # -------------------------------------------
    # For fading channels, the cumulative distribution function (CDF) of capacity
    # helps characterize the probability of achieving a certain rate :cite:`goldsmith2005wireless`.

    print(f"\nRunning Example 4: Capacity CDF Analysis ({time.time() - start_time:.1f}s elapsed)")
    example_start_time = time.time()

    # Calculate capacity CDFs for different channel types at multiple SNR values
    # Use fewer SNR points in fast mode
    if PERFORMANCE_MODE == "ultra_fast":
        snr_points = [0, 20]  # Just low and high
    elif PERFORMANCE_MODE == "fast":
        snr_points = [0, 10, 20]  # Low, medium, high
    else:
        snr_points = [0, 5, 10, 15, 20]  # Full range

    channel_types = [rayleigh_channel, rician_channel]
    channel_names = ["Rayleigh Fading", "Rician Fading (K=5)"]
    channel_colors = ["#3498db", "#e74c3c"]  # Blue for Rayleigh, Red for Rician

    # Pre-allocate storage arrays for better memory efficiency
    capacity_values_dict: dict[str, dict[float, torch.Tensor]] = {}
    cdf_dict: dict[str, dict[float, torch.Tensor]] = {}

    # Calculate all CDFs first with vectorized operations where possible
    print("  Calculating capacity CDFs...")
    for i, (channel, name) in enumerate(zip(channel_types, channel_names)):
        capacity_values_dict[name] = {}
        cdf_dict[name] = {}

        # Convert SNR points to tensor for vectorized calculations
        if hasattr(analyzer, "capacity_cdf_batch"):
            # Use batch processing if available
            print(f"  Using batch CDF calculation for {name}")
            snr_tensor = torch.tensor(snr_points, device=device)
            batch_values, batch_cdfs = analyzer.capacity_cdf_batch(channel, snr_db=snr_tensor, num_realizations=NUM_MONTE_CARLO)
            # Store results
            for j, snr in enumerate(snr_points):
                capacity_values_dict[name][snr] = batch_values[j]
                cdf_dict[name][snr] = batch_cdfs[j]
        else:
            # Individual calculations
            for j, snr in enumerate(snr_points):
                print(f"  Computing CDF for {name} at SNR = {snr} dB")
                values, cdf = analyzer.capacity_cdf(channel, snr_db=snr, num_realizations=NUM_MONTE_CARLO)
                capacity_values_dict[name][snr] = values
                cdf_dict[name][snr] = cdf

    # Create visualization with adaptive quality
    fig_size = (10, 6) if PERFORMANCE_MODE == "ultra_fast" else (12, 8)
    fig4, ax4 = plt.subplots(figsize=fig_size)

    # Pre-calculate median capacities to avoid repeated computation
    median_capacities = {}
    outage_capacities = {}
    outage_prob = 0.1  # 10% outage probability

    # Calculate median and outage capacities from the CDFs
    for name in channel_names:
        outage_capacity_values = []
        for snr in snr_points:
            # Get pre-computed values
            capacity_values = capacity_values_dict[name][snr]
            cdf_values = cdf_dict[name][snr]

            # Find median capacity (CDF = 0.5)
            median_idx = torch.argmin(torch.abs(cdf_values - 0.5))
            median_capacities[(name, snr)] = capacity_values[median_idx].item()

            # Find outage capacity (CDF = outage_prob)
            outage_idx = torch.argmin(torch.abs(cdf_values - outage_prob))
            outage_capacity = capacity_values[outage_idx].item()
            outage_capacity_values.append(outage_capacity)

        # Store the outage capacity at the highest SNR for annotations
        outage_capacities[name] = outage_capacity_values[-1]

    legend_elements = []

    # Plot CDFs with optimized rendering
    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        for j, snr in enumerate(snr_points):
            # Get pre-computed values
            capacity_values = capacity_values_dict[name][snr]
            cdf = cdf_dict[name][snr]

            # Convert to numpy once for plotting
            capacity_numpy = capacity_values.cpu().numpy()
            cdf_numpy = cdf.cpu().numpy()

            # Calculate alpha based on SNR index for visual differentiation
            alpha = 0.4 + 0.6 * (j / len(snr_points))

            # Plot with reduced number of points in ultra_fast mode
            labelx = f"{name} at {snr}dB" if i == 0 else None  # Only label first set to avoid duplicates
            if PERFORMANCE_MODE == "ultra_fast":
                # Subsample points for faster plotting
                step = max(1, len(capacity_numpy) // 50)
                (line,) = ax4.plot(capacity_numpy[::step], cdf_numpy[::step], "-", color=color, alpha=alpha, linewidth=2, label=labelx)
            else:
                (line,) = ax4.plot(capacity_numpy, cdf_numpy, "-", color=color, alpha=alpha, linewidth=2.5, label=labelx)

            # Add to legend only for the first channel (to avoid cluttering)
            if i == 0:
                legend_elements.append(line)

    # Add channel types to legend
    from matplotlib.patches import Patch

    for name, color in zip(channel_names, channel_colors):
        legend_elements.append(Patch(facecolor=color, label=name))

    # Add median lines - simplified in ultra_fast mode
    if PERFORMANCE_MODE != "ultra_fast":
        for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
            for j, snr in enumerate(snr_points):
                if j == len(snr_points) - 1:  # Only for highest SNR
                    median_capacity = median_capacities[(name, snr)]

                    # Add vertical line at median
                    ax4.axvline(x=median_capacity, color=color, linestyle="--", alpha=0.7, linewidth=1)

                    # Annotate median
                    ax4.text(median_capacity + 0.2, 0.5, f"Median\n{median_capacity:.2f}", color=color, fontsize=10, ha="left", va="center")

    # Add outage capacity indicators - simplified in ultra_fast mode
    if PERFORMANCE_MODE != "ultra_fast":
        # Add horizontal line at outage probability
        ax4.axhline(y=outage_prob, color="black", linestyle=":", alpha=0.5)
        ax4.text(0.5, outage_prob + 0.02, "10% Outage Probability", fontsize=10, va="bottom")

        for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
            outage_capacity = outage_capacities[name]

            # Plot outage capacity point
            ax4.plot(outage_capacity, outage_prob, "o", color=color, markersize=10, markeredgecolor="black", markeredgewidth=1.5)

            # Skip detailed annotations in fast mode
            if PERFORMANCE_MODE != "fast":
                # Add annotation for outage capacity
                ax4.annotate(
                    f"{name}\nOutage Capacity\n{outage_capacity:.2f} at 10%",
                    xy=(outage_capacity, outage_prob),
                    xytext=(outage_capacity + (-3 if i == 0 else 1), outage_prob - 0.1),
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
                )

    # Customize the plot - adapt level of detail based on performance mode
    title_fontsize = 14 if PERFORMANCE_MODE == "ultra_fast" else 16
    label_fontsize = 12 if PERFORMANCE_MODE == "ultra_fast" else 14

    ax4.set_xlabel("Capacity (bits/channel use)", fontsize=label_fontsize, fontweight="bold")
    ax4.set_ylabel("Probability (Capacity ≤ x)", fontsize=label_fontsize, fontweight="bold")
    ax4.set_title("Capacity CDF for Fading Channels at Different SNR Values", fontsize=title_fontsize, fontweight="bold")
    ax4.grid(True, alpha=0.3, linestyle="--")

    # Add SNR values to legend
    snr_patches = [Line2D([0], [0], color="gray", alpha=0.4 + 0.6 * (j / len(snr_points)), linewidth=2.5, label=f"SNR = {snr} dB") for j, snr in enumerate(snr_points)]

    # Create a legend with adapted complexity
    if PERFORMANCE_MODE == "ultra_fast":
        ax4.legend(handles=snr_patches + legend_elements, loc="lower right", fontsize=10)
    else:
        ax4.legend(handles=snr_patches + legend_elements, loc="lower right", fontsize=11, framealpha=0.9, fancybox=True, shadow=True, ncol=2)

    # Add annotations only in detailed modes
    if PERFORMANCE_MODE not in ["ultra_fast", "fast"]:
        ax4.annotate("Steeper CDF slopes\nindicate less variability", xy=(3.5, 0.2), xytext=(4.5, 0.2), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax4.annotate("Rician fading has higher\ncapacity than Rayleigh\ndue to LOS component", xy=(6, 0.8), xytext=(4, 0.8), arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Replace tight_layout with direct figure adjustments to avoid warnings
    fig4.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.15)

    # Report timing for this example
    example_time = time.time() - example_start_time
    print(f"  Example 4 completed in {example_time:.1f} seconds")

# %%
# Conclusions: Key Insights from Capacity Analysis
# -------------------------------------------------
# From our channel capacity analysis, we can draw several important conclusions:
#
# * **Modulation Order vs. SNR Tradeoff**: Higher-order modulations achieve greater
#   capacity at high SNR, while simpler schemes like BPSK are more robust at low SNR.
#
# * **Channel Impact**: Channel conditions significantly affect achievable capacity.
#   AWGN channels provide the best performance, followed by Rician and then Rayleigh
#   fading channels.
#
# * **MIMO Gains**: Capacity scales approximately linearly with min(nTx, nRx) at high SNR,
#   demonstrating the significant multiplexing gain of MIMO systems.
#
# * **Outage Capacity**: For fading channels, outage capacity provides a more
#   reliable performance metric than average capacity for real-world system design.
#
# * **Performance-Complexity Tradeoff**: The fast mode settings demonstrated how
#   performance analysis can be done with reduced computational complexity while
#   still obtaining meaningful insights.

# %%
# Performance Optimization Techniques
# ------------------------------------
# This example implements several techniques to improve computational efficiency:
#
# 1. **Fast Mode Option**: Enables quick execution by reducing sample sizes and
#    limiting the number of examples run.
#
# 2. **Parallel Processing**: The CapacityAnalyzer uses multiple CPU cores to
#    speed up calculations.
#
# 3. **Histogram-based Estimation**: Uses faster histogram-based methods for
#    capacity estimation rather than more computationally intensive KNN methods.
#
# 4. **Selective Visualization**: Only create detailed visualizations when needed,
#    using simpler plots for quick analysis.
#
# 5. **Adaptive Sample Sizes**: Sample sizes scaled based on visualization needs.

# %%
# Summary of Runtime Performance
# -------------------------------
# The total runtime depends on which examples are executed and what settings are used.

# Report execution time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.1f} seconds")

# Print performance statistics
if FAST_MODE:
    print(f"Fast mode enabled: Running with {NUM_SYMBOLS} symbols, {NUM_SNR_POINTS} SNR points")
    print(f"Examples executed: {EXAMPLES_TO_RUN}")
    print("\nTip: For more detailed analysis, set FAST_MODE = False")
else:
    print(f"Detailed mode: Running with {NUM_SYMBOLS} symbols, {NUM_SNR_POINTS} SNR points")
    print(f"Examples executed: {EXAMPLES_TO_RUN}")
    print("\nTip: For faster execution, set FAST_MODE = True")

# Calculate and print average time per example
examples_run = len(EXAMPLES_TO_RUN)
if examples_run > 0:
    print(f"Average time per example: {total_time/examples_run:.1f} seconds")

# %%
# Further Reading
# ----------------
# For more details on channel capacity analysis, refer to:
#
# * Shannon's original paper on information theory: :cite:`shannon1948mathematical`
# * The comprehensive textbook on digital communications: :cite:`proakis2007digital`
# * MIMO communication systems: :cite:`goldsmith2003capacity`
# * Wireless communications principles: :cite:`rappaport2024wireless`
