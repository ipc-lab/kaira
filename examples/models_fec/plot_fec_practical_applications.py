"""
====================================================================
Interactive FEC Code Comparison for Real-World Applications
====================================================================

This example showcases various forward error correction (FEC) coding schemes in
real-world applications through interactive, visually rich demonstrations. We'll
explore how different codes perform under various channel conditions and application
constraints, with a focus on the practical tradeoffs between redundancy, error
correction capability, and computational complexity.
"""

from typing import Dict, List, Set, cast

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# %%
# Setting up
# ----------------------
# We'll set random seeds for reproducibility and configure our visualization settings
# to create beautiful and informative plots.

torch.manual_seed(42)
np.random.seed(42)

# Configure visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Create custom colormaps for our visualizations
error_cmap = LinearSegmentedColormap.from_list("ErrorMap", ["#ffffff", "#ff9999"])
correct_cmap = LinearSegmentedColormap.from_list("CorrectMap", ["#ffffff", "#99ff99"])
gradient_cmap = LinearSegmentedColormap.from_list("GradientMap", ["#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494"])

# %%
# Visualizing Real-World FEC Applications
# -------------------------------------------------------------------------------------------
# Let's create a beautiful infographic that shows where different FEC codes
# are used in real-world applications, based on their characteristics.

# Define application domains with their typical requirements
applications = [
    {"name": "Deep Space Communication", "reliability": 0.9999, "latency": "High", "codes": ["Reed-Solomon", "Turbo Codes", "LDPC"], "color": "#3498db"},
    {"name": "Satellite Broadcasting", "reliability": 0.999, "latency": "Medium", "codes": ["LDPC", "BCH", "Reed-Solomon"], "color": "#2ecc71"},
    {"name": "Mobile Communication", "reliability": 0.99, "latency": "Low", "codes": ["Convolutional", "Turbo Codes", "LDPC"], "color": "#e74c3c"},
    {"name": "Storage Systems", "reliability": 0.99999, "latency": "Medium", "codes": ["Reed-Solomon", "LDPC", "Hamming"], "color": "#9b59b6"},
    {"name": "Optical Communication", "reliability": 0.9999, "latency": "Low", "codes": ["BCH", "FEC-Enhanced Ethernet", "Reed-Solomon"], "color": "#f39c12"},
    {"name": "WiFi/LAN", "reliability": 0.99, "latency": "Very Low", "codes": ["Hamming", "Convolutional", "CRC+ARQ"], "color": "#1abc9c"},
]

# Create a mapping of latency descriptions to numeric values for plotting
latency_map: Dict[str, int] = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4, "Very High": 5}


# Function to safely convert any value to float
def safe_float(value):
    """Safely convert a value to float, handling various input types."""
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (ValueError, TypeError):
        # Return a default value or raise a more informative error
        return 0.0  # Or some default value


# Extract data for plotting
app_names = [app["name"] for app in applications]
reliability = [safe_float(app["reliability"]) for app in applications]  # Safely handle any type
latency = [latency_map[cast(str, app["latency"])] for app in applications]
colors = [app["color"] for app in applications]

# Create the bubble chart visualization
plt.figure(figsize=(16, 10))

# Convert reliability to log scale for better visualization
# (1 - reliability) represents the error rate, and we're taking the negative log
error_emphasis = [-np.log10(1 - r) for r in reliability]

# Size bubbles based on the reliability requirement
bubble_sizes = [e * 100 + 300 for e in error_emphasis]

# Create scatter plot
scatter = plt.scatter(latency, error_emphasis, s=bubble_sizes, c=colors, alpha=0.7, edgecolors="black", linewidth=2)

# Add application labels with shadow effect for better visibility
for i, (x, y, name) in enumerate(zip(latency, error_emphasis, app_names)):
    txt = plt.annotate(name, xy=(x, y), xytext=(0, 0), textcoords="offset points", fontsize=12, fontweight="bold", color="white", ha="center", va="center")
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground=colors[i])])

# Create a separate legend for coding schemes
all_codes: Set[str] = set()
for app in applications:
    for code in cast(List[str], app["codes"]):  # Fix: explicit type cast
        all_codes.add(code)
all_codes_list = sorted(list(all_codes))  # Fix: convert to list after operations
code_colors = plt.cm.tab10(np.linspace(0, 1, len(all_codes_list)))

# Add a custom legend for the coding schemes
legend_elements = []
for i, (code, color) in enumerate(zip(all_codes_list, code_colors)):
    count = sum(1 for app in applications if code in cast(List[str], app["codes"]))  # Fix: explicit type cast
    legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=f"{code} ({count} applications)"))

plt.legend(handles=legend_elements, title="FEC Coding Schemes", loc="upper left", fontsize=12, title_fontsize=14)

# Add a title and axis labels
plt.title("FEC Codes in Real-World Applications", fontsize=20, fontweight="bold", pad=20)
plt.xlabel("Latency Sensitivity", fontsize=16)
plt.ylabel("Reliability Requirement (-log₁₀(Error Rate))", fontsize=16)

# Set x-axis tick labels
plt.xticks(list(latency_map.values()), list(latency_map.keys()), fontsize=12)

# Add a grid for better readability
plt.grid(True, alpha=0.3, linestyle="--")

# Add annotations explaining the chart
plt.figtext(0.5, 0.02, "Bubble size represents reliability requirement: larger bubbles need more reliable transmission", ha="center", fontsize=14, bbox={"facecolor": "#f8f9fa", "alpha": 0.8, "pad": 5, "boxstyle": "round,pad=0.5"})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# %%
# Visualizing FEC Code Performance Across Channel Conditions
# -------------------------------------------------------------------------------------------------------------------------------------
# Now, let's create an interactive visualization showing how different FEC codes
# perform across varying channel conditions.

# Define the FEC codes we'll compare
fec_codes = [
    {"name": "No Coding", "color": "#3498db", "line_style": "-"},
    {"name": "Hamming(7,4)", "color": "#2ecc71", "line_style": "--"},
    {"name": "BCH(15,7)", "color": "#e74c3c", "line_style": "-."},
    {"name": "Reed-Solomon(255,223)", "color": "#9b59b6", "line_style": ":"},
    {"name": "Turbo Code", "color": "#f39c12", "line_style": "-"},
    {"name": "LDPC", "color": "#1abc9c", "line_style": "--"},
]

# Define performance curves (simplified models)
# x-axis: SNR in dB, y-axis: Bit Error Rate (BER)
snr_range = np.linspace(0, 20, 100)  # SNR from 0 to 20 dB


# Simplified BER models for visualization
def no_coding_ber(snr):
    """Simplified BER model for uncoded transmission."""
    return 0.5 * np.exp(-snr / 10)


def hamming_ber(snr):
    """Simplified BER model for Hamming code."""
    uncoded = no_coding_ber(snr)
    # Hamming can correct 1 error in 7 bits
    return 21 * (uncoded**2) * (1 - uncoded) ** 5 + 35 * (uncoded**3) * (1 - uncoded) ** 4 + 35 * (uncoded**4) * (1 - uncoded) ** 3 + 21 * (uncoded**5) * (1 - uncoded) ** 2 + 7 * (uncoded**6) * (1 - uncoded) + uncoded**7


def bch_ber(snr):
    """Simplified BER model for BCH code."""
    uncoded = no_coding_ber(snr)
    # BCH(15,7) can correct 2 errors
    return uncoded**3 / 5  # Simplified approximation


def rs_ber(snr):
    """Simplified BER model for Reed-Solomon code."""
    uncoded = no_coding_ber(snr)
    # RS has excellent burst error correction
    return uncoded**4 / 10  # Simplified approximation


def turbo_ber(snr):
    """Simplified BER model for Turbo code."""
    # Turbo codes have a characteristic "waterfall" region
    return 0.5 * np.exp(-snr / 5) / (1 + np.exp((snr - 10) / 2))


def ldpc_ber(snr):
    """Simplified BER model for LDPC code."""
    # LDPC codes also have a waterfall region but different characteristics
    return 0.5 * np.exp(-snr / 5) / (1 + np.exp((snr - 8) / 1.5))


# Calculate BER curves
ber_curves = [no_coding_ber(snr_range), hamming_ber(snr_range), bch_ber(snr_range), rs_ber(snr_range), turbo_ber(snr_range), ldpc_ber(snr_range)]

# Create the performance comparison plot
plt.figure(figsize=(16, 10))

# Plot each BER curve
for i, (codee, ber) in enumerate(zip(fec_codes, ber_curves)):
    plt.semilogy(snr_range, ber, label=codee["name"], color=codee["color"], linestyle=codee["line_style"], linewidth=3)

# Add vertical lines at key SNR thresholds for different applications
applications = [
    {"name": "Voice Communication", "snr": 4, "ber": 1e-3, "color": "#3498db"},
    {"name": "Digital TV", "snr": 8, "ber": 1e-6, "color": "#2ecc71"},
    {"name": "Data Transmission", "snr": 12, "ber": 1e-8, "color": "#e74c3c"},
    {"name": "Storage Systems", "snr": 15, "ber": 1e-12, "color": "#9b59b6"},
]

# Add application requirement points
for app in applications:
    plt.axvline(x=app["snr"], color=app["color"], linestyle="--", alpha=0.5)
    plt.axhline(y=app["ber"], color=app["color"], linestyle="--", alpha=0.5)

    # Draw a circle at the intersection
    plt.scatter([app["snr"]], [app["ber"]], color=app["color"], s=100, zorder=5, edgecolor="black", linewidth=1.5)

    # Add a label with the application name
    plt.annotate(app["name"], xy=(app["snr"], app["ber"]), xytext=(10, -10), textcoords="offset points", fontsize=12, fontweight="bold", color=app["color"], bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=app["color"], alpha=0.8))

# Enhance the plot aesthetics
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("Signal-to-Noise Ratio (SNR) in dB", fontsize=14)
plt.ylabel("Bit Error Rate (BER)", fontsize=14)
plt.title("FEC Code Performance Comparison", fontsize=18, fontweight="bold")
plt.legend(loc="lower left", fontsize=12, title="Coding Schemes", title_fontsize=14)

# Set axis limits
plt.xlim(0, 20)
plt.ylim(1e-15, 1)

# Add explanatory annotations
txt = plt.figtext(0.25, 0.02, "← Better (Lower BER is better)", ha="center", fontsize=14, color="#e74c3c", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# %%
# Visualizing FEC Overhead in Practical Applications
# -----------------------------------------------------------------------------------------------------------------
# Next, let's create a visualization that helps understand the overhead introduced
# by different FEC schemes and how it affects throughput in real-world applications.

# Set up the application scenarios
scenarios = [
    {"name": "Audio Streaming", "data_rate": 128, "unit": "Kbps", "color": "#3498db"},
    {"name": "Standard Definition Video", "data_rate": 2, "unit": "Mbps", "color": "#2ecc71"},
    {"name": "High Definition Video", "data_rate": 8, "unit": "Mbps", "color": "#e74c3c"},
    {"name": "4K Video Streaming", "data_rate": 25, "unit": "Mbps", "color": "#9b59b6"},
    {"name": "Cloud Backup", "data_rate": 100, "unit": "Mbps", "color": "#f39c12"},
]

# Define FEC schemes and their overhead
fec_schemes = [
    {"name": "No FEC", "overhead": 0.0, "reliability": 0.9},
    {"name": "Hamming", "overhead": 0.75, "reliability": 0.95},  # 4 data bits, 3 parity bits
    {"name": "Reed-Solomon (Light)", "overhead": 0.14, "reliability": 0.99},  # e.g., (255,223)
    {"name": "Reed-Solomon (Strong)", "overhead": 0.3, "reliability": 0.999},  # e.g., (255,191)
    {"name": "LDPC", "overhead": 0.5, "reliability": 0.9999},  # Rate 2/3 code
    {"name": "Turbo Code", "overhead": 1.0, "reliability": 0.99999},  # Rate 1/2 code
]

# Create the visualization
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 1, height_ratios=[2, 1.5], figure=fig, hspace=0.3)

# First subplot: Overhead impact on throughput
ax1 = fig.add_subplot(gs[0])

# Set up bar chart data
scenarios_names = [s["name"] for s in scenarios]
x = np.arange(len(scenarios))
width = 0.15  # the width of the bars
multiplier = 0

# Plot bars for each FEC scheme
for fec in fec_schemes:
    # Calculate effective data rate after FEC overhead
    effective_rates = [safe_float(s["data_rate"]) / (1 + safe_float(fec["overhead"])) for s in scenarios]  # Fix: explicit type conversion

    offset = width * multiplier
    rects = ax1.bar(x + offset, effective_rates, width, label=fec["name"], alpha=0.8, edgecolor="black", linewidth=1)

    # Add reliability annotation inside bars
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height > safe_float(scenarios[i]["data_rate"]) * 0.15:  # Only add text if bar is tall enough
            ax1.text(rect.get_x() + rect.get_width() / 2.0, height * 0.5, f'{fec["reliability"]:.3f}', ha="center", va="center", fontsize=9, color="white", fontweight="bold", rotation=90)

    multiplier += 1

# Set labels and title
ax1.set_ylabel("Effective Throughput", fontsize=14)
ax1.set_title("Impact of FEC Overhead on Effective Throughput by Application", fontsize=18, fontweight="bold")
ax1.set_xticks(x + width * len(fec_schemes) / 2 - width / 2)
ax1.set_xticklabels(scenarios_names, fontsize=12)
ax1.legend(title="FEC Scheme", title_fontsize=12)

# Add original data rate markers
for i, scenario in enumerate(scenarios):
    ax1.axhline(y=scenario["data_rate"], color=scenario["color"], linestyle="--", alpha=0.5)
    ax1.text(len(scenarios) - 0.2, scenario["data_rate"], f'Original: {scenario["data_rate"]} {scenario["unit"]}', va="bottom", ha="right", fontsize=10, color=scenario["color"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=scenario["color"], alpha=0.8))

# Add grid for better readability
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Second subplot: Reliability vs. Overhead tradeoff
ax2 = fig.add_subplot(gs[1])

# Extract data for the scatter plot
overheads = [safe_float(fec["overhead"]) for fec in fec_schemes]
reliabilities = [safe_float(fec["reliability"]) for fec in fec_schemes]
names = [fec["name"] for fec in fec_schemes]

# Convert reliability to log scale of error rate for better visualization
error_log = [-np.log10(1 - r) for r in reliabilities]

# Create scatter plot
scatter = ax2.scatter(overheads, error_log, s=[(e + 1) * 100 for e in error_log], c=error_log, cmap="viridis", alpha=0.8, edgecolors="black")

# Add labels for each point
for i, (x, y, name) in enumerate(zip(overheads, error_log, names)):
    ax2.annotate(name, xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=12, fontweight="bold")

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("Reliability (-log₁₀ of Error Rate)", fontsize=12)

# Set labels and title
ax2.set_xlabel("FEC Overhead Ratio", fontsize=14)
ax2.set_ylabel("Reliability (Higher is Better)", fontsize=14)
ax2.set_title("Tradeoff Between FEC Overhead and Transmission Reliability", fontsize=16, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.7)

# Add explanatory text
plt.figtext(0.5, 0.01, "Higher overhead provides better error correction but reduces effective throughput", ha="center", fontsize=14, bbox={"facecolor": "#f8f9fa", "alpha": 0.8, "pad": 5})

# Replace tight_layout with subplots_adjust
fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.1, hspace=0.3)
plt.show()

# %%
# Visualizing FEC Performance with Different Error Patterns
# ------------------------------------------------------------------------------------------------------------------------------------
# Different FEC codes handle different error patterns with varying effectiveness.
# Let's visualize how different codes perform against random errors vs. burst errors.

# Set up the visualization
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, height_ratios=[1, 3], figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Create two subplots to show different error patterns
ax_pattern1 = fig.add_subplot(gs[0, 0])
ax_pattern2 = fig.add_subplot(gs[0, 1])

# Define error patterns
random_errors = np.zeros(100)
random_error_indices = np.random.choice(100, 10, replace=False)
random_errors[random_error_indices] = 1

burst_errors = np.zeros(100)
burst_start = np.random.randint(0, 90)
burst_errors[burst_start : burst_start + 10] = 1

# Visualize the error patterns
ax_pattern1.bar(np.arange(100), random_errors, color="#e74c3c", edgecolor="black", width=1)
ax_pattern1.set_title("Random Error Pattern", fontsize=14, fontweight="bold")
ax_pattern1.set_ylabel("Error Present")
ax_pattern1.set_ylim(0, 1.2)
ax_pattern1.set_xticks([])

ax_pattern2.bar(np.arange(100), burst_errors, color="#3498db", edgecolor="black", width=1)
ax_pattern2.set_title("Burst Error Pattern", fontsize=14, fontweight="bold")
ax_pattern2.set_ylabel("Error Present")
ax_pattern2.set_ylim(0, 1.2)
ax_pattern2.set_xticks([])

# Add pattern effectiveness visualization
ax_perf = fig.add_subplot(gs[1, :])

# Define FEC codes and their effectiveness against different error patterns
fec_codes = [
    {"name": "No Coding", "random_effectiveness": "0.0", "burst_effectiveness": "0.0"},
    {"name": "Parity Check", "random_effectiveness": "0.1", "burst_effectiveness": "0.05"},
    {"name": "Hamming", "random_effectiveness": "0.5", "burst_effectiveness": "0.1"},
    {"name": "Convolutional", "random_effectiveness": "0.7", "burst_effectiveness": "0.5"},
    {"name": "BCH", "random_effectiveness": "0.8", "burst_effectiveness": "0.4"},
    {"name": "Reed-Solomon", "random_effectiveness": "0.6", "burst_effectiveness": "0.9"},
    {"name": "Interleaved RS", "random_effectiveness": "0.7", "burst_effectiveness": "0.95"},
    {"name": "LDPC", "random_effectiveness": "0.9", "burst_effectiveness": "0.7"},
    {"name": "Turbo", "random_effectiveness": "0.85", "burst_effectiveness": "0.8"},
    {"name": "Polar Codes", "random_effectiveness": "0.95", "burst_effectiveness": "0.75"},
]

# Extract data for plotting
names = [code["name"] for code in fec_codes]
random_eff = [float(code["random_effectiveness"]) for code in fec_codes]  # Fix: explicit type conversion
burst_eff = [float(code["burst_effectiveness"]) for code in fec_codes]  # Fix: explicit type conversion

# Set up x positions
x = np.arange(len(names))
width = 0.35

# Create grouped bar chart
rects1 = ax_perf.bar(x - width / 2, random_eff, width, label="Random Errors", color="#e74c3c", edgecolor="black", alpha=0.8)
rects2 = ax_perf.bar(x + width / 2, burst_eff, width, label="Burst Errors", color="#3498db", edgecolor="black", alpha=0.8)

# Add labels, title, and legend
ax_perf.set_ylabel("Error Correction Effectiveness", fontsize=14)
ax_perf.set_title("FEC Code Performance by Error Pattern Type", fontsize=18, fontweight="bold")
ax_perf.set_xticks(x)
ax_perf.set_xticklabels(names, rotation=45, ha="right", fontsize=12)
ax_perf.legend(fontsize=12)
ax_perf.set_ylim(0, 1.1)


# Add value labels above bars
def add_labels(rects):
    """Add value labels above the bars."""
    for rect in rects:
        height = rect.get_height()
        ax_perf.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")  # 3 points vertical offset


add_labels(rects1)
add_labels(rects2)

# Add a grid for better readability
ax_perf.grid(axis="y", linestyle="--", alpha=0.7)

# Add annotations explaining the patterns
ax_pattern1.annotate("Random errors are \nindependent bit flips\nscattered throughout\nthe transmission", xy=(50, 0.6), xytext=(10, 0), textcoords="offset points", ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.8))

ax_pattern2.annotate("Burst errors occur when\nmultiple consecutive bits\nare corrupted, often due\nto physical interference", xy=(50, 0.6), xytext=(10, 0), textcoords="offset points", ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#3498db", alpha=0.8))

# Add an explanatory note
plt.figtext(0.5, 0.01, "Different FEC codes are optimized for different error patterns - choose based on your channel characteristics", ha="center", fontsize=14, bbox={"facecolor": "#f8f9fa", "alpha": 0.8, "pad": 5})

# Replace tight_layout with subplots_adjust
fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.12, hspace=0.3, wspace=0.3)
plt.show()

# %%
# Conclusion
# ---------------------
# In this example, we've created visually engaging demonstrations of how FEC codes
# are used in real-world applications:
#
# Key points:
# - Different application domains have different requirements for latency and reliability
# - FEC codes vary in their effectiveness against different error patterns
# - There's a fundamental tradeoff between overhead (redundancy) and error correction capability
# - The choice of FEC scheme should be tailored to the specific application needs and channel conditions
#
# These visualizations help demonstrate the practical considerations in selecting
# appropriate error correction strategies for different communication systems.
#
# References:
# - :cite:`lin2004error` - Comprehensive coverage of error control coding fundamentals
# - :cite:`moon2005error` - Mathematical methods for error correction algorithms
# - :cite:`richardson2008modern` - Modern approaches to coding theory including LDPC and Turbo codes
# - :cite:`berlekamp1968algebraic` - Classic text on algebraic coding theory
