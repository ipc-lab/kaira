"""
==================================================================
Advanced Polar Code Visualization with Decoding Animations
==================================================================

This example demonstrates advanced visualizations for Polar codes, including
channel polarization visualization, successive cancellation decoding steps,
and performance comparisons between different decoders.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from tqdm import tqdm

from kaira.channels.analog import AWGNChannel
from kaira.models.fec.decoders import BeliefPropagationPolarDecoder, SuccessiveCancellationDecoder
from kaira.models.fec.encoders import PolarCodeEncoder

# %%
# Setting up
# --------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# Configure visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Custom color schemes
polarization_cmap = LinearSegmentedColormap.from_list("polarization", ["#c62828", "#ffeb3b", "#2e7d32"], N=256)
modern_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# %%
# Channel Polarization Visualization
# --------------------------------------


def visualize_channel_polarization(n_bits, snr_db=3.0):
    """Visualize the channel polarization process for polar codes."""

    # Calculate channel capacities for demonstration
    n = 2**n_bits

    # Simulate channel polarization using Bhattacharyya parameters
    # This is a simplified visualization of the concept
    noise_variance = 10 ** (-snr_db / 10)
    base_reliability = 1 - 2 * np.exp(-1 / (2 * noise_variance))

    # Generate polarized channel reliabilities
    reliabilities = []
    for stage in range(n_bits + 1):
        if stage == 0:
            reliabilities.append([base_reliability])
        else:
            prev_rel = reliabilities[-1]
            new_rel = []
            for r in prev_rel:
                # Good channels become better, bad channels become worse
                good_channel = min(1.0, 2 * r - r**2)
                bad_channel = r**2
                new_rel.extend([bad_channel, good_channel])
            reliabilities.append(new_rel)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(f"Polar Code Channel Polarization (N={n}, SNR={snr_db}dB)", fontsize=16, fontweight="bold")

    # Polarization tree visualization
    for stage, stage_rel in enumerate(reliabilities):
        num_channels = len(stage_rel)
        y_positions = np.linspace(-1, 1, num_channels)

        # Draw channels as circles with color based on reliability
        for i, (y_pos, rel) in enumerate(zip(y_positions, stage_rel)):
            color = polarization_cmap(rel)
            circle = Circle((stage, y_pos), 0.08, facecolor=color, edgecolor="black", linewidth=1, zorder=10)
            ax1.add_patch(circle)

            # Add reliability text for final stage
            if stage == len(reliabilities) - 1:
                ax1.text(stage + 0.2, y_pos, f"{rel:.3f}", ha="left", va="center", fontsize=8, fontweight="bold")

            # Draw connections to next stage
            if stage < len(reliabilities) - 1:
                # Connect to two channels in next stage
                next_indices = [2 * i, 2 * i + 1]
                for next_idx in next_indices:
                    if next_idx < len(reliabilities[stage + 1]):
                        next_y = np.linspace(-1, 1, len(reliabilities[stage + 1]))[next_idx]
                        arrow = FancyArrowPatch((stage + 0.08, y_pos), (stage + 0.92, next_y), arrowstyle="-", color="gray", alpha=0.6, linewidth=1)
                        ax1.add_patch(arrow)

    ax1.set_xlim(-0.5, len(reliabilities) - 0.5)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlabel("Polarization Stage")
    ax1.set_ylabel("Channel Index")
    ax1.set_title("Channel Polarization Tree")
    ax1.set_xticks(range(len(reliabilities)))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=polarization_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label("Channel Reliability", fontsize=10)

    # Final channel reliabilities histogram
    final_reliabilities = reliabilities[-1]
    ax2.bar(range(len(final_reliabilities)), final_reliabilities, color=[polarization_cmap(r) for r in final_reliabilities], edgecolor="black", linewidth=1)
    ax2.set_xlabel("Bit Channel Index")
    ax2.set_ylabel("Reliability")
    ax2.set_title("Final Channel Reliabilities")
    ax2.grid(True, alpha=0.3)

    # Information bit selection
    k = n // 2  # Half rate for demonstration
    sorted_indices = np.argsort(final_reliabilities)
    info_bits = sorted_indices[-k:]  # Most reliable channels
    frozen_bits = sorted_indices[:-k]  # Least reliable channels

    bit_types = np.zeros(n)
    bit_types[info_bits] = 1

    colors = ["red" if bt == 0 else "green" for bt in bit_types]
    ax3.bar(range(n), np.ones(n), color=colors, alpha=0.7, edgecolor="black", linewidth=1)
    ax3.set_xlabel("Bit Position")
    ax3.set_ylabel("Bit Type")
    ax3.set_title(f"Information vs Frozen Bits (Rate = {k/n:.2f})")
    ax3.set_yticks([0.5])
    ax3.set_yticklabels(["Frozen=Red, Info=Green"])
    ax3.grid(True, axis="x", alpha=0.3)

    # Capacity evolution
    capacities = []
    for stage_rel in reliabilities:
        avg_capacity = np.mean([r for r in stage_rel])
        capacities.append(avg_capacity)

    ax4.plot(range(len(capacities)), capacities, "o-", color=modern_palette[0], linewidth=3, markersize=8)
    ax4.set_xlabel("Polarization Stage")
    ax4.set_ylabel("Average Channel Capacity")
    ax4.set_title("Capacity Evolution During Polarization")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    return fig, info_bits, frozen_bits


# %%
# Visualize Channel Polarization
# --------------------------------------

print("Polar Code Channel Polarization Analysis")
print("=" * 50)

n_bits = 3  # N = 8
fig, info_bits, frozen_bits = visualize_channel_polarization(n_bits, snr_db=2.0)
plt.show()

print(f"Information bit positions: {sorted(info_bits)}")
print(f"Frozen bit positions: {sorted(frozen_bits)}")

# %%
# Successive Cancellation Decoding Visualization
# -----------------------------------------------


class SuccessiveCancellationVisualizer:
    """Visualize successive cancellation decoding step by step."""

    def __init__(self, n, info_bits, frozen_bits):
        self.n = n
        self.info_bits = set(info_bits)
        self.frozen_bits = set(frozen_bits)
        self.decisions = np.zeros(n)
        self.llrs = np.zeros(n)
        self.step_history = []

    def decode_step(self, received_llrs, step):
        """Perform one step of SC decoding."""
        # Simplified SC decoding for visualization
        if step < self.n:
            if step in self.frozen_bits:
                # Frozen bit - set to 0
                self.decisions[step] = 0
                self.llrs[step] = float("inf")  # Certain decision
            else:
                # Information bit - make decision based on LLR
                self.llrs[step] = received_llrs[step] + np.sum(self.decisions[:step])
                self.decisions[step] = 1 if self.llrs[step] < 0 else 0

            self.step_history.append({"step": step, "decisions": self.decisions.copy(), "llrs": self.llrs.copy(), "current_bit": step})

    def visualize_step(self, step_data):
        """Visualize a single SC decoding step."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle(f"Successive Cancellation Decoding - Step {step_data['step'] + 1}", fontsize=16, fontweight="bold")

        current_bit = step_data["current_bit"]
        decisions = step_data["decisions"]
        llrs = step_data["llrs"]

        # Current decisions
        colors = []
        for i in range(self.n):
            if i < current_bit:
                colors.append("green" if i in self.info_bits else "red")
            elif i == current_bit:
                colors.append("yellow")
            else:
                colors.append("lightgray")

        ax1.bar(range(self.n), np.ones(self.n), color=colors, edgecolor="black", linewidth=2)

        # Add decision values
        for i in range(current_bit + 1):
            ax1.text(i, 0.5, str(int(decisions[i])), ha="center", va="center", fontsize=14, fontweight="bold", color="white")

        ax1.set_xlabel("Bit Position")
        ax1.set_ylabel("Decision Status")
        ax1.set_title("Decoding Progress")
        ax1.set_ylim(0, 1)

        # Add legend
        ax1.text(0.02, 0.98, "Red=Frozen, Green=Info, Yellow=Current, Gray=Pending", transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), verticalalignment="top")

        # LLR values
        ax2.bar(range(self.n), llrs[: self.n], color=[colors[i] for i in range(self.n)], edgecolor="black", linewidth=1)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Bit Position")
        ax2.set_ylabel("Log-Likelihood Ratio")
        ax2.set_title("LLR Values")
        ax2.grid(True, alpha=0.3)

        # Decoding tree (simplified)
        tree_depth = int(np.log2(self.n))
        for level in range(tree_depth + 1):
            num_nodes = 2**level
            y_positions = np.linspace(-1, 1, num_nodes) if num_nodes > 1 else [0]

            for i, y_pos in enumerate(y_positions):
                # Color based on processing status
                if level == tree_depth:  # Leaf nodes
                    bit_idx = i
                    if bit_idx <= current_bit:
                        color = "green" if bit_idx in self.info_bits else "red"
                    else:
                        color = "lightgray"
                else:
                    color = "lightblue"

                circle = Circle((level, y_pos), 0.1, facecolor=color, edgecolor="black", linewidth=1, zorder=10)
                ax3.add_patch(circle)

                # Add connections
                if level < tree_depth:
                    for child in [2 * i, 2 * i + 1]:
                        if child < 2 ** (level + 1):
                            child_y = np.linspace(-1, 1, 2 ** (level + 1))[child] if 2 ** (level + 1) > 1 else 0
                            arrow = FancyArrowPatch((level + 0.1, y_pos), (level + 0.9, child_y), arrowstyle="-", color="gray", alpha=0.6, linewidth=1)
                            ax3.add_patch(arrow)

        ax3.set_xlim(-0.5, tree_depth + 0.5)
        ax3.set_ylim(-1.3, 1.3)
        ax3.set_xlabel("Tree Level")
        ax3.set_title("Decoding Tree")
        ax3.set_xticks(range(tree_depth + 1))

        # Statistics
        if current_bit >= 0:
            info_decoded = sum(1 for i in range(current_bit + 1) if i in self.info_bits)
            frozen_decoded = sum(1 for i in range(current_bit + 1) if i in self.frozen_bits)

            stats_text = f"""Decoding Statistics:

• Current bit: {current_bit}
• Information bits decoded: {info_decoded}/{len(self.info_bits)}
• Frozen bits processed: {frozen_decoded}/{len(self.frozen_bits)}
• Current decision: {'Frozen (0)' if current_bit in self.frozen_bits else f'Info ({int(decisions[current_bit])})'}
• LLR: {llrs[current_bit]:.3f}
• Progress: {100*(current_bit + 1)/self.n:.1f}%"""

            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        ax4.set_title("Step Information")
        ax4.axis("off")

        return fig


# %%
# Run Successive Cancellation Visualization
# ------------------------------------------

# Create polar code
n = 8
k = 4
info_positions = info_bits[:k]  # Use the most reliable positions
frozen_positions = [i for i in range(n) if i not in info_positions]

encoder = PolarCodeEncoder(k, n, load_rank=False, info_indices=np.array([i in info_positions for i in range(n)]))

# Generate test message
message = torch.randint(0, 2, (1, k), dtype=torch.float32)
codeword = encoder(message)

print("\nSuccessive Cancellation Decoding Example")
print(f"Message bits: {message.squeeze().int().tolist()}")
print(f"Encoded codeword: {codeword.squeeze().int().tolist()}")

# Add noise
snr_db = 3.0
channel = AWGNChannel(snr_db=snr_db)
bipolar_codeword = 2.0 * codeword - 1.0
received = channel(bipolar_codeword)

# Convert to LLRs
noise_variance = 10 ** (-snr_db / 10)
received_llrs = 2 * received.squeeze() / noise_variance

print(f"Received LLRs: {received_llrs.numpy()}")

# Create and run SC visualizer
sc_viz = SuccessiveCancellationVisualizer(n, info_positions, frozen_positions)

# Simulate SC decoding steps
for step in range(n):
    sc_viz.decode_step(received_llrs.numpy(), step)

# Show key steps
for step_idx in [0, 2, 4, 7]:
    if step_idx < len(sc_viz.step_history):
        sc_viz.visualize_step(sc_viz.step_history[step_idx])
        plt.show()

# %%
# Performance Comparison: SC vs BP Decoding
# ------------------------------------------


def compare_polar_decoders():
    """Compare performance between SC and BP decoders."""

    # Test parameters
    snr_range = np.arange(-2, 8, 1)
    num_trials = 200

    # Initialize decoders
    sc_decoder = SuccessiveCancellationDecoder(encoder)
    bp_decoder = BeliefPropagationPolarDecoder(encoder, bp_iters=10)

    results = {"SC": [], "BP": []}

    for snr_db in tqdm(snr_range, desc="Testing decoders"):
        channel = AWGNChannel(snr_db=snr_db)

        # Test both decoders
        for decoder_name, decoder in [("SC", sc_decoder), ("BP", bp_decoder)]:
            errors = 0
            total_bits = 0

            for _ in range(num_trials):
                # Generate random message
                msg = torch.randint(0, 2, (1, k), dtype=torch.float32)
                encoded = encoder(msg)

                # Transmit through channel
                bipolar = 2.0 * encoded - 1.0
                received = channel(bipolar)

                # Decode
                decoded = decoder(received)

                # Count errors
                errors += torch.sum(msg != decoded).item()
                total_bits += msg.numel()

            ber = errors / total_bits
            results[decoder_name].append(ber)

    return snr_range, results


# Run comparison
print("\nComparing SC vs BP decoders...")
snr_range, decoder_results = compare_polar_decoders()

# %%
# Visualize Decoder Comparison
# --------------------------------------

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
fig.suptitle("Polar Code Decoder Performance Comparison", fontsize=16, fontweight="bold")

# BER comparison
for i, (decoder_name, ber_values) in enumerate(decoder_results.items()):
    color = modern_palette[i]
    ax1.semilogy(snr_range, ber_values, "o-", color=color, linewidth=2.5, markersize=8, label=decoder_name)

ax1.grid(True, which="both", alpha=0.3)
ax1.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Bit Error Rate", fontsize=12, fontweight="bold")
ax1.set_title("BER Performance Comparison")
ax1.legend(fontsize=12)
ax1.set_ylim(1e-4, 1)

# Performance gain
snr_target = 3
snr_idx = np.where(snr_range == snr_target)[0][0]
sc_ber = decoder_results["SC"][snr_idx]
bp_ber = decoder_results["BP"][snr_idx]
gain_db = 10 * np.log10(sc_ber / bp_ber) if bp_ber > 0 else 0

categories = ["SC Decoder", "BP Decoder"]
bers = [sc_ber, bp_ber]
colors = [modern_palette[0], modern_palette[1]]

bars = ax2.bar(categories, bers, color=colors, alpha=0.8, edgecolor="black")
ax2.set_yscale("log")
ax2.set_ylabel("Bit Error Rate", fontsize=12, fontweight="bold")
ax2.set_title(f"BER at SNR = {snr_target} dB")
ax2.grid(True, axis="y", alpha=0.3)

# Add value labels
for bar, ber in zip(bars, bers):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height * 1.5, f"{ber:.1e}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Complexity comparison (conceptual)
complexities = {"SC": np.log2(n) * n, "BP": 10 * n}  # Simplified complexity model
ax3.bar(complexities.keys(), complexities.values(), color=colors, alpha=0.8, edgecolor="black")
ax3.set_ylabel("Relative Complexity", fontsize=12, fontweight="bold")
ax3.set_title("Decoding Complexity Comparison")
ax3.grid(True, axis="y", alpha=0.3)

# Summary statistics
summary_text = f"""Polar Code Analysis Summary:

• Code parameters: N={n}, K={k}, R={k/n:.2f}
• Information positions: {sorted(info_positions)}
• Frozen positions: {sorted(frozen_positions)}

Performance at {snr_target} dB SNR:
• SC Decoder BER: {sc_ber:.2e}
• BP Decoder BER: {bp_ber:.2e}
• BP Gain: {gain_db:.1f} dB

Complexity (relative):
• SC: O(N log N) = {complexities['SC']:.0f}
• BP: O(I × N) = {complexities['BP']:.0f}

Key Insights:
• BP offers better performance
• SC has lower complexity
• Both leverage polarization"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
ax4.set_title("Analysis Summary")
ax4.axis("off")

plt.show()

# %%
# Polarization Effect Visualization
# --------------------------------------


def visualize_polarization_effect():
    """Show how different SNRs affect channel polarization."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle("Channel Polarization Effect at Different SNRs", fontsize=16, fontweight="bold")

    snr_values = [-2, 0, 2, 4, 6, 8]

    for idx, snr_db in enumerate(snr_values):
        ax = axes[idx // 3, idx % 3]

        # Calculate polarized channel reliabilities
        n = 16
        noise_variance = 10 ** (-snr_db / 10)
        base_reliability = 1 - 2 * np.exp(-1 / (2 * noise_variance))

        # Simulate polarization for visualization
        reliabilities = [base_reliability]
        num_stages = int(np.log2(n))  # Calculate stages from n
        for _ in range(num_stages):  # stages for N=n
            new_rel = []
            for r in reliabilities:
                bad = r**2
                good = min(1.0, 2 * r - r**2)
                new_rel.extend([bad, good])
            reliabilities = new_rel

        # Sort for better visualization
        reliabilities.sort()

        # Plot
        colors = [polarization_cmap(r) for r in reliabilities]
        ax.bar(range(len(reliabilities)), reliabilities, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Reliability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add statistics
        good_channels = sum(1 for r in reliabilities if r > 0.9)
        bad_channels = sum(1 for r in reliabilities if r < 0.1)
        ax.text(0.7, 0.8, f"Good: {good_channels}\nBad: {bad_channels}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    return fig


# Show polarization effect
visualize_polarization_effect()
plt.show()

# %%
# Conclusion
# --------------------------------------
print("\n" + "=" * 60)
print("ADVANCED POLAR CODE VISUALIZATION SUMMARY")
print("=" * 60)
print(
    """
This example demonstrated advanced visualization techniques for Polar codes:

🔹 Channel Polarization:
  • Polarization tree structure
  • Reliability evolution
  • Information vs frozen bit selection
  • Capacity analysis across stages

🔹 Successive Cancellation Decoding:
  • Step-by-step decision process
  • LLR computation visualization
  • Decoding tree representation
  • Progress tracking

🔹 Decoder Comparison:
  • SC vs BP performance analysis
  • Complexity trade-offs
  • SNR sensitivity study
  • Practical implementation insights

🔹 Polarization Effects:
  • SNR impact on polarization
  • Channel quality distribution
  • Good/bad channel identification

Key Insights:
• Higher SNRs improve polarization effectiveness
• BP decoding offers better performance at higher complexity
• Channel polarization creates distinct reliable/unreliable channels
• Proper frozen bit selection is crucial for performance
• Visual analysis aids in understanding polar code behavior
"""
)
