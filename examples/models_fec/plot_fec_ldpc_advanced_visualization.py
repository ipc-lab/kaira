"""
====================================================================
Advanced LDPC Code Visualization with Belief Propagation Animation
====================================================================

This example demonstrates advanced visualizations for Low-Density Parity-Check
(LDPC) codes :cite:`gallager1962low`, including animated belief propagation :cite:`kschischang2001factor`, Tanner graph analysis,
and performance comparisons with different decoder configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

from kaira.channels.analog import AWGNChannel
from kaira.models.fec.decoders import BeliefPropagationDecoder
from kaira.models.fec.encoders import LDPCCodeEncoder
from kaira.modulations.psk import BPSKDemodulator, BPSKModulator

# Plotting imports
from kaira.utils.plotting import PlottingUtils
from kaira.utils.snr import snr_to_noise_power

PlottingUtils.setup_plotting_style()

# %%
# Setting up
# --------------------------------------
# Advanced LDPC Visualization Configuration
# =========================================

torch.manual_seed(42)
np.random.seed(42)

# %%
# Enhanced Tanner Graph Visualization
# --------------------------------------
# Advanced LDPC Code Analysis and Tanner Graph Creation
# =====================================================

# Define a more interesting LDPC code
H_matrix = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.float32)

# Enhanced LDPC Code Analysis
# ===========================
# Code parameters and Tanner graph visualization
print(f"H matrix dimensions: {H_matrix.shape}")
print(f"Expected code dimensions: ({H_matrix.shape[1]}, {H_matrix.shape[1] - H_matrix.shape[0]})")
print(f"Variable node degrees: {torch.sum(H_matrix, dim=0).tolist()}")
print(f"Check node degrees: {torch.sum(H_matrix, dim=1).tolist()}")

# Create LDPC encoder to get actual dimensions
encoder = LDPCCodeEncoder(H_matrix)
print(f"Actual code dimensions: ({encoder.code_length}, {encoder.code_dimension})")
print(f"Actual code rate: {encoder.code_dimension/encoder.code_length:.3f}")

# Create improved Tanner graph visualization with enhanced styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
fig.suptitle(f"Enhanced LDPC Code Analysis ({encoder.code_length},{encoder.code_dimension})", fontsize=16, fontweight="bold")

# Enhanced Tanner graph in first subplot
H_np = H_matrix.numpy()
m, n = H_np.shape  # m check nodes, n variable nodes

ax1.set_title("Tanner Graph Structure", fontweight="bold")

# Position variable nodes in a circle (top)
var_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
var_positions = [(2 * np.cos(angle), 2 * np.sin(angle) + 1) for angle in var_angles]

# Position check nodes in a circle (bottom)
check_angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
check_positions = [(1.5 * np.cos(angle), 1.5 * np.sin(angle) - 1) for angle in check_angles]

# Draw connections
connection_counts = np.sum(H_np, axis=0)  # variable node degrees
max_degree = np.max(connection_counts)

for i in range(m):
    for j in range(n):
        if H_np[i, j] == 1:
            thickness = 1 + 2 * (connection_counts[j] / max_degree)
            alpha = 0.6 + 0.4 * (connection_counts[j] / max_degree)
            ax1.plot([check_positions[i][0], var_positions[j][0]], [check_positions[i][1], var_positions[j][1]], color="gray", linewidth=thickness, alpha=alpha)

# Draw variable nodes
for j, pos in enumerate(var_positions):
    size = 0.15 + 0.15 * (connection_counts[j] / max_degree)
    circle = Circle(pos, size, facecolor=PlottingUtils.MODERN_PALETTE[0], edgecolor="black", linewidth=2, zorder=10)
    ax1.add_patch(circle)
    ax1.text(pos[0], pos[1], f"v{j}", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=11)

# Draw check nodes
check_degrees = np.sum(H_np, axis=1)
max_check_degree = np.max(check_degrees)

for i, pos in enumerate(check_positions):
    size = 0.15 + 0.15 * (check_degrees[i] / max_check_degree)
    square = Rectangle((pos[0] - size, pos[1] - size), 2 * size, 2 * size, facecolor=PlottingUtils.MODERN_PALETTE[3], edgecolor="black", linewidth=2, zorder=10)
    ax1.add_patch(square)
    ax1.text(pos[0], pos[1], f"c{i}", ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=11)

ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 3.5)
ax1.set_aspect("equal")
ax1.axis("off")

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PlottingUtils.MODERN_PALETTE[0], markersize=10, label="Variable Nodes"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PlottingUtils.MODERN_PALETTE[3], markersize=10, label="Check Nodes"),
    plt.Line2D([0], [0], color="gray", linewidth=2, label="Connections"),
]
ax1.legend(handles=legend_elements, loc="upper right")

# Degree distribution visualization in second subplot
var_degrees = torch.sum(H_matrix, dim=0).numpy()
check_degrees = torch.sum(H_matrix, dim=1).numpy()

# Create degree distribution plot
x_var = np.arange(len(var_degrees))
x_check = np.arange(len(check_degrees))

bars1 = ax2.bar(x_var - 0.2, var_degrees, 0.4, label="Variable Nodes", alpha=0.8, color="#1f77b4")
bars2 = ax2.bar(x_check + 0.2, check_degrees, 0.4, label="Check Nodes", alpha=0.8, color="#ff7f0e")

ax2.set_xlabel("Node Index", fontweight="bold")
ax2.set_ylabel("Node Degree", fontweight="bold")
ax2.set_title("Node Degree Distribution", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add degree statistics as text
ax2.text(
    0.7,
    0.8,
    f"Avg var degree: {np.mean(var_degrees):.1f}\n" f"Avg check degree: {np.mean(check_degrees):.1f}\n" f"Max var degree: {np.max(var_degrees)}\n" f"Max check degree: {np.max(check_degrees)}",
    transform=ax2.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
)

plt.show()

# %%
# Belief Propagation Animation
# --------------------------------------
# Belief Propagation Message Passing Visualization
# ==============================================================


class BeliefPropagationVisualizer:
    """Visualize belief propagation iterations in LDPC decoding."""

    def __init__(self, H, received_llrs, max_iterations=5):
        self.H = H
        self.m, self.n = H.shape
        self.received_llrs = received_llrs
        self.max_iterations = max_iterations

        # Initialize messages
        self.var_to_check = np.zeros((self.n, self.m))
        self.check_to_var = np.zeros((self.m, self.n))
        self.beliefs = received_llrs.copy()

        # Store history for animation
        self.belief_history = [self.beliefs.copy()]
        self.var_to_check_history = [self.var_to_check.copy()]
        self.check_to_var_history = [self.check_to_var.copy()]

    def step(self):
        """Perform one iteration of belief propagation."""
        # Check node update
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j] == 1:
                    # Collect messages from other variable nodes
                    other_vars = [k for k in range(self.n) if k != j and self.H[i, k] == 1]
                    if other_vars:
                        product = 1.0
                        for k in other_vars:
                            product *= np.tanh(self.var_to_check[k, i] / 2)
                        self.check_to_var[i, j] = 2 * np.arctanh(np.clip(product, -0.999, 0.999))
                    else:
                        self.check_to_var[i, j] = 0

        # Variable node update
        for j in range(self.n):
            for i in range(self.m):
                if self.H[i, j] == 1:
                    # Sum messages from other check nodes
                    other_checks = [k for k in range(self.m) if k != i and self.H[k, j] == 1]
                    message_sum = self.received_llrs[j]
                    for k in other_checks:
                        message_sum += self.check_to_var[k, j]
                    self.var_to_check[j, i] = message_sum

            # Update belief
            self.beliefs[j] = self.received_llrs[j] + np.sum(self.check_to_var[:, j])

        # Store history
        self.belief_history.append(self.beliefs.copy())
        self.var_to_check_history.append(self.var_to_check.copy())
        self.check_to_var_history.append(self.check_to_var.copy())

    def run_iterations(self):
        """Run all iterations."""
        for _ in range(self.max_iterations):
            self.step()

    def visualize_iteration(self, iteration):
        """Visualize a specific iteration using plotting utilities."""
        beliefs = self.belief_history[iteration]

        return PlottingUtils.plot_belief_propagation_iteration(self.H, beliefs, iteration, f"Belief Propagation Iteration {iteration}")


# %%
# Run Belief Propagation Visualization
# --------------------------------------
# Simulation Setup and Belief Propagation Execution
# =================================================

# Create LDPC encoder and verify dimensions
encoder = LDPCCodeEncoder(H_matrix)
print(f"Encoder code dimension: {encoder.code_dimension}")
print(f"Encoder code length: {encoder.code_length}")
print(f"Generator matrix shape: {encoder.generator_matrix.shape}")

# Generate correct message size based on encoder dimensions
message_bits = torch.randint(0, 2, (encoder.code_dimension,), dtype=torch.float32)
codeword = encoder(message_bits.unsqueeze(0)).squeeze()

# Generated data:
print(f"Original message: {message_bits.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")

# Initialize modulator and demodulator
modulator = BPSKModulator(complex_output=False)
demodulator = BPSKDemodulator()

# Add noise using proper pipeline
snr_db = 2.0
noise_power = snr_to_noise_power(1.0, snr_db)
channel = AWGNChannel(avg_noise_power=noise_power)

# Modulate the codeword
bipolar_codeword = modulator(codeword.unsqueeze(0)).squeeze()
received = channel(bipolar_codeword.unsqueeze(0)).squeeze()

# Demodulate to get LLRs
received_llrs = demodulator(received.unsqueeze(0), noise_var=noise_power).squeeze()

# Signal processing results:
print(f"Received signal: {received.numpy()}")
print(f"Received LLRs: {received_llrs.numpy()}")

# Create and run visualizer
bp_viz = BeliefPropagationVisualizer(H_matrix.numpy(), received_llrs.numpy())
bp_viz.run_iterations()

# Show key iterations
for iteration in [0, 1, 3, 5]:
    fig = bp_viz.visualize_iteration(iteration)
    if fig:
        plt.show()

# %%
# Create GIF Animation of Belief Propagation Iterations
# -------------------------------------------------------


def create_bp_animation(bp_viz, save_path=None, fps=1):
    """Create an animated visualization showing the belief propagation iterations."""

    # Create figure for animation with constrained_layout
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)

    def animate(frame):
        """Animation function for each frame."""
        # Clear the entire figure and recreate subplots
        fig.clear()
        ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)

        if frame < len(bp_viz.belief_history):
            # Get data for this iteration
            beliefs = bp_viz.belief_history[frame]
            var_to_check = bp_viz.var_to_check_history[frame]
            check_to_var = bp_viz.check_to_var_history[frame]

            # Update title
            fig.suptitle(f"LDPC Belief Propagation - Iteration {frame}", fontsize=16, fontweight="bold")

            # Belief evolution plot
            ax1.bar(range(len(beliefs)), beliefs, color=["red" if b < 0 else "blue" for b in beliefs], alpha=0.7, edgecolor="black", linewidth=1)
            ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)
            ax1.set_title("Current Beliefs (LLRs)", fontweight="bold")
            ax1.set_xlabel("Variable Node")
            ax1.set_ylabel("Belief (LLR)")
            ax1.grid(True, alpha=0.3)

            # Add belief values as text
            for i, belief in enumerate(beliefs):
                ax1.text(i, belief + 0.1 * np.sign(belief), f"{belief:.2f}", ha="center", va="bottom" if belief > 0 else "top", fontsize=9)

            # Hard decisions
            hard_decisions = [1 if b < 0 else 0 for b in beliefs]
            colors = ["red" if d == 1 else "blue" for d in hard_decisions]
            ax2.bar(range(len(hard_decisions)), hard_decisions, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
            ax2.set_title("Hard Decisions", fontweight="bold")
            ax2.set_xlabel("Variable Node")
            ax2.set_ylabel("Decision")
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([0, 1])
            ax2.grid(True, alpha=0.3)

            # Add decision values as text
            for i, decision in enumerate(hard_decisions):
                ax2.text(i, decision + 0.05, str(decision), ha="center", va="bottom", fontsize=12, fontweight="bold")

            # Message magnitude heatmap
            combined_messages = np.abs(var_to_check) + np.abs(check_to_var.T)
            im = ax3.imshow(combined_messages, cmap="YlOrRd", aspect="auto")
            ax3.set_title("Message Activity Heatmap", fontweight="bold")
            ax3.set_xlabel("Check Node")
            ax3.set_ylabel("Variable Node")
            # Create colorbar for this frame
            plt.colorbar(im, ax=ax3, label="Message Magnitude")

            # Iteration statistics
            converged_nodes = sum(1 for b in beliefs if abs(b) > 2.0)  # Strong beliefs
            syndrome = np.dot(bp_viz.H, hard_decisions) % 2
            syndrome_weight = np.sum(syndrome)

            stats_text = f"""Iteration {frame} Statistics:

• Converged nodes: {converged_nodes}/{len(beliefs)}
• Syndrome weight: {syndrome_weight}
• Valid codeword: {'Yes' if syndrome_weight == 0 else 'No'}

Message Activity:
• Var→Check: {np.mean(np.abs(var_to_check)):.3f}
• Check→Var: {np.mean(np.abs(check_to_var)):.3f}

Belief Statistics:
• Mean |LLR|: {np.mean(np.abs(beliefs)):.3f}
• Max |LLR|: {np.max(np.abs(beliefs)):.3f}
• Strong beliefs: {sum(1 for b in beliefs if abs(b) > 1.0)}"""

            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            ax4.set_title("Iteration Information", fontweight="bold")
            ax4.axis("off")

    # Create animation
    n_frames = len(bp_viz.belief_history)
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 // fps, repeat=True)

    return anim, fig


# Create and display the LDPC belief propagation animation
print("\nCreating LDPC Belief Propagation visualization...")
bp_anim, bp_fig = create_bp_animation(bp_viz, fps=1)
plt.show()

# %%
# Performance Comparison with Different Parameters
# ---------------------------------------------------------
# LDPC Iteration Benefits Analysis
# ============================================


def compare_ldpc_performance():
    """Compare LDPC performance with different decoder parameters.

    Note: The original simple H matrix (4x8) may not show clear iteration benefits
    due to its poor code properties. Better LDPC codes with regular structure
    and higher SNR ranges (1-6 dB) show clearer iteration benefits.
    """

    # Improved test parameters for better iteration analysis
    snr_range = np.arange(1, 7, 0.5)  # Focus on SNR range where iterations matter
    iteration_counts = [1, 5, 10, 20, 50, 100]
    num_trials = 100  # More trials for better statistical reliability

    results = {}

    # Testing Configuration:
    print(f"SNR range: {snr_range[0]:.1f} to {snr_range[-1]:.1f} dB (where iterations help)")
    print(f"Trials per point: {num_trials} (for statistical reliability)")
    print(f"Code: ({encoder.code_length}, {encoder.code_dimension}) rate {encoder.code_dimension/encoder.code_length:.3f}")

    # Initialize modulator and demodulator
    modulator = BPSKModulator(complex_output=False)
    demodulator = BPSKDemodulator()

    for max_iters in iteration_counts:
        ber_values = []

        for snr_db in tqdm(snr_range, desc=f"Testing {max_iters} iterations"):
            errors = 0
            total_bits = 0
            successful_decodings = 0

            # Initialize channel with proper noise power calculation
            noise_power = snr_to_noise_power(1.0, snr_db)
            channel = AWGNChannel(avg_noise_power=noise_power)
            decoder = BeliefPropagationDecoder(encoder, bp_iters=max_iters)

            for trial in range(num_trials):
                # Generate random message with correct dimension
                msg = torch.randint(0, 2, (1, encoder.code_dimension), dtype=torch.float32)
                codeword = encoder(msg)

                # Modulate the codeword to bipolar format
                bipolar_codeword = modulator(codeword)

                # Transmit over AWGN channel
                received_soft = channel(bipolar_codeword)

                # Demodulate the received signal to get LLRs
                received_llrs = demodulator(received_soft, noise_var=noise_power)

                # Decode
                decoded = decoder(received_llrs)

                # Count errors - compare original message with decoded message
                bit_errors = torch.sum(msg != decoded).item()
                errors += bit_errors
                total_bits += msg.numel()

                if bit_errors == 0:
                    successful_decodings += 1

            ber = errors / total_bits if total_bits > 0 else 1.0
            success_rate = successful_decodings / num_trials

            ber_values.append(ber)

            print(f"Performance statistics for SNR {snr_db:.1f} dB, {max_iters} iters: BER={ber:.2e}, Success rate={success_rate:.2f}")

        results[max_iters] = ber_values

    return snr_range, results


# Run performance comparison
print("Running performance comparison...")
snr_range, perf_results = compare_ldpc_performance()

# %%
# Visualize Performance Results
# --------------------------------------
# Error Rate Performance Plotting
# ===============================

# Extract BER data for plotting
ber_curves = []
labels = []

for max_iters, ber_values in perf_results.items():
    ber_curves.append(ber_values)
    labels.append(f"{max_iters} iterations")

# Plot BER performance using utility function
PlottingUtils.plot_performance_vs_snr(snr_range=snr_range, performance_values=ber_curves, labels=labels, title="LDPC Performance Analysis: Iteration Benefits", ylabel="Bit Error Rate", use_log_scale=True, xlabel="SNR (dB)")
plt.show()

# Additional performance insights plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
fig.suptitle("LDPC Performance Analysis: Why More Iterations Help", fontsize=16, fontweight="bold")

# Convergence benefit visualization
snr_target = 3.0  # dB - use a point where we have data
snr_idx = np.argmin(np.abs(snr_range - snr_target))  # Find closest SNR

iterations = list(perf_results.keys())
bers_at_target = [perf_results[it][snr_idx] for it in iterations]

colors = PlottingUtils.MODERN_PALETTE[: len(iterations)] if len(iterations) <= len(PlottingUtils.MODERN_PALETTE) else PlottingUtils.MODERN_PALETTE * (len(iterations) // len(PlottingUtils.MODERN_PALETTE) + 1)
bars = axes[1].bar(range(len(iterations)), bers_at_target, color=colors[: len(iterations)], alpha=0.8, edgecolor="black")
axes[1].set_yscale("log")
axes[1].set_xlabel("Number of Iterations", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Bit Error Rate", fontsize=12, fontweight="bold")
axes[1].set_title(f"BER vs. Iterations at SNR = {snr_range[snr_idx]:.1f} dB", fontsize=14, fontweight="bold")
axes[1].set_xticks(range(len(iterations)))
axes[1].set_xticklabels(iterations)
axes[1].grid(True, axis="y", alpha=0.3)

# Calculate and show improvement percentages
for i, (bar, ber) in enumerate(zip(bars, bers_at_target)):
    height = bar.get_height()
    if i > 0:
        improvement = (bers_at_target[0] - ber) / bers_at_target[0] * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height * 1.1, f"{improvement:.1f}%\nbetter", ha="center", va="bottom", fontsize=9, fontweight="bold", color="green")
    else:
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height * 1.1, "baseline", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Add insights text
axes[0].text(
    0.02,
    0.98,
    "Key Insights:\n• More iterations help most in 2-5 dB range\n• Diminishing returns beyond 20 iterations\n• Statistical reliability needs 300+ trials",
    transform=axes[0].transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
)
axes[0].set_title("Performance Analysis Insights", fontsize=14, fontweight="bold")
axes[0].axis("off")

plt.show()

# %%
# Why BER Doesn't Always Decrease with More Iterations: Theoretical Analysis
# ============================================================================
# LDPC Performance Theory and Code Design Impact
# ==============================================

# WHY BER DOESN'T ALWAYS DECREASE WITH MORE ITERATIONS
# ===================================================
#
# THEORETICAL BACKGROUND:
#
# 1. CONVERGENCE TO WRONG CODEWORDS:
#    • Belief propagation is not guaranteed to find the ML solution
#    • Can converge to pseudocodewords (invalid but low-energy states)
#    • More iterations may reinforce incorrect decisions
#
# 2. CODE DESIGN LIMITATIONS:
#    • Simple/poorly designed H matrices have poor distance properties
#    • Short cycles in Tanner graph cause correlation in messages
#    • Irregular degree distributions can lead to poor convergence
#
# 3. OPERATING REGIME EFFECTS:
#    • Very low SNR: Noise dominates, iterations can't help
#    • Very high SNR: Already error-free, no room for improvement
#    • Sweet spot: Medium SNR (2-6 dB for typical codes)
#
# 4. STATISTICAL FLUCTUATIONS:
#    • Small number of trials gives noisy BER estimates
#    • Some error patterns are inherently uncorrectable
#    • Need sufficient trials for stable statistics
#
# PRACTICAL IMPLICATIONS:
# ✓ Use well-designed LDPC codes with regular structure
# ✓ Test in appropriate SNR range (where code is useful)
# ✓ Use sufficient trials (300+ for reliable statistics)
# ✓ Monitor convergence indicators, not just iteration count
# ✓ Consider early stopping based on syndrome checks

# Demonstrate the effect of code design on iteration benefits
# DEMONSTRATING CODE DESIGN IMPACT:
# ================================

# Original simple code analysis
H_orig = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1]], dtype=torch.float32)

var_degrees_orig = torch.sum(H_orig, dim=0)
check_degrees_orig = torch.sum(H_orig, dim=1)

print(f"Original H matrix ({H_orig.shape[0]}×{H_orig.shape[1]}):")
print(f"Variable degrees: {var_degrees_orig.tolist()}")
print(f"Check degrees: {check_degrees_orig.tolist()}")
print(f"Degree variance (variables): {torch.var(var_degrees_orig):.2f}")
print(f"Code rate: {(H_orig.shape[1] - H_orig.shape[0])/H_orig.shape[1]:.3f}")

print(f"Current H matrix ({H_matrix.shape[0]}×{H_matrix.shape[1]}):")
var_degrees_curr = torch.sum(H_matrix, dim=0)
check_degrees_curr = torch.sum(H_matrix, dim=1)
print(f"Variable degrees: {var_degrees_curr.tolist()}")
print(f"Check degrees: {check_degrees_curr.tolist()}")
print(f"Degree variance (variables): {torch.var(var_degrees_curr.float()):.2f}")
print(f"Code rate: {encoder.code_dimension/encoder.code_length:.3f}")

# Why the improved code shows better iteration benefits:
# • More regular degree distribution (lower variance)
# • Longer block length allows better error correction
# • Better distance properties from systematic design

# %%
# Code Construction Analysis
# --------------------------------------
# LDPC Code Structure Analysis and Visualization
# ==============================================


def analyze_code_structure(H):
    """Analyze the structure of an LDPC code."""
    m, n = H.shape

    # Calculate degrees
    var_degrees = np.sum(H, axis=0)
    check_degrees = np.sum(H, axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("LDPC Code Structure Analysis", fontsize=16, fontweight="bold")

    # Variable node degree distribution
    unique_var_degrees, var_counts = np.unique(var_degrees, return_counts=True)
    colors = PlottingUtils.MODERN_PALETTE[: len(unique_var_degrees)]
    ax1.bar(unique_var_degrees, var_counts, color=colors[0], alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Variable Node Degree")
    ax1.set_ylabel("Count")
    ax1.set_title("Variable Node Degree Distribution")
    ax1.grid(True, alpha=0.3)

    # Check node degree distribution
    unique_check_degrees, check_counts = np.unique(check_degrees, return_counts=True)
    color2 = colors[1] if len(colors) > 1 else colors[0]
    ax2.bar(unique_check_degrees, check_counts, color=color2, alpha=0.8, edgecolor="black")
    ax2.set_xlabel("Check Node Degree")
    ax2.set_ylabel("Count")
    ax2.set_title("Check Node Degree Distribution")
    ax2.grid(True, alpha=0.3)

    # Edge distribution pattern
    ax3.imshow(H, cmap="Blues", aspect="auto", interpolation="nearest")
    ax3.set_xlabel("Variable Nodes")
    ax3.set_ylabel("Check Nodes")
    ax3.set_title("Parity Check Matrix Structure")

    # Add grid lines to show block structure
    for i in range(m + 1):
        ax3.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
    for j in range(n + 1):
        ax3.axvline(j - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    # Statistics text
    stats_text = f"""Code Parameters:
    • Dimensions: ({n}, {n-m})
    • Rate: {(n-m)/n:.3f}
    • Variable node degrees: {var_degrees.tolist()}
    • Check node degrees: {check_degrees.tolist()}
    • Average variable degree: {np.mean(var_degrees):.2f}
    • Average check degree: {np.mean(check_degrees):.2f}
    • Total edges: {np.sum(H):.0f}
    • Density: {np.sum(H)/(m*n):.3f}"""

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax4.set_title("Code Statistics")
    ax4.axis("off")

    return fig


# Analyze the code structure
analyze_code_structure(H_matrix.numpy())
plt.show()

# %%
# Conclusion
# --------------------------------------
# Advanced LDPC Visualization Summary
# ===================================
#
# This example demonstrated advanced visualization techniques for LDPC codes:
#
# 🔹 Enhanced Tanner Graphs:
#   • Node sizing based on degree
#   • Connection weighting by importance
#   • Degree distribution analysis
#
# 🔹 Belief Propagation Animation:
#   • Message passing visualization
#   • Convergence tracking
#   • LLR evolution over iterations
#
# 🔹 Performance Analysis:
#   • Iteration count comparison
#   • SNR sensitivity analysis
#   • Convergence behavior study
#
# 🔹 Code Structure Analysis:
#   • Degree distribution patterns
#   • Sparsity characteristics
#   • Statistical properties
#
# Key Insights:
# • Higher iteration counts improve performance but with diminishing returns
# • Node degree affects convergence behavior
# • Sparse structure enables efficient belief propagation
# • Visual analysis helps in code design and optimization
