"""
========================================================
Syndrome Decoding Visualization
========================================================

This example demonstrates syndrome decoding, a key technique in forward error
correction (FEC) that efficiently corrects errors using a parity-check matrix.
We'll visualize the syndrome computation and the error correction process with
animated, interactive graphics.
"""

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import CirclePolygon, FancyArrowPatch, Rectangle

from kaira.channels import BinarySymmetricChannel
from kaira.models.fec.encoders import HammingCodeEncoder

# %%
# Setting Up Visualization Environment
# ----------------------------------------------------------------------------------
# Let's set up our visualization environment with modern styling and a custom
# color palette for clearer and more engaging visualizations.

torch.manual_seed(42)
np.random.seed(42)

# Configure better visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Create custom colormaps for our visualizations
correct_bit_color = "#2ecc71"  # Green
error_bit_color = "#e74c3c"  # Red
highlight_color = "#f39c12"  # Orange
syndrome_color = "#9b59b6"  # Purple
background_color = "#f8f9fa"  # Light gray background

# Define a modern color palette
modern_palette = sns.color_palette("viridis", 8)
accent_palette = sns.color_palette("Set2", 8)

# %%
# Introducing Syndrome Decoding
# ---------------------------------------------------------------
# Syndrome decoding is an efficient decoding technique used in linear block codes
# like Hamming codes, BCH codes, and Reed-Solomon codes. It uses the syndrome - a
# compact representation of error patterns - to correct errors without checking all
# possible codewords.
#
# Let's create a basic (7,4) Hamming code to demonstrate syndrome decoding:

# Create a (7,4) Hamming code
code = HammingCodeEncoder(mu=3)  # mu=3 creates a (7,4) Hamming code

# Display the generator and parity-check matrices
G = code.generator_matrix
H = code.check_matrix  # HammingCodeEncoder uses check_matrix instead of parity_check_matrix

print(f"Generator matrix G (shape {G.shape}):")
print(G)
print("\nParity-check matrix H (shape {H.shape}):")
print(H)
print("\nVerify that G·H^T = 0 (modulo 2):")
print((G @ H.T) % 2)

# %%
# Visualizing the Hamming Code Matrix Structure
# -------------------------------------------------------------------------------------------------------
# Let's create a visualization of the generator and parity-check matrices to better
# understand their structure and relationship:

fig = plt.figure(figsize=(14, 10), facecolor=background_color)
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)

# Plot the generator matrix G
ax1 = fig.add_subplot(gs[0], facecolor=background_color)
G_np = G.numpy()
ax1.matshow(G_np, cmap="Blues", alpha=0.8)

# Add grid lines and cell values with improved styling
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        color = "white" if G_np[i, j] > 0.5 else "black"
        txt = ax1.text(j, i, str(int(G_np[i, j])), ha="center", va="center", fontsize=16, fontweight="bold", color=color)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black" if color == "white" else "white")])

# Add row and column labels with better styling
for i in range(G.shape[0]):
    ax1.text(-0.7, i, f"Row {i}", ha="center", va="center", fontsize=12, fontweight="bold", color=modern_palette[0])
for j in range(G.shape[1]):
    ax1.text(j, -0.7, f"Col {j}", ha="center", va="center", fontsize=12, fontweight="bold", color=modern_palette[0])

# Add explanation of the matrix structure
ax1.text(
    G.shape[1] + 0.5,
    G.shape[0] / 2,
    "Generator Matrix Structure:\n" "- Identity matrix (I) on the left\n" "- Parity submatrix (P) on the right\n" "- Form: G = [I | P]\n" "- Used for encoding: c = m·G",
    ha="left",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
)

# Add colored rectangles to show the structure with animation-like styling
identity_box = Rectangle((-0.5, -0.5), 4, 4, linewidth=3, edgecolor=modern_palette[0], facecolor="none", alpha=0.8, linestyle="-", zorder=10)
ax1.add_patch(identity_box)
ax1.text(1.5, -1.2, "Identity (I)", ha="center", fontsize=14, color=modern_palette[0], fontweight="bold")

parity_box = Rectangle((3.5, -0.5), 3, 4, linewidth=3, edgecolor=modern_palette[2], facecolor="none", alpha=0.8, linestyle="-", zorder=10)
ax1.add_patch(parity_box)
ax1.text(5, -1.2, "Parity (P)", ha="center", fontsize=14, color=modern_palette[2], fontweight="bold")

ax1.set_title("Generator Matrix G", fontsize=18, fontweight="bold")
ax1.set_xticks(np.arange(G.shape[1]))
ax1.set_yticks(np.arange(G.shape[0]))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xlim(-1, G.shape[1] + 8)
ax1.set_ylim(G.shape[0] - 0.5, -1.5)

# Plot the parity-check matrix H
ax2 = fig.add_subplot(gs[1], facecolor=background_color)
H_np = H.numpy()
ax2.matshow(H_np, cmap="Purples", alpha=0.8)

# Add grid lines and cell values with improved styling
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        color = "white" if H_np[i, j] > 0.5 else "black"
        txt = ax2.text(j, i, str(int(H_np[i, j])), ha="center", va="center", fontsize=16, fontweight="bold", color=color)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black" if color == "white" else "white")])

# Add row and column labels with better styling
for i in range(H.shape[0]):
    ax2.text(-0.7, i, f"Row {i}", ha="center", va="center", fontsize=12, fontweight="bold", color=modern_palette[4])
for j in range(H.shape[1]):
    ax2.text(j, -0.7, f"Col {j}", ha="center", va="center", fontsize=12, fontweight="bold", color=modern_palette[4])

# Add explanation of the matrix structure
ax2.text(
    H.shape[1] + 0.5,
    H.shape[0] / 2,
    "Parity-Check Matrix Structure:\n" "- Transposed parity submatrix (P^T) on the left\n" "- Identity matrix (I) on the right\n" "- Form: H = [P^T | I]\n" "- Used for syndrome: s = r·H^T",
    ha="left",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
)

# Add colored rectangles to show the structure with animation-like styling
parity_trans_box = Rectangle((-0.5, -0.5), 4, 3, linewidth=3, edgecolor=modern_palette[6], facecolor="none", alpha=0.8, linestyle="-", zorder=10)
ax2.add_patch(parity_trans_box)
ax2.text(1.5, -1.2, "Parity Transposed (P^T)", ha="center", fontsize=14, color=modern_palette[6], fontweight="bold")

identity_box = Rectangle((3.5, -0.5), 3, 3, linewidth=3, edgecolor=modern_palette[4], facecolor="none", alpha=0.8, linestyle="-", zorder=10)
ax2.add_patch(identity_box)
ax2.text(5, -1.2, "Identity (I)", ha="center", fontsize=14, color=modern_palette[4], fontweight="bold")

# Add G·H^T = 0 verification
verification_text = "Verification: G·H^T = 0 (mod 2)\nThis ensures that valid codewords yield zero syndrome"
ax2.text(H.shape[1] + 0.5, H.shape[0] + 1, verification_text, ha="left", va="center", fontsize=14, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.4))

ax2.set_title("Parity-Check Matrix H", fontsize=18, fontweight="bold")
ax2.set_xticks(np.arange(H.shape[1]))
ax2.set_yticks(np.arange(H.shape[0]))
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xlim(-1, H.shape[1] + 8)
ax2.set_ylim(H.shape[0] - 0.5, -1.5)

# Add a title for the entire figure with modern styling
title = fig.suptitle("Hamming Code (7,4) Matrix Structure Visualization", fontsize=22, fontweight="bold", y=0.98)
title.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

# Add an arrow connecting the two matrices to show their relationship
arrow = FancyArrowPatch((G.shape[1] / 2, G.shape[0] + 0.5), (H.shape[1] / 2, -0.5), connectionstyle="arc3,rad=0.3", arrowstyle="fancy,head_length=10,head_width=10", color="black", linewidth=2, alpha=0.6, transform=fig.transFigure, figure=fig)
fig.patches.append(arrow)

fig.text(0.38, 0.5, "Relationship: G·H^T = 0", ha="center", va="center", fontsize=14, fontweight="bold", transform=fig.transFigure, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

# Replace tight_layout() with explicit figure adjustments
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4)
plt.show()

# %%
# Encoding and Channel Simulation
# -----------------------------------------------------------------------
# Now let's demonstrate the encoding process and simulate transmission over a
# noisy binary symmetric channel (BSC):

# Create a message to encode
message = torch.tensor([1, 0, 1, 1])
print(f"Original message: {message}")

# Encode the message
codeword = code(message)  # HammingCodeEncoder uses __call__ method for encoding
print(f"Encoded codeword: {codeword}")

# Simulate transmission over a noisy BSC channel with 15% error probability
channel = BinarySymmetricChannel(crossover_prob=0.15)
received = channel(codeword)
print(f"Received vector: {received}")

# Check for errors
errors = (codeword != received).int()
print(f"Error pattern: {errors}")
print(f"Number of errors: {errors.sum().item()}")

# %%
# Visualizing the Encoding and Channel Transmission
# ----------------------------------------------------------------------------------------------------------------
# Let's create a more informative visualization of the encoding and transmission process:

fig = plt.figure(figsize=(14, 8), facecolor=background_color)
gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 0.5, 1], hspace=0.4)

# Plot the original message
ax1 = fig.add_subplot(gs[0], facecolor=background_color)
message_colors = ["#e74c3c" if bit == 1 else "#3498db" for bit in message]
bars1 = ax1.bar(np.arange(len(message)), message.numpy(), color=message_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels
for i, v in enumerate(message):
    txt = ax1.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

# Add message bits labeling with better styling
for i in range(len(message)):
    circle = CirclePolygon((i, -0.15), 0.15, resolution=20, facecolor=modern_palette[0], alpha=0.7, edgecolor="black")
    ax1.add_patch(circle)
    ax1.text(i, -0.15, f"m{i}", ha="center", va="center", fontweight="bold", color="white")

ax1.set_title("Original Message (4 bits)", fontsize=16, fontweight="bold", color=modern_palette[0])
ax1.set_ylim(-0.3, 1.3)
ax1.set_yticks([0, 1])
ax1.set_xticks(np.arange(len(message)))
ax1.set_xlim(-0.5, len(message) - 0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Add encoding process visualization
ax2 = fig.add_subplot(gs[1], facecolor=background_color)
ax2.axis("off")

# Create a fancy arrow to show the encoding process
arrow = FancyArrowPatch((len(message) / 2 - 0.5, 0), (len(codeword) / 2 - 0.5, 1), connectionstyle="arc3,rad=0.0", arrowstyle="fancy,head_length=10,head_width=10", color=modern_palette[2], linewidth=3, alpha=0.8, transform=ax2.transData)
ax2.add_patch(arrow)

encoding_text = "Encoding: c = m·G\nAdding 3 parity bits for error correction"
ax2.text(len(message) / 2 - 0.5, 0.5, encoding_text, ha="center", va="center", fontsize=14, fontweight="bold", color=modern_palette[2], bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

# Show the math operation
m_times_G = message.numpy().dot(G.numpy()) % 2
encoding_math = "Message × Generator matrix (mod 2):\n"
encoding_math += f"{message.numpy()} × G = {m_times_G} = {codeword.numpy()}"
ax2.text(len(codeword) - 1.5, 0.5, encoding_math, ha="center", va="center", fontsize=12, color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Plot the encoded codeword with improved styling
ax3 = fig.add_subplot(gs[2], facecolor=background_color)

# Different coloring for message and parity bits
codeword_colors = []
for i, bit in enumerate(codeword):
    if i < len(message):
        # Message bits - use same colors as original
        codeword_colors.append(message_colors[i])
    else:
        # Parity bits - use a different color palette
        codeword_colors.append("#27ae60" if bit == 1 else "#2ecc71")

bars3 = ax3.bar(np.arange(len(codeword)), codeword.numpy(), color=codeword_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with improved styling
for i, v in enumerate(codeword):
    txt = ax3.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

# Add codeword bits labeling with better styling
for i in range(len(codeword)):
    if i < len(message):
        # Information bits
        circle = CirclePolygon((i, -0.15), 0.15, resolution=20, facecolor=modern_palette[0], alpha=0.7, edgecolor="black")
        ax3.add_patch(circle)
        ax3.text(i, -0.15, f"c{i}", ha="center", va="center", fontweight="bold", color="white")
    else:
        # Parity bits
        circle = CirclePolygon((i, -0.15), 0.15, resolution=20, facecolor=modern_palette[2], alpha=0.7, edgecolor="black")
        ax3.add_patch(circle)
        ax3.text(i, -0.15, f"p{i-len(message)}", ha="center", va="center", fontweight="bold", color="white")

# Add rectangles to visually separate information and parity bits
info_rect = Rectangle((-0.5, -0.5), len(message), 2, linewidth=2, edgecolor=modern_palette[0], facecolor="none", alpha=0.5)
parity_rect = Rectangle((len(message) - 0.5, -0.5), len(codeword) - len(message), 2, linewidth=2, edgecolor=modern_palette[2], facecolor="none", alpha=0.5)
ax3.add_patch(info_rect)
ax3.add_patch(parity_rect)
ax3.text(len(message) / 2 - 0.5, -0.5, "Information Bits", ha="center", va="bottom", fontsize=12, color=modern_palette[0], fontweight="bold")
ax3.text(len(message) + (len(codeword) - len(message)) / 2 - 0.5, -0.5, "Parity Bits", ha="center", va="bottom", fontsize=12, color=modern_palette[2], fontweight="bold")

ax3.set_title("Encoded Codeword (7 bits)", fontsize=16, fontweight="bold")
ax3.set_ylim(-0.6, 1.3)
ax3.set_yticks([0, 1])
ax3.set_xticks(np.arange(len(codeword)))
ax3.set_xlim(-0.5, len(codeword) - 0.5)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Add a title for the entire figure with modern styling
title = fig.suptitle("Hamming Code (7,4) Encoding Process", fontsize=20, fontweight="bold", y=0.98)
title.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

# Use explicit figure adjustments instead of tight_layout
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4)
plt.show()

# %%
# Visualizing the Channel Transmission and Errors
# --------------------------------------------------------------------------------------------------------------
# Now let's visualize the channel transmission and resulting errors:

fig = plt.figure(figsize=(14, 10), facecolor=background_color)
gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 0.6, 1, 1], hspace=0.4)

# Plot the original codeword
ax1 = fig.add_subplot(gs[0], facecolor=background_color)
codeword_colors = []
for i, bit in enumerate(codeword):
    if i < len(message):
        # Information bits
        codeword_colors.append("#3498db" if bit == 0 else "#e74c3c")
    else:
        # Parity bits
        codeword_colors.append("#2ecc71" if bit == 0 else "#27ae60")

bars1 = ax1.bar(np.arange(len(codeword)), codeword.numpy(), color=codeword_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with better styling
for i, v in enumerate(codeword):
    txt = ax1.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

ax1.set_title("Transmitted Codeword", fontsize=16, fontweight="bold")
ax1.set_ylim(-0.3, 1.3)
ax1.set_yticks([0, 1])
ax1.set_xticks(np.arange(len(codeword)))
ax1.set_xlim(-0.5, len(codeword) - 0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Add channel simulation visualization
ax2 = fig.add_subplot(gs[1], facecolor=background_color)
ax2.axis("off")

# Create a fancy arrow to show the channel transmission process
arrow = FancyArrowPatch((len(codeword) / 2 - 0.5, 0), (len(received) / 2 - 0.5, 1), connectionstyle="arc3,rad=0.0", arrowstyle="fancy,head_length=10,head_width=10", color=error_bit_color, linewidth=3, alpha=0.8, transform=ax2.transData)
ax2.add_patch(arrow)

# Add noise symbols along the channel path
for i in range(len(codeword)):
    if codeword[i] != received[i]:
        # Add lightning bolt symbols for errors
        error_x: int = int(i)
        for j_val in np.linspace(0.2, 0.8, 3):
            ax2.text(error_x, float(j_val), "⚡", ha="center", va="center", fontsize=20, color=error_bit_color, path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
    else:
        # Add checkmark symbols for correct transmission
        ax2.text(i, 0.5, "✓", ha="center", va="center", fontsize=16, color=correct_bit_color, alpha=0.7, path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")])

channel_text = f"Binary Symmetric Channel (p = {channel.crossover_prob:.2f})"
ax2.text(len(codeword) / 2 - 0.5, 0.5, channel_text, ha="center", va="center", fontsize=14, fontweight="bold", color="black", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

# Plot the received vector
ax3 = fig.add_subplot(gs[2], facecolor=background_color)

# Color based on original value and error status
received_colors = []
for i, (orig, rec) in enumerate(zip(codeword, received)):
    if orig != rec:
        # Error - highlight in error color
        received_colors.append(error_bit_color)
    else:
        # No error - use same color as original
        if i < len(message):
            # Information bits
            received_colors.append("#3498db" if rec == 0 else "#e74c3c")
        else:
            # Parity bits
            received_colors.append("#2ecc71" if rec == 0 else "#27ae60")

bars3 = ax3.bar(np.arange(len(received)), received.numpy(), color=received_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with better styling
for i, v in enumerate(received):
    txt = ax3.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    # Add error indicators
    if codeword[i] != received[i]:
        ax3.text(i, 1.15, "ERROR", ha="center", va="center", fontsize=10, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=error_bit_color, alpha=0.8))

ax3.set_title("Received Vector (with errors)", fontsize=16, fontweight="bold")
ax3.set_ylim(-0.3, 1.3)
ax3.set_yticks([0, 1])
ax3.set_xticks(np.arange(len(received)))
ax3.set_xlim(-0.5, len(received) - 0.5)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Plot the error pattern
ax4 = fig.add_subplot(gs[3], facecolor=background_color)
error_colors = [error_bit_color if err == 1 else "#95a5a6" for err in errors]
bars4 = ax4.bar(np.arange(len(errors)), errors.numpy(), color=error_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with better styling
for i, v in enumerate(errors):
    txt = ax4.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    # Add before/after indicators for errors
    if v == 1:
        ax4.text(i, -0.2, f"{int(codeword[i])}→{int(received[i])}", ha="center", fontsize=10, color=error_bit_color, fontweight="bold")

# Add error summary
num_errors = errors.sum().item()
error_text = f"Number of errors: {num_errors}/{len(errors)}"
ax4.text(len(errors) - 1.5, 0.5, error_text, ha="center", va="center", fontsize=14, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

ax4.set_title("Error Pattern", fontsize=16, fontweight="bold", color=error_bit_color)
ax4.set_ylim(-0.3, 1.3)
ax4.set_yticks([0, 1])
ax4.set_xticks(np.arange(len(errors)))
ax4.set_xlim(-0.5, len(errors) - 0.5)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Add a title for the entire figure with modern styling
title = fig.suptitle("Channel Transmission Simulation", fontsize=20, fontweight="bold", y=0.98)
title.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

# Add explanation of the channel model
explanation = """Binary Symmetric Channel (BSC) Model:
• Each bit has probability p of being flipped (error)
• Errors occur independently for each bit
• In this simulation: p = 0.15 (15% chance of error)
• Hamming(7,4) code can correct up to 1 error bit

The error pattern shows which bits were flipped during transmission."""

fig.text(0.5, 0.01, explanation, ha="center", fontsize=12, bbox=dict(boxstyle="round4,pad=0.5", facecolor="wheat", alpha=0.5))

# Use explicit figure adjustments instead of tight_layout
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.4)
plt.show()

# %%
# Syndrome Decoding Process
# -----------------------------------------------------
# Now, let's compute the syndrome and perform decoding using the syndrome decoding approach:

# Compute the syndrome
syndrome = (received @ H.T) % 2
print(f"Syndrome: {syndrome}")

# Decode using syndrome
decoded_message, syndrome = code.inverse_encode(received)  # HammingCodeEncoder uses inverse_encode
decoded_codeword = code(decoded_message)  # Re-encode to get the full codeword

# Check if decoding was successful
success = torch.all(decoded_codeword == codeword)
print(f"Decoding successful: {success}")

# Extract the original message from the decoded codeword
print(f"Decoded message: {decoded_message}")

# %%
# Visualizing the Syndrome Computation and Decoding Process
# ------------------------------------------------------------------------------------------------------------------------------------
# Let's create a comprehensive visualization of the syndrome decoding process:

fig = plt.figure(figsize=(16, 14), facecolor=background_color)
gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1.2, 0.8, 1], hspace=0.5)

# Plot the received vector
ax1 = fig.add_subplot(gs[0], facecolor=background_color)

# Color based on original value and error status
received_colors = []
for i, (orig, rec) in enumerate(zip(codeword, received)):
    if orig != rec:
        # Error - highlight in error color
        received_colors.append(error_bit_color)
    else:
        # No error - use same color as original
        if i < len(message):
            # Information bits
            received_colors.append("#3498db" if rec == 0 else "#e74c3c")
        else:
            # Parity bits
            received_colors.append("#2ecc71" if rec == 0 else "#27ae60")

bars1 = ax1.bar(np.arange(len(received)), received.numpy(), color=received_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with improved styling
for i, v in enumerate(received):
    txt = ax1.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    # Add bit position labels with better styling
    ax1.text(i, -0.15, f"r{i}", ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.5))

ax1.set_title("Received Vector (with errors)", fontsize=16, fontweight="bold")
ax1.set_ylim(-0.3, 1.3)
ax1.set_yticks([0, 1])
ax1.set_xticks(np.arange(len(received)))
ax1.set_xlim(-0.5, len(received) - 0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Plot the syndrome computation process
ax2 = fig.add_subplot(gs[1], facecolor=background_color)

# Show the syndrome computation as a matrix operation
syndrome_np = syndrome.numpy()
H_T_np = H.T.numpy()
received_np = received.numpy().reshape(-1, 1)

# Show the matrix shapes
ax2.text(0, 0, f"r = {received_np.reshape(1, -1)}", fontsize=14)
ax2.text(0, 1, "H^T =", fontsize=14)

# Draw the H transpose matrix with improved styling
for i in range(H_T_np.shape[0]):
    for j in range(H_T_np.shape[1]):
        color = "#9b59b6" if H_T_np[i, j] == 1 else "#95a5a6"
        ax2.add_patch(Rectangle((2 + j * 0.5, 1 + i * 0.5), 0.5, 0.5, facecolor=color, edgecolor="black", alpha=0.8))
        ax2.text(2 + j * 0.5 + 0.25, 1 + i * 0.5 + 0.25, str(int(H_T_np[i, j])), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if H_T_np[i, j] == 1 else "black")

# Draw the syndrome computation
ax2.text(6, 2, "=", fontsize=18, fontweight="bold")

# Draw the syndrome vector with animation-like styling
for i in range(syndrome_np.shape[0]):
    color = "#f39c12" if syndrome_np[i] == 1 else "#95a5a6"
    ax2.add_patch(Rectangle((7, 1 + i * 0.5), 0.5, 0.5, facecolor=color, edgecolor="black", alpha=0.8))
    ax2.text(7 + 0.25, 1 + i * 0.5 + 0.25, str(int(syndrome_np[i])), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if syndrome_np[i] == 1 else "black")

# Add the syndrome formula
ax2.text(8, 2, f"= {syndrome.numpy()} (mod 2)", fontsize=14)
ax2.text(8, 3, "Syndrome calculation:", fontsize=14, fontweight="bold")

# Add the mathematical equation with detailed computation
syndrome_calc = ""
for i in range(H.shape[0]):
    row_calc = " ⊕ ".join([f"({received[j]}×{H[i,j]})" for j in range(len(received))])
    syndrome_calc += f"s{i} = {row_calc} = {syndrome[i]}\n"

ax2.text(8, 3.5, syndrome_calc, fontsize=12, va="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.6))

# Add title and explanation
ax2.set_title("Syndrome Computation: s = r·H^T (mod 2)", fontsize=16, fontweight="bold", color=syndrome_color)
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 6)
ax2.axis("off")

# Show the syndrome-based error correction process
ax3 = fig.add_subplot(gs[2], facecolor=background_color)

# Draw the syndrome lookup table
ax3.text(0, 1.5, "Syndrome Decoding Table:", fontsize=14, fontweight="bold")

# Create the syndrome lookup table with better styling
table_data = [
    ["Syndrome", "Error Pattern", "Error Position"],
    ["000", "[0 0 0 0 0 0 0]", "No errors"],
    ["001", "[0 0 0 0 0 0 1]", "Position 6"],
    ["010", "[0 0 0 0 0 1 0]", "Position 5"],
    ["011", "[0 0 0 0 0 1 1]", "Position ?"],
    ["100", "[0 0 0 0 1 0 0]", "Position 4"],
    ["101", "[0 0 0 0 1 0 1]", "Position ?"],
    ["110", "[0 0 0 0 1 1 0]", "Position ?"],
    ["111", "[0 0 0 1 0 0 0]", "Position 3"],
]

# Only showing a subset of the table for clarity
for i, row in enumerate(table_data[:6]):
    for j, cell in enumerate(row):
        # Highlight the syndrome that matches
        if j == 0 and i > 0 and "".join([str(int(s)) for s in syndrome]) == cell:
            cell_color = syndrome_color
            text_color = "white"
        elif j == 0:
            cell_color = "#95a5a6"
            text_color = "black"
        elif j == 1:
            cell_color = "#5dade2"
            text_color = "black"
        else:
            cell_color = "#ecf0f1"
            text_color = "black"

        ax3.add_patch(Rectangle((j * 2, 1.5 - i * 0.3), 2, 0.3, facecolor=cell_color, edgecolor="black", alpha=0.8))
        ax3.text(j * 2 + 1, 1.5 - i * 0.3 + 0.15, cell, ha="center", va="center", fontsize=10, color=text_color)

# Show the syndrome lookup process with animation-like styling
syndrome_bin = "".join([str(int(s)) for s in syndrome])
for i, row in enumerate(table_data[1:6]):
    if row[0] == syndrome_bin:
        # Draw arrow to matching syndrome
        arrow = FancyArrowPatch((8, 0.9), (0.5, 1.5 - i * 0.3 - 0.15), connectionstyle="arc3,rad=-0.3", arrowstyle="->", color=syndrome_color, linewidth=2, alpha=0.8)
        ax3.add_patch(arrow)

        # Show the correction process
        correction_text = f"Found syndrome {syndrome_bin} in table\n"
        correction_text += f"Identified error pattern: {row[1]}\n"
        correction_text += f"Error position: {row[2]}"

        ax3.text(9, 0.9, correction_text, fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.6))

# Add an explanation of syndrome decoding
explanation = """Syndrome Decoding Process:
1. Calculate syndrome s = r·H^T (mod 2)
2. Look up syndrome in table to find most likely error pattern
3. Correct received vector by XORing with error pattern
4. Extract original message from corrected codeword"""

ax3.text(0, 0, explanation, fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

ax3.set_title("Error Pattern Identification using Syndrome", fontsize=16, fontweight="bold")
ax3.set_xlim(0, 15)
ax3.set_ylim(0, 2)
ax3.axis("off")

# Plot the decoded result
ax4 = fig.add_subplot(gs[3], facecolor=background_color)

# First, show the decoded codeword
codeword_colors = []
for i, bit in enumerate(decoded_codeword):
    if i < len(message):
        # Information bits
        codeword_colors.append("#3498db" if bit == 0 else "#e74c3c")
    else:
        # Parity bits
        codeword_colors.append("#2ecc71" if bit == 0 else "#27ae60")

bars_decoded = ax4.bar(np.arange(len(decoded_codeword)), decoded_codeword.numpy(), color=codeword_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

# Add value labels with improved styling
for i, v in enumerate(decoded_codeword):
    txt = ax4.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

    # Highlight corrected bits
    if received[i] != decoded_codeword[i]:
        ax4.text(i, 1.15, "FIXED", ha="center", va="center", fontsize=10, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=correct_bit_color, alpha=0.8))

# Highlight the extracted message with a rectangle
message_rect = Rectangle((-0.5, -0.5), len(message), 2, linewidth=2, edgecolor=modern_palette[0], facecolor="none", alpha=0.8)
ax4.add_patch(message_rect)
ax4.text(len(message) / 2 - 0.5, -0.5, "Decoded Message", ha="center", va="bottom", fontsize=14, color=modern_palette[0], fontweight="bold")

# Add success indicator
if success:
    ax4.text(len(decoded_codeword) - 1.5, 0.5, "Decoding Successful!", ha="center", va="center", fontsize=14, fontweight="bold", color=correct_bit_color, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
else:
    ax4.text(len(decoded_codeword) - 1.5, 0.5, "Decoding Failed!", ha="center", va="center", fontsize=14, fontweight="bold", color=error_bit_color, bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.3))

ax4.set_title("Decoded Result", fontsize=16, fontweight="bold")
ax4.set_ylim(-0.6, 1.3)
ax4.set_yticks([0, 1])
ax4.set_xticks(np.arange(len(decoded_codeword)))
ax4.set_xlim(-0.5, len(decoded_codeword) - 0.5)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Add a title for the entire figure with modern styling
title = fig.suptitle("Syndrome Decoding Process for Hamming Code", fontsize=22, fontweight="bold", y=0.98)
title.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

# Add a comprehensive explanation of syndrome decoding
explanation = """Syndrome Decoding in Hamming Codes:

Key Concepts:
• Syndrome (s) uniquely identifies error patterns with up to 1 bit error
• Zero syndrome (all 0s) indicates no errors, or undetectable errors
• In (7,4) Hamming code, the syndrome has 3 bits, identifying 7 possible error positions
• The syndrome is computed as s = r·H^T (mod 2)

Advantages:
• Efficient decoding with O(n) complexity instead of comparing with all 2^k codewords
• Directly identifies error positions without exhaustive search
• Can correct up to ⌊(d-1)/2⌋ errors, where d is the minimum distance

Hamming(7,4) can correct any single-bit error, but will fail if there are 2 or more errors."""

fig.text(0.5, 0.01, explanation, ha="center", fontsize=12, bbox=dict(boxstyle="round4,pad=0.7", facecolor="wheat", alpha=0.6))

# Use explicit figure adjustments instead of tight_layout
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.5)
plt.show()

# %%
# Testing Error Correction Capabilities
# ---------------------------------------------------------------------------------
# Let's simulate multiple transmissions with different error patterns to visualize
# the error correction capabilities of the Hamming code:


# Create a function to simulate transmission and decoding
def simulate_transmission(code, message, error_pattern=None, p=0.15):
    """Simulate transmission with a specific error pattern or random errors."""
    # Encode the message
    codeword = code(message)  # Using __call__ method for encoding

    if error_pattern is not None:
        # Apply the specified error pattern
        received = (codeword + error_pattern) % 2
    else:
        # Random channel errors
        channel = BinarySymmetricChannel(crossover_prob=p)
        received = channel(codeword)

    # Compute actual error pattern
    errors = (received != codeword).int()

    # Decode the received vector
    decoded_message, _ = code.inverse_encode(received)  # Using inverse_encode for decoding
    decoded = code(decoded_message)  # Re-encode to get the full codeword

    # Check if decoding was successful
    success = torch.all(decoded == codeword)

    return {"message": message, "codeword": codeword, "received": received, "errors": errors, "num_errors": errors.sum().item(), "decoded": decoded, "success": success}


# Create test cases with different error patterns
message = torch.tensor([1, 0, 1, 1])
test_cases = [
    # No errors
    {"name": "No Errors", "error_pattern": torch.zeros(7)},
    # Single-bit errors (should all be corrected)
    {"name": "1 Error (bit 0)", "error_pattern": torch.tensor([1, 0, 0, 0, 0, 0, 0])},
    {"name": "1 Error (bit 3)", "error_pattern": torch.tensor([0, 0, 0, 1, 0, 0, 0])},
    {"name": "1 Error (bit 6)", "error_pattern": torch.tensor([0, 0, 0, 0, 0, 0, 1])},
    # Two-bit errors (will likely fail)
    {"name": "2 Errors", "error_pattern": torch.tensor([1, 0, 1, 0, 0, 0, 0])},
    # Three-bit errors (will likely fail)
    {"name": "3 Errors", "error_pattern": torch.tensor([1, 1, 1, 0, 0, 0, 0])},
]

# Run the simulations
results = []
for case in test_cases:
    result = simulate_transmission(code, message, case["error_pattern"])
    result["name"] = case["name"]
    results.append(result)

# %%
# Visualizing Error Correction Performance
# -------------------------------------------------------------------------------------------
# Let's create a comprehensive visualization showing the performance across
# different error patterns:

# Number of test cases to display
fig = plt.figure(figsize=(16, 4 * len(results)), facecolor=background_color)
gs = GridSpec(len(results), 3, figure=fig, width_ratios=[3, 3, 2], wspace=0.3, hspace=0.6)

for i, result in enumerate(results):
    # Get the data for this test case
    name = result["name"]
    errors = result["errors"]
    received = result["received"]
    decoded = result["decoded"]
    codeword = result["codeword"]
    success = result["success"]
    num_errors = result["num_errors"]

    # Plot the received vector with errors
    ax1 = fig.add_subplot(gs[i, 0], facecolor=background_color)

    # Color based on original value and error status
    received_colors = []
    for j, (orig, rec) in enumerate(zip(codeword, received)):
        if orig != rec:
            # Error - highlight in error color
            received_colors.append(error_bit_color)
        else:
            # No error - use same color as original
            received_colors.append("#95a5a6" if rec == 0 else "#b3b3b3")

    bars1 = ax1.bar(np.arange(len(received)), received.numpy(), color=received_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

    # Add value labels with improved styling
    for j, v in enumerate(received):
        txt = ax1.text(j, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

        # Add error indicators
        if errors[j] == 1:
            ax1.text(j, 1.15, "ERROR", ha="center", va="center", fontsize=10, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=error_bit_color, alpha=0.8))

    ax1.set_title(f"Received Vector: {name}", fontsize=16, fontweight="bold")
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_yticks([0, 1])
    ax1.set_xticks(np.arange(len(received)))
    ax1.set_xlim(-0.5, len(received) - 0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Plot the decoded codeword
    ax2 = fig.add_subplot(gs[i, 1], facecolor=background_color)

    # Colors for the decoded codeword
    decoded_colors = []
    for j, (rec, dec, orig) in enumerate(zip(received, decoded, codeword)):
        if rec != dec:
            # Bit was corrected
            decoded_colors.append(correct_bit_color)
        elif dec != orig:
            # Incorrect decoding
            decoded_colors.append(error_bit_color)
        else:
            # Correctly transmitted and unchanged
            decoded_colors.append("#95a5a6" if dec == 0 else "#b3b3b3")

    bars2 = ax2.bar(np.arange(len(decoded)), decoded.numpy(), color=decoded_colors, edgecolor="black", linewidth=1.5, alpha=0.8)

    # Add value labels with improved styling
    for j, v in enumerate(decoded):
        txt = ax2.text(j, v + 0.05, str(int(v)), ha="center", fontweight="bold", fontsize=14)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="white")])

        # Add correction indicators
        if received[j] != decoded[j]:
            ax2.text(j, 1.15, "FIXED", ha="center", va="center", fontsize=10, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=correct_bit_color, alpha=0.8))
        elif decoded[j] != codeword[j]:
            ax2.text(j, 1.15, "WRONG", ha="center", va="center", fontsize=10, color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=error_bit_color, alpha=0.8))

    # Highlight the message part
    message_rect = Rectangle((-0.5, -0.5), 4, 2, linewidth=2, edgecolor=modern_palette[0], facecolor="none", alpha=0.5)
    ax2.add_patch(message_rect)
    ax2.text(1.5, -0.5, "Message Bits", ha="center", va="bottom", fontsize=12, color=modern_palette[0], fontweight="bold")

    ax2.set_title("Decoded Vector", fontsize=16, fontweight="bold")
    ax2.set_ylim(-0.6, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_xticks(np.arange(len(decoded)))
    ax2.set_xlim(-0.5, len(decoded) - 0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Display decoding result summary
    ax3 = fig.add_subplot(gs[i, 2], facecolor=background_color)

    # Create a summary box with animation-like styling
    summary_text = []
    summary_text.append(f"Test Case: {name}")
    summary_text.append(f"Number of Errors: {num_errors}")

    # Compute the syndrome
    syndrome = (received @ H.T.to(received.dtype)) % 2
    syndrome_bin = "".join([str(int(s)) for s in syndrome])
    summary_text.append(f"Syndrome: {syndrome_bin}")

    # Success or failure message
    if success:
        summary_text.append("✓ Decoding Successful!")
        summary_text.append("All errors corrected")
        result_color = correct_bit_color
    else:
        summary_text.append("✗ Decoding Failed!")
        if num_errors > 1:
            summary_text.append(f"Too many errors ({num_errors}) to correct")
        else:
            summary_text.append("Unexpected decoding failure")
        result_color = error_bit_color

    # Draw a fancy box with the result
    box_height = 1.0
    box_y = 0.5
    result_box = Rectangle((0.5, box_y), 2, box_height, linewidth=3, edgecolor=result_color, facecolor="white", alpha=0.8, zorder=5)
    ax3.add_patch(result_box)

    # Add a header
    header_box = Rectangle((0.5, box_y + box_height), 2, 0.3, linewidth=0, facecolor=result_color, alpha=0.8, zorder=6)
    ax3.add_patch(header_box)
    ax3.text(1.5, box_y + box_height + 0.15, "Decoding Result", ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=7)

    # Add the summary text with better styling
    for j, text in enumerate(summary_text):
        color = result_color if j >= len(summary_text) - 2 else "black"
        fontweight = "bold" if j >= len(summary_text) - 2 else "normal"
        fontsize = 12 if j >= len(summary_text) - 2 else 11

        ax3.text(1.5, box_y + 0.8 - j * 0.15, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight=fontweight, zorder=8)

    # Add an explanatory note about error correction capability
    if num_errors <= 1:
        explanation = "Hamming(7,4) can correct 1 error"
    else:
        explanation = "Hamming(7,4) cannot correct 2+ errors"

    ax3.text(1.5, 0.1, explanation, ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax3.axis("off")

# Add a title for the entire figure with modern styling
title = fig.suptitle("Error Correction Performance of Hamming Code (7,4)", fontsize=22, fontweight="bold", y=0.98)
title.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

# Add a comprehensive explanation of error correction capabilities
explanation = """Hamming Code Error Correction Performance:

• Hamming(7,4) has minimum distance d = 3
• Can detect up to 2 errors and correct 1 error: t = ⌊(d-1)/2⌋ = ⌊(3-1)/2⌋ = 1
• With 0 or 1 errors: Guaranteed successful decoding
• With 2+ errors: Decoding will likely fail

Error correction process:
1. Calculate syndrome s = r·H^T
2. If syndrome is all zeros, assume no errors
3. Otherwise, use the syndrome to locate the error position
4. Flip the bit at the identified error position

This visualization demonstrates that Hamming codes are powerful for protecting against
single-bit errors but fail when multiple errors occur."""

fig.text(0.5, 0.01, explanation, ha="center", fontsize=12, bbox=dict(boxstyle="round4,pad=0.7", facecolor="wheat", alpha=0.6))

# Use explicit figure adjustments instead of tight_layout
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.6, wspace=0.3)
plt.show()

# %%
# Conclusion
# ---------------------
# In this tutorial, we've demonstrated syndrome decoding for Hamming codes with
# comprehensive visualizations:
#
# Key points:
# - Syndrome decoding provides an efficient way to decode linear block codes
# - The syndrome uniquely identifies error patterns with up to 1 error in Hamming codes
# - Hamming(7,4) code can correct any single-bit error but fails with 2+ errors
# - The decoding process uses matrix operations to compute syndrome and identify errors
#
# Syndrome decoding forms the foundation for more complex decoding algorithms in
# more powerful FEC codes like BCH and Reed-Solomon codes.
#
# References:
# - :cite:`lin2004error` - Comprehensive coverage of syndrome decoding
# - :cite:`moon2005error` - Mathematical foundations of syndrome computation
# - :cite:`mackay2003information` - Information theory perspective on decoding
