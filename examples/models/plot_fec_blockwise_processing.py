"""
====================================================================
Block-wise Processing for FEC
====================================================================

This example demonstrates how to perform block-wise processing of data for forward
error correction (FEC) using the `apply_blockwise` utility function. Block-wise
processing is essential in many coding schemes like block codes, systematic codes,
and interleaved coding.
"""

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle

from kaira.models.fec.utils import apply_blockwise

# %%
# Setting up
# ----------------------
# First, we set a random seed to ensure reproducibility of results.

torch.manual_seed(42)
np.random.seed(42)

# Configure visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
if hasattr(sns, "set_context"):
    sns.set_context("notebook", font_scale=1.2)

# Create custom colormaps for our visualizations
error_cmap = LinearSegmentedColormap.from_list("ErrorMap", ["#ffffff", "#ff9999"])
correct_cmap = LinearSegmentedColormap.from_list("CorrectMap", ["#ffffff", "#99ff99"])

# Define colors for our visualizations
original_color = "#3498db"  # Blue
encoded_color = "#2ecc71"  # Green
error_color = "#e74c3c"  # Red
corrected_color = "#9b59b6"  # Purple
highlight_color = "#f39c12"  # Orange
parity_color = "#e67e22"  # Dark orange

# %%
# Block-wise Processing Fundamentals
# ---------------------------------------------------------------------------------
# In forward error correction (FEC), data is often processed in fixed-size blocks.
# The `apply_blockwise` function allows us to apply operations to these blocks
# while maintaining the original tensor structure.
#
# Let's start with a simple example: applying a NOT operation to blocks of binary data.

# Create a binary tensor
binary_data = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1])
block_size = 2

print(f"Original binary data: {binary_data}")

# Apply NOT operation (1 -> 0, 0 -> 1) to each block of size 2
inverted_data = apply_blockwise(binary_data, block_size, lambda block: 1 - block)

print(f"Data after block-wise NOT operation: {inverted_data}")

# %%
# Visualizing Block-wise Operations
# -------------------------------------------------------------------------
# We can visualize how the block-wise operation transforms the data:

plt.figure(figsize=(10, 4))

# Plot original data
plt.subplot(2, 1, 1)
plt.bar(np.arange(len(binary_data)), binary_data.numpy(), color="blue")
plt.title("Original Binary Data")
plt.yticks([0, 1])
plt.grid(axis="y")

# Add vertical lines to show block boundaries
for i in range(1, len(binary_data) // block_size):
    plt.axvline(x=i * block_size - 0.5, color="gray", linestyle="--", alpha=0.7)

# Plot inverted data
plt.subplot(2, 1, 2)
plt.bar(np.arange(len(inverted_data)), inverted_data.numpy(), color="orange")
plt.title("After Block-wise NOT Operation")
plt.yticks([0, 1])
plt.grid(axis="y")

# Add vertical lines to show block boundaries
for i in range(1, len(inverted_data) // block_size):
    plt.axvline(x=i * block_size - 0.5, color="gray", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
# Complex Block Operations
# ----------------------------------------------------
# Now, let's look at more complex operations that can be performed block-wise.
# For example, we might want to add parity bits to each block, which is a simple
# form of error detection.


def add_parity(block):
    """Add an even parity bit to each block."""
    # Calculate parity - sum of 1s modulo 2
    parity = torch.sum(block, dim=-1) % 2

    # Create a new tensor for the extended block (original + parity bit)
    extended_shape = list(block.shape)
    extended_shape[-1] += 1  # Add one dimension for the parity bit

    # Create the extended block
    extended_block = torch.zeros(extended_shape, dtype=block.dtype, device=block.device)

    # Copy the original data
    extended_block[..., :-1] = block

    # Set the parity bit (even parity: use 1 if sum is odd)
    extended_block[..., -1] = parity

    return extended_block


# Create a longer binary tensor
data = torch.randint(0, 2, (12,))
print(f"Original data: {data}")

# We can't use apply_blockwise directly here because our function changes the block size
# We need to reshape manually
block_size = 3
num_blocks = len(data) // block_size
reshaped_data = data.view(num_blocks, block_size)

# Add parity to each block
blocks_with_parity = torch.zeros((num_blocks, block_size + 1), dtype=data.dtype)
for i in range(num_blocks):
    blocks_with_parity[i] = add_parity(reshaped_data[i])

print(f"Data with parity bits (block-wise): {blocks_with_parity.view(-1)}")

# %%
# Error Detection with Parity
# -------------------------------------------------------------
# Now let's simulate some transmission errors and see how parity bits help
# detect them.


# Function to check parity
def check_parity(block):
    """Check if a block has even parity."""
    return torch.sum(block) % 2 == 0


# Introduce random errors
encoded_data = blocks_with_parity.view(-1)
error_positions = torch.randint(0, len(encoded_data), (2,))
corrupted_data = encoded_data.clone()

print(f"Original encoded data: {encoded_data}")
print(f"Introducing errors at positions: {error_positions}")

for pos in error_positions:
    corrupted_data[pos] = 1 - corrupted_data[pos]  # Flip the bit

print(f"Corrupted data: {corrupted_data}")

# Reshape into blocks for parity checking
block_size_with_parity = block_size + 1
corrupted_blocks = corrupted_data.view(-1, block_size_with_parity)

# Check parity for each block
for i, block in enumerate(corrupted_blocks):
    is_valid = check_parity(block)
    print(f"Block {i+1}: {block}, Parity valid: {is_valid}")

# %%
# Visualizing Error Detection
# -------------------------------------------------------------
# Let's visualize the original, encoded, and corrupted data, highlighting where
# errors were introduced and which blocks had parity violations.

plt.figure(figsize=(12, 6))

# Original data
plt.subplot(3, 1, 1)
plt.bar(np.arange(len(data)), data.numpy(), color="blue")
plt.title("Original Data")
plt.yticks([0, 1])
plt.grid(axis="y")

# Encoded data with parity
plt.subplot(3, 1, 2)
plt.bar(np.arange(len(encoded_data)), encoded_data.numpy(), color="green")
plt.title("Encoded Data with Parity Bits")
plt.yticks([0, 1])
plt.grid(axis="y")

# Add vertical lines to show block boundaries
for i in range(1, len(encoded_data) // block_size_with_parity):
    plt.axvline(x=i * block_size_with_parity - 0.5, color="gray", linestyle="--", alpha=0.7)

# Highlight parity bits
for i in range(num_blocks):
    pos = (i + 1) * block_size_with_parity - 1
    plt.scatter(pos, encoded_data[pos], color="purple", s=100, marker="o", zorder=3)

# Corrupted data
plt.subplot(3, 1, 3)
plt.bar(np.arange(len(corrupted_data)), corrupted_data.numpy(), color="orange")
plt.title("Corrupted Data with Errors")
plt.yticks([0, 1])
plt.grid(axis="y")

# Add vertical lines to show block boundaries
for i in range(1, len(corrupted_data) // block_size_with_parity):
    plt.axvline(x=i * block_size_with_parity - 0.5, color="gray", linestyle="--", alpha=0.7)

# Highlight parity bits
for i in range(num_blocks):
    pos = (i + 1) * block_size_with_parity - 1
    plt.scatter(pos, corrupted_data[pos], color="purple", s=100, marker="o", zorder=3)

# Highlight errors
for pos in error_positions:
    plt.scatter(pos, corrupted_data[pos], color="red", s=150, marker="x", zorder=3)

# Highlight blocks with parity errors
for i, block in enumerate(corrupted_blocks):
    if not check_parity(block):
        block_start = i * block_size_with_parity
        block_end = (i + 1) * block_size_with_parity - 1
        plt.axvspan(block_start - 0.5, block_end + 0.5, alpha=0.2, color="red")

plt.tight_layout()
plt.show()

# %%
# Using apply_blockwise for Multi-return Functions
# ---------------------------------------------------------------------------------------------------------------
# The `apply_blockwise` function can also handle functions that return multiple values.
# Let's demonstrate this with a function that returns both the processed block and
# a flag indicating if an error was detected.


def process_and_check(block):
    """Process a block and check for errors.

    Returns:
        tuple: (processed_block, error_detected)
    """
    # Simple processing: XOR with a fixed pattern
    pattern = torch.tensor([1, 0, 1, 0])
    processed = block ^ pattern[: block.size(-1)]

    # Check for a condition (e.g., at least two 1s)
    error_detected = torch.sum(block, dim=-1) < 2

    return processed, error_detected


# Create test data
test_data = torch.tensor(
    [
        [1, 1, 0, 0],  # Has two 1s
        [0, 0, 1, 0],  # Has one 1
        [1, 1, 1, 0],  # Has three 1s
        [0, 0, 0, 0],  # Has zero 1s
    ]
)

# Apply blockwise processing
processed_data, errors = apply_blockwise(test_data, 4, process_and_check)

print("Original blocks:")
for i, block in enumerate(test_data):
    print(f"Block {i+1}: {block}")

print("\nProcessed blocks and error flags:")
for i, (block, error) in enumerate(zip(processed_data, errors)):
    print(f"Block {i+1}: {block}, Error detected: {error}")

# %%
# Advanced Example: Systematic Encoding
# -----------------------------------------------------------------------------------
# Many FEC schemes use systematic encoding, where the original data is preserved
# and parity bits are added. Let's implement a simple (7,4) Hamming code using
# block-wise processing.
#
# The (7,4) Hamming code adds 3 parity bits to 4 data bits.


def hamming_encode(block):
    """Encode a 4-bit block using (7,4) Hamming code.

    The positions 0, 1, and 3 (0-indexed) are parity bits. The positions 2, 4, 5, and 6 contain the
    original data.
    """
    if block.size(-1) != 4:
        raise ValueError("Block must contain 4 bits")

    # Create the 7-bit codeword with zeros initially
    codeword = torch.zeros(7, dtype=block.dtype, device=block.device)

    # Place data bits at positions 2, 4, 5, 6
    codeword[2] = block[0]
    codeword[4] = block[1]
    codeword[5] = block[2]
    codeword[6] = block[3]

    # Calculate parity bits
    # P1 (position 0) checks bits at positions 2,4,6
    codeword[0] = (codeword[2] + codeword[4] + codeword[6]) % 2

    # P2 (position 1) checks bits at positions 2,5,6
    codeword[1] = (codeword[2] + codeword[5] + codeword[6]) % 2

    # P3 (position 3) checks bits at positions 4,5,6
    codeword[3] = (codeword[4] + codeword[5] + codeword[6]) % 2

    return codeword


# Create 4-bit data blocks
data_blocks = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])

# Encode each block manually since our function changes block size
encoded_blocks = torch.zeros((len(data_blocks), 7), dtype=data_blocks.dtype)
for i, block in enumerate(data_blocks):
    encoded_blocks[i] = hamming_encode(block)

print("Original 4-bit data blocks:")
for i, block in enumerate(data_blocks):
    print(f"Block {i+1}: {block}")

print("\nEncoded 7-bit Hamming codewords:")
for i, block in enumerate(encoded_blocks):
    print(f"Block {i+1}: {block}")

# %%
# Hamming Code Error Correction
# ---------------------------------------------------------------
# The Hamming (7,4) code can correct 1-bit errors per codeword. Let's demonstrate
# this by introducing errors and then correcting them.


def hamming_decode(codeword):
    """Decode a 7-bit Hamming codeword, correcting up to 1 error.

    Returns:
        tuple: (corrected_codeword, original_data, error_position)
    """
    # Calculate syndrome
    s1 = (codeword[0] + codeword[2] + codeword[4] + codeword[6]) % 2
    s2 = (codeword[1] + codeword[2] + codeword[5] + codeword[6]) % 2
    s3 = (codeword[3] + codeword[4] + codeword[5] + codeword[6]) % 2

    syndrome = s1 + 2 * s2 + 4 * s3

    # Correct error if syndrome is non-zero
    corrected = codeword.clone()
    if syndrome > 0:
        # The syndrome value indicates the position of the error (1-based)
        error_pos = syndrome - 1
        corrected[error_pos] = 1 - corrected[error_pos]  # Flip the bit
    else:
        error_pos = None

    # Extract the original data bits
    original_data = torch.tensor([corrected[2], corrected[4], corrected[5], corrected[6]])

    return corrected, original_data, error_pos


# Introduce single-bit errors
corrupted_blocks = encoded_blocks.clone()
error_positions = []

for i in range(len(corrupted_blocks)):
    # Randomly choose one position in each block to flip
    error_pos = torch.randint(0, 7, (1,)).item()
    error_positions.append(error_pos)
    corrupted_blocks[i, error_pos] = 1 - corrupted_blocks[i, error_pos]

# Decode and correct
results = []
for i, block in enumerate(corrupted_blocks):
    corrected, decoded_data, error_pos = hamming_decode(block)
    results.append((corrected, decoded_data, error_pos))

# Show results
print("Results of decoding corrupted codewords:")
for i, (corrected, decoded_data, detected_pos) in enumerate(results):
    original_block = data_blocks[i]
    corrupted_block = corrupted_blocks[i]
    actual_error_pos = error_positions[i]

    print(f"Block {i+1}:")
    print(f"  Original data: {original_block}")
    print(f"  Encoded: {encoded_blocks[i]}")
    print(f"  Corrupted: {corrupted_block} (error at position {actual_error_pos})")
    print(f"  Corrected: {corrected} (detected error at position {detected_pos})")
    print(f"  Decoded data: {decoded_data}")
    print(f"  Successful correction: {torch.all(decoded_data == original_block)}")
    print()

# %%
# Visualizing Hamming Code Error Correction
# ---------------------------------------------------------------------------------------------
# Now let's create an enhanced visualization of how Hamming codes work, showing the
# encoding, error introduction, and correction processes visually.

fig = plt.figure(figsize=(15, 12))
# Modify GridSpec to have more space between subplots
gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2], width_ratios=[4, 1], hspace=0.6, wspace=0.4, bottom=0.05, top=0.9, left=0.05, right=0.95)

# Create empty lists to store plot objects for animation
bit_rects: list[Rectangle] = []
arrows: list[dict] = []
syndrome_texts: list = []  # Fixed: removed incorrect type annotation

# Select one example block to visualize in detail
block_idx = 0  # We'll use the first block
original_data = data_blocks[block_idx]
encoded = encoded_blocks[block_idx]
corrupted = corrupted_blocks[block_idx]
corrected, decoded, error_pos = results[block_idx]
actual_error_pos = error_positions[block_idx]

# 1. Original data visualization
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Original 4-bit Data", fontsize=16, fontweight="bold")

# Plot data bits
for i, bit in enumerate(original_data):
    color = original_color if bit == 1 else "white"
    rect = ax1.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, edgecolor="black", linewidth=1.5))
    ax1.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if bit == 1 else "black")
    # Add bit labels
    ax1.text(i, -0.3, f"d{i+1}", ha="center", va="center", fontsize=12)

ax1.set_xlim(-0.8, len(original_data) - 0.2)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)

# Add an arrow pointing to the encoding step
ax1.annotate("Hamming Encoding", xy=(1.5, -1), xytext=(1.5, -0.6), ha="center", fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=2, color=highlight_color))

# 2. Hamming code visualization with parity calculation
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("(7,4) Hamming Encoded Data", fontsize=16, fontweight="bold")

# Plot the codeword bits with different colors for data and parity
for i, bit in enumerate(encoded):
    # Determine if this is a parity bit or data bit
    is_parity = i == 0 or i == 1 or i == 3

    # Choose color based on bit type and value
    if is_parity:
        color = parity_color if bit == 1 else "white"
        label = f"p{i+1}"
    else:
        color = encoded_color if bit == 1 else "white"
        # Map position to original data index
        if i == 2:
            data_idx = 0
        elif i == 4:
            data_idx = 1
        elif i == 5:
            data_idx = 2
        else:  # i == 6
            data_idx = 3
        label = f"d{data_idx+1}"

    # Draw rectangle for bit
    rect = ax2.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, edgecolor="black", linewidth=1.5))

    # Add bit value
    ax2.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if bit == 1 else "black")

    # Add bit labels
    ax2.text(i, -0.3, label, ha="center", va="center", fontsize=12)

# Add parity check information
# P1 checks bits 0, 2, 4, 6
positions_p1 = [0, 2, 4, 6]
for pos in positions_p1:
    if pos != 0:  # Don't highlight the parity bit itself
        circle = Circle((pos, 0.5), 0.45, fill=False, edgecolor="blue", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)

# P2 checks bits 1, 2, 5, 6
positions_p2 = [1, 2, 5, 6]
for pos in positions_p2:
    if pos != 1:  # Don't highlight the parity bit itself
        circle = Circle((pos, 0.5), 0.42, fill=False, edgecolor="green", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)

# P3 checks bits 3, 4, 5, 6
positions_p3 = [3, 4, 5, 6]
for pos in positions_p3:
    if pos != 3:  # Don't highlight the parity bit itself
        circle = Circle((pos, 0.5), 0.39, fill=False, edgecolor="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)

# Add parity check legends
ax2.text(-0.8, 0.5, "P1:", ha="right", va="center", fontsize=10, color="blue", fontweight="bold")
ax2.text(-0.8, 0.3, "P2:", ha="right", va="center", fontsize=10, color="green", fontweight="bold")
ax2.text(-0.8, 0.1, "P3:", ha="right", va="center", fontsize=10, color="red", fontweight="bold")

ax2.set_xlim(-0.8, len(encoded) - 0.2)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["left"].set_visible(False)

# Add an arrow pointing to the transmission
ax2.annotate("Transmission\nthrough noisy channel", xy=(3, -1), xytext=(3, -0.6), ha="center", fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=2, color=error_color))

# 3. Corrupted data visualization
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_title("Received Data (with error) and Correction", fontsize=16, fontweight="bold")

# Plot corrupted codeword
for i, bit in enumerate(corrupted):
    # Determine if this bit has an error
    has_error = i == actual_error_pos

    # Determine if this is a parity bit or data bit
    is_parity = i == 0 or i == 1 or i == 3

    # Choose color based on bit type, value, and error status
    if has_error:
        color = error_color
    else:
        if is_parity:
            color = parity_color if bit == 1 else "white"
        else:
            color = encoded_color if bit == 1 else "white"

    # Draw rectangle for bit
    rect = ax3.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, edgecolor="black", linewidth=1.5))

    # Add bit value
    ax3.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if (bit == 1 or has_error) else "black")

    # Mark error with a cross
    if has_error:
        ax3.text(i, 1.3, "✗", color=error_color, fontsize=20, ha="center", fontweight="bold", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
        ax3.text(i, 1.7, "Error", color=error_color, fontsize=12, ha="center", fontweight="bold")

# Calculate syndrome visually
s1 = (corrupted[0] + corrupted[2] + corrupted[4] + corrupted[6]) % 2
s2 = (corrupted[1] + corrupted[2] + corrupted[5] + corrupted[6]) % 2
s3 = (corrupted[3] + corrupted[4] + corrupted[5] + corrupted[6]) % 2
syndrome = s1 + 2 * s2 + 4 * s3

# Show syndrome calculation
if syndrome > 0:
    ax3.text(-0.8, -1.0, f"Syndrome: (S3 S2 S1) = ({int(s3)}{int(s2)}{int(s1)}) = {syndrome}", fontsize=12, fontweight="bold", bbox=dict(facecolor="lightyellow", alpha=0.7, boxstyle="round,pad=0.5", edgecolor="orange"))

    ax3.text(-0.8, -1.5, f"Error position: {syndrome-1}", fontsize=12, fontweight="bold", bbox=dict(facecolor="lightyellow", alpha=0.7, boxstyle="round,pad=0.5", edgecolor="orange"))
else:
    ax3.text(-0.8, -1.0, "No errors detected", fontsize=12, fontweight="bold", bbox=dict(facecolor="lightgreen", alpha=0.7, boxstyle="round,pad=0.5"))

# Show corrected codeword
ax3.text(-0.8, -2.0, "Corrected codeword:", fontsize=12, fontweight="bold")

# Plot corrected codeword
y_offset = -2.3
for i, bit in enumerate(corrected):
    # Determine if this was corrupted and corrected
    was_corrected = i == actual_error_pos

    # Determine if this is a parity bit or data bit
    is_parity = i == 0 or i == 1 or i == 3

    # Choose color based on bit type, value, and correction status
    if was_corrected:
        color = corrected_color
    else:
        if is_parity:
            color = parity_color if bit == 1 else "white"
        else:
            color = encoded_color if bit == 1 else "white"

    # Draw rectangle for bit
    rect = ax3.add_patch(Rectangle((i - 0.4, y_offset), 0.8, 1, facecolor=color, edgecolor="black", linewidth=1.5))

    # Add bit value
    ax3.text(i, y_offset + 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if (bit == 1 or was_corrected) else "black")

    # Mark corrected bit
    if was_corrected:
        ax3.text(i, y_offset + 1.3, "✓", color="green", fontsize=20, ha="center", fontweight="bold", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

ax3.set_xlim(-0.8, len(corrupted) - 0.2)
ax3.set_ylim(-2.8, 2.0)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.spines["left"].set_visible(False)

# 4. Sidebar for parity check explanations
ax_info = fig.add_subplot(gs[:, 1])
ax_info.set_title("Parity Check Matrix", fontsize=14, fontweight="bold")

# Show the parity check matrix for Hamming (7,4) code
check_matrix = """
P1: [1 0 1 0 1 0 1]
P2: [0 1 1 0 0 1 1]
P3: [0 0 0 1 1 1 1]
"""

ax_info.text(0.1, 0.8, "Parity Checks:", fontsize=12, fontweight="bold")
ax_info.text(0.1, 0.75, "P1 = d1 + d2 + d4", fontsize=10, color="blue")
ax_info.text(0.1, 0.7, "P2 = d1 + d3 + d4", fontsize=10, color="green")
ax_info.text(0.1, 0.65, "P3 = d2 + d3 + d4", fontsize=10, color="red")

# Add syndrome table
ax_info.text(0.1, 0.5, "Syndrome Lookup Table:", fontsize=12, fontweight="bold")
syndrome_table = """
000: No error
001: Error in P1
010: Error in P2
011: Error in d1
100: Error in P3
101: Error in d2
110: Error in d3
111: Error in d4
"""
ax_info.text(0.1, 0.45, syndrome_table, fontsize=9, fontfamily="monospace")

# Add explanation of Hamming code capabilities
ax_info.text(0.1, 0.15, "Hamming Code Properties:", fontsize=12, fontweight="bold")
ax_info.text(0.1, 0.1, "• Can detect up to 2 errors", fontsize=10)
ax_info.text(0.1, 0.05, "• Can correct 1 error", fontsize=10)
ax_info.text(0.1, 0.0, "• Efficient: only 3 parity", fontsize=10)
ax_info.text(0.1, -0.05, "  bits for 4 data bits", fontsize=10)

ax_info.set_xlim(0, 1)
ax_info.set_ylim(-0.1, 1)
ax_info.set_xticks([])
ax_info.set_yticks([])
ax_info.spines["top"].set_visible(False)
ax_info.spines["right"].set_visible(False)
ax_info.spines["bottom"].set_visible(False)
ax_info.spines["left"].set_visible(False)

# Add a title for the figure
fig.suptitle("Hamming (7,4) Code: Encoding, Error Detection, and Correction", fontsize=18, fontweight="bold", y=0.98)

# Replace tight_layout with manual adjustment
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# Instead, use explicit figure-level adjustment
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
plt.show()

# %%
# Conclusion
# ---------------------
# In this example, we've demonstrated the power of block-wise processing for forward
# error correction (FEC) coding:
#
# Key points:
# - Block-wise processing is fundamental to many error correction schemes
# - The `apply_blockwise` function provides a convenient way to apply operations
#   on blocks of data
# - We demonstrated several practical applications of block-wise processing:
# - Simple parity-based error detection
# - Complex operations that return multiple values
# q- Implementation of a (7,4) Hamming code for error correction
#
# These techniques form the foundation for more advanced error correction codes
# like BCH, Reed-Solomon, and LDPC codes, which can correct multiple errors per block.
#
# References:
# - :cite:`lin2004error` - Provides detailed treatments of block coding and parity checks
# - :cite:`moon2005error` - Covers mathematical methods for various error correction codes
# - :cite:`golay1949notes` - Historical paper on efficient coding techniques
# - :cite:`richardson2008modern` - Modern approaches to coding theory
