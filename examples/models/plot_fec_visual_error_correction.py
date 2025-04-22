"""
====================================================================
Visualizing Error Correction in Action
====================================================================

This example provides a visually rich demonstration of forward error correction (FEC)
techniques in action. We'll visualize how different coding schemes protect data as it
passes through noisy channels, with animated and interactive visualizations that show
the encoding, transmission, corruption, and decoding processes.
"""

import matplotlib.animation as animation
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from kaira.channels import BinarySymmetricChannel

# %%
# Setting up
# ----------------------
# First, we set a random seed to ensure reproducibility of results and
# configure our visualization settings.

torch.manual_seed(42)
np.random.seed(42)

# Configure better visualization styles
plt.style.use("seaborn-v0_8-whitegrid")
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

# %%
# Repetition Code Visualization
# ----------------------------------------------------------------------
# The repetition code is one of the simplest error correction codes. It works by
# repeating each bit multiple times and using a majority vote to decode.
# Let's visualize this process with a 5-bit repetition code.


def encode_repetition(data, repetition=5):
    """Encode data using a repetition code."""
    encoded = torch.zeros(len(data) * repetition, dtype=data.dtype)
    for i, bit in enumerate(data):
        encoded[i * repetition : (i + 1) * repetition] = bit
    return encoded


def decode_repetition(encoded_data, repetition=5):
    """Decode repetition code using majority voting."""
    data_len = len(encoded_data) // repetition
    decoded = torch.zeros(data_len, dtype=encoded_data.dtype)

    for i in range(data_len):
        block = encoded_data[i * repetition : (i + 1) * repetition]
        # Majority vote
        decoded[i] = 1 if torch.sum(block) > repetition / 2 else 0

    return decoded


# Create some example data
data = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
repetition = 5

# Encode the data
encoded_data = encode_repetition(data, repetition)

# Create a noisy channel (Binary Symmetric Channel)
p_error = 0.2  # Error probability
bsc = BinarySymmetricChannel(p_error)

# Pass the encoded data through the channel
received_data = bsc(encoded_data)

# Decode the received data
decoded_data = decode_repetition(received_data, repetition)

# Calculate error metrics
transmission_errors = (encoded_data != received_data).sum().item()
decoding_errors = (data != decoded_data).sum().item()

print(f"Original data: {data}")
print(f"Encoded data: {encoded_data}")
print(f"Received data: {received_data}")
print(f"Decoded data: {decoded_data}")
print(f"Number of transmission errors: {transmission_errors}")
print(f"Number of decoding errors: {decoding_errors}")

# %%
# Static Visualization of the Repetition Code Process
# -------------------------------------------------------------------------------------------------------------------------
# Let's create a detailed visual representation of the entire process, from the original
# data to the final decoded result, showing each step along the way.

# Create a figure with a complex layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 1, height_ratios=[1, 1.5, 1.5, 1], hspace=0.4)

# 1. Original data
ax1 = fig.add_subplot(gs[0])
ax1.set_title("Original Data", fontsize=16, fontweight="bold")

# Plot the original data as a bit sequence
bars1 = ax1.bar(np.arange(len(data)), data.numpy(), color=[original_color if bit == 1 else "white" for bit in data], edgecolor="black", linewidth=1.5)

# Add bit values as text
for i, bit in enumerate(data):
    ax1.text(i, bit + 0.05, str(int(bit)), ha="center", va="bottom", fontweight="bold", fontsize=14)

    # Add bit index labels
    ax1.text(i, -0.15, f"bit {i}", ha="center", va="top", fontsize=10)

ax1.set_ylim(-0.3, 1.3)
ax1.set_xlim(-0.5, len(data) - 0.5)
ax1.set_yticks([0, 1])
ax1.set_xticks(np.arange(len(data)))
ax1.set_xticklabels([])

# Add an arrow pointing to the encoding step
ax1.annotate("", xy=(len(data) / 2, -0.5), xytext=(len(data) / 2, -0.3), arrowprops=dict(arrowstyle="->", lw=2, color=highlight_color))

# 2. Encoding process
ax2 = fig.add_subplot(gs[1])
ax2.set_title(f"Encoded Data (Repetition Code, {repetition}x)", fontsize=16, fontweight="bold")

# Plot the encoded data
encoded_x = np.arange(len(encoded_data))
bars2 = ax2.bar(encoded_x, encoded_data.numpy(), color=[encoded_color if bit == 1 else "white" for bit in encoded_data], edgecolor="black", linewidth=1.5)

# Add bit values and grouping
for i in range(len(data)):
    # Highlight groups with rectangles
    start = i * repetition - 0.4
    width = repetition - 0.2
    height = 1.4
    rect = Rectangle((start, -0.2), width, height, linewidth=2, edgecolor=highlight_color, facecolor="none", alpha=0.7)
    ax2.add_patch(rect)

    # Add group labels
    ax2.text(i * repetition + (repetition - 1) / 2, 1.3, f"bit {i}", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Add bit values within each group
    for j in range(repetition):
        idx = i * repetition + j
        ax2.text(idx, encoded_data[idx] + 0.05, str(int(encoded_data[idx])), ha="center", va="bottom", fontsize=9)

# Add an explanation of the encoding process
textstr = f"Each bit is repeated {repetition} times for redundancy"
ax2.text(0.5, -0.4, textstr, transform=ax2.transAxes, ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

ax2.set_ylim(-0.5, 1.5)
ax2.set_xlim(-0.5, len(encoded_data) - 0.5)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

# Add an arrow pointing to the channel
ax2.annotate(f"Noisy Channel\n(p_error = {p_error:.1f})", xy=(len(encoded_data) / 2, -0.7), xytext=(len(encoded_data) / 2, -0.5), ha="center", fontsize=12, arrowprops=dict(arrowstyle="->", lw=2, color=error_color))

# 3. Transmission through the noisy channel
ax3 = fig.add_subplot(gs[2])
ax3.set_title("Received Data (with errors)", fontsize=16, fontweight="bold")

# Plot the received data
bars3 = ax3.bar(encoded_x, received_data.numpy(), color=["white" for _ in range(len(received_data))], edgecolor="black", linewidth=1.5)

# Color the bars and highlight errors
for i in range(len(received_data)):
    # Determine color based on whether there was an error
    if received_data[i] != encoded_data[i]:
        bars3[i].set_color(error_color)
        # Mark error with an X
        ax3.text(i, received_data[i] + 0.1, "✗", color="black", fontsize=12, ha="center", va="center", fontweight="bold", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
    else:
        bars3[i].set_color(encoded_color if received_data[i] == 1 else "white")

    # Add bit values
    ax3.text(i, received_data[i] + 0.05, str(int(received_data[i])), ha="center", va="bottom", fontsize=9, color="black" if received_data[i] == encoded_data[i] else "white")

# Add group rectangles and labels (matching the encoded data)
for i in range(len(data)):
    start = i * repetition - 0.4
    width = repetition - 0.2
    height = 1.4
    rect = Rectangle((start, -0.2), width, height, linewidth=2, edgecolor=highlight_color, facecolor="none", alpha=0.7)
    ax3.add_patch(rect)

    # Count errors in this group
    group_errors = sum([1 for j in range(repetition) if received_data[i * repetition + j] != encoded_data[i * repetition + j]])

    # Add group label with error count
    ax3.text(i * repetition + (repetition - 1) / 2, 1.3, f"bit {i}: {group_errors} errors", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white" if group_errors <= repetition // 2 else "salmon", alpha=0.8))

# Add an explanation of the transmission errors
textstr = f"The channel flipped {transmission_errors} bits"
ax3.text(0.5, -0.4, textstr, transform=ax3.transAxes, ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="salmon", alpha=0.5))

ax3.set_ylim(-0.5, 1.5)
ax3.set_xlim(-0.5, len(received_data) - 0.5)
ax3.set_yticks([0, 1])
ax3.set_xticks([])

# Add an arrow pointing to the decoding step
ax3.annotate("Majority Decoding", xy=(len(encoded_data) / 2, -0.7), xytext=(len(encoded_data) / 2, -0.5), ha="center", fontsize=12, arrowprops=dict(arrowstyle="->", lw=2, color=corrected_color))

# 4. Decoded result
ax4 = fig.add_subplot(gs[3])
ax4.set_title("Decoded Data (after error correction)", fontsize=16, fontweight="bold")

# Plot the decoded data with comparison to original
x_pos = np.arange(len(data))
width = 0.4

# Plot original data as reference (transparent)
ax4.bar(x_pos - width / 2, data.numpy(), width, alpha=0.3, color=[original_color if bit == 1 else "white" for bit in data], edgecolor="black", linewidth=1.5, label="Original")

# Plot decoded data
bars4 = ax4.bar(x_pos + width / 2, decoded_data.numpy(), width, color=[corrected_color if bit == 1 else "white" for bit in decoded_data], edgecolor="black", linewidth=1.5, label="Decoded")

# Mark errors and matches
for i in range(len(data)):
    # Show the original and decoded values
    ax4.text(i - width / 2, data[i] + 0.05, str(int(data[i])), ha="center", va="bottom", fontsize=12)

    ax4.text(i + width / 2, decoded_data[i] + 0.05, str(int(decoded_data[i])), ha="center", va="bottom", fontsize=12)

    # Mark if there was an error in decoding
    if data[i] != decoded_data[i]:
        ax4.plot([i - width / 2, i + width / 2], [data[i] + 0.2, decoded_data[i] + 0.2], "r--", linewidth=2)
        ax4.text(i, 1.1, "✗", color=error_color, fontsize=16, ha="center", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
    else:
        ax4.text(i, 1.1, "✓", color="green", fontsize=14, ha="center", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

# Add bit indices
for i in range(len(data)):
    ax4.text(i, -0.15, f"bit {i}", ha="center", va="top", fontsize=10)

# Add a legend
ax4.legend(loc="upper right", fontsize=12)

# Add text explaining decoding result
if decoding_errors == 0:
    result_text = "All bits were correctly decoded despite channel errors!"
    color = "lightgreen"
else:
    result_text = f"{decoding_errors} bits were incorrectly decoded"
    color = "salmon"

ax4.text(0.5, -0.4, result_text, transform=ax4.transAxes, ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))

ax4.set_ylim(-0.5, 1.5)
ax4.set_xlim(-0.5, len(data) - 0.5)
ax4.set_yticks([0, 1])
ax4.set_xticks(np.arange(len(data)))

# Add a title for the entire figure
fig.suptitle("Repetition Code: Encoding, Transmission, and Decoding Process", fontsize=20, fontweight="bold", y=0.98)

# Replacing tight_layout with subplots_adjust
fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.05, hspace=0.4)
plt.show()

# %%
# Animating the Error Correction Process
# -------------------------------------------------------------------------------------------
# Now let's create an animation that shows how errors are introduced in the transmission
# and then corrected by the repetition code's majority voting mechanism.


def create_repetition_code_animation():
    """Create an animated visualization of the repetition code process."""
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.2], gridspec_kw={"hspace": 0.4})

    # Original and encoded data (same as before)
    data = torch.tensor([1, 0, 1, 0])
    repetition = 5
    encoded_data = encode_repetition(data, repetition)

    # We'll animate the transmission process
    received_data = encoded_data.clone()  # Start with no errors
    decoded_data = decode_repetition(received_data, repetition)

    # Set up the plots
    ax1.set_title("Transmitted Data (Errors Introduced Over Time)", fontsize=14, fontweight="bold")
    ax2.set_title("Decoded Result (Using Majority Voting)", fontsize=14, fontweight="bold")

    # Initialize the bars for encoded/received data
    bars1 = ax1.bar(np.arange(len(encoded_data)), encoded_data.numpy(), color=[encoded_color if bit == 1 else "white" for bit in encoded_data], edgecolor="black", linewidth=1.5)

    # Initialize the bars for decoded data
    bars2 = ax2.bar(np.arange(len(data)), decoded_data.numpy(), color=[corrected_color if bit == 1 else "white" for bit in decoded_data], edgecolor="black", linewidth=1.5)

    # Add group rectangles to the encoded data view
    for i in range(len(data)):
        start = i * repetition - 0.4
        width = repetition - 0.2
        height = 1.4
        rect = Rectangle((start, -0.2), width, height, linewidth=2, edgecolor=highlight_color, facecolor="none", alpha=0.7)
        ax1.add_patch(rect)

        # Add group labels
        ax1.text(i * repetition + (repetition - 1) / 2, 1.3, f"bit {i}", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Add bit indices to the decoded view
    for i in range(len(data)):
        ax2.text(i, -0.15, f"bit {i}", ha="center", va="top", fontsize=10)

    # Add original data as a reference line in the decoded view
    for i in range(len(data)):
        ax2.axhline(y=data[i], xmin=(i - 0.4) / len(data), xmax=(i + 0.4) / len(data), color=original_color, linestyle="--", linewidth=2, alpha=0.7)
        ax2.text(i, data[i] + 0.05, f"Original: {int(data[i])}", ha="center", va="bottom", fontsize=8, color=original_color)

    # Set axes limits
    ax1.set_ylim(-0.3, 1.5)
    ax1.set_xlim(-0.5, len(encoded_data) - 0.5)
    ax1.set_yticks([0, 1])
    ax1.set_xticks([])

    ax2.set_ylim(-0.3, 1.3)
    ax2.set_xlim(-0.5, len(data) - 0.5)
    ax2.set_yticks([0, 1])
    ax2.set_xticks(np.arange(len(data)))

    # Add a status text for errors and correction
    status_text1 = ax1.text(0.02, 0.02, "", transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    status_text2 = ax2.text(0.02, 0.02, "", transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Add a title for the entire figure
    fig.suptitle("Error Introduction and Correction Animation", fontsize=16, fontweight="bold", y=0.98)

    # Number of frames: we'll introduce one error at a time
    num_errors = 8  # We'll introduce up to 8 errors

    # Function to update the figure for animation
    def update(frame):
        """Update the figure for each frame of the animation."""
        if frame < num_errors:
            # Introduce a new error in this frame
            error_idx = np.random.randint(0, len(encoded_data))
            received_data[error_idx] = 1 - received_data[error_idx]  # Flip the bit

            # Update the decoded data
            new_decoded = decode_repetition(received_data, repetition)

            # Update the transmission visualization
            for i, bar in enumerate(bars1):
                if received_data[i] != encoded_data[i]:
                    bar.set_color(error_color)
                else:
                    bar.set_color(encoded_color if received_data[i] == 1 else "white")

            # Update the decoded visualization
            for i, bar in enumerate(bars2):
                bars2[i].set_height(new_decoded[i].item())
                bar.set_color(corrected_color if new_decoded[i] == 1 else "white")

                # Highlight in red if it doesn't match the original
                if new_decoded[i] != data[i]:
                    bar.set_color(error_color)

            # Update error counts and status texts
            transmission_errors = (encoded_data != received_data).sum().item()
            decoding_errors = (data != new_decoded).sum().item()

            status_text1.set_text(f"Transmission errors: {transmission_errors}/{len(encoded_data)}")
            status_text2.set_text(f"Decoding errors: {decoding_errors}/{len(data)}")

            if decoding_errors > 0:
                status_text2.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="salmon", alpha=0.8))
            else:
                status_text2.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

        # Fix: Convert bars1 and bars2 to lists before concatenating with other lists
        return list(bars1) + list(bars2) + [status_text1, status_text2]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_errors, interval=1000, blit=False, repeat=True)

    # Fix: Remove tight_layout to avoid warnings
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.4)

    return ani


# Create and display the animation
repetition_animation = create_repetition_code_animation()
HTML(repetition_animation.to_jshtml())  # For interactive notebook display

# %%
# Error Correction Capability Visualization
# ---------------------------------------------------------------------------------------------
# Let's visualize how the error correction capability changes with different repetition
# code parameters and error probabilities.

# Define parameters
repetition_values = [3, 5, 7, 9]  # Different repetition factors
p_values = [0.1, 0.2, 0.3, 0.4]  # Different error probabilities
num_trials = 1000  # Number of simulation trials
data_length = 10  # Length of the original data

# Run simulations
results = np.zeros((len(repetition_values), len(p_values)))

for i, rep in enumerate(repetition_values):
    for j, p in enumerate(p_values):
        error_count = 0

        # Run multiple trials
        for _ in range(num_trials):
            # Generate random data
            data = torch.randint(0, 2, (data_length,), dtype=torch.float32)

            # Encode, transmit, and decode
            encoded = encode_repetition(data, rep)
            channel = BinarySymmetricChannel(p)
            received = channel(encoded)
            decoded = decode_repetition(received, rep)

            # Count decoding errors
            error_count += (data != decoded).sum().item()

        # Calculate error rate
        results[i, j] = error_count / (num_trials * data_length)

# Visualize the results
plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(results, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=[f"p={p}" for p in p_values], yticklabels=[f"r={r}" for r in repetition_values])

plt.xlabel("Channel Error Probability", fontsize=14)
plt.ylabel("Repetition Factor", fontsize=14)
plt.title("Decoding Error Rate for Different Repetition Codes", fontsize=16, fontweight="bold")

# Add explanatory annotations
plt.text(
    0.5,
    -0.15,
    "Error rate increases with channel error probability and decreases with repetition factor.\n" "Odd repetition values work better because they avoid ties in majority voting.",
    ha="center",
    va="center",
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()

# %%
# Comparison with Other Error Correction Codes
# ------------------------------------------------------------------------------------------------------
# The repetition code is simple but inefficient. Let's compare its theoretical performance
# with other error correction codes like Hamming codes and Reed-Solomon codes.

# Performance comparison data (theoretical)
# Format: [Code Name, Code Rate, Min Distance, Error Correction Capability]
code_data: list[list[str | int | float]] = [
    ["Repetition (3)", 1 / 3, 3, 1],
    ["Repetition (5)", 1 / 5, 5, 2],
    ["Repetition (7)", 1 / 7, 7, 3],
    ["Hamming (7,4)", 4 / 7, 3, 1],
    ["Hamming (15,11)", 11 / 15, 3, 1],
    ["Reed-Solomon (7,3)", 3 / 7, 5, 2],
    ["Reed-Solomon (15,9)", 9 / 15, 7, 3],
    ["BCH (15,7)", 7 / 15, 5, 2],
    ["LDPC (Rate 1/2)", 1 / 2, "~8", "~4"],
]

# Convert data to arrays
code_names: list[str] = [str(code[0]) for code in code_data]
code_rates = [code[1] if isinstance(code[1], (int, float)) else 0.5 for code in code_data]
min_distances = [code[2] if isinstance(code[2], (int, float)) else 8 for code in code_data]
error_capabilities = [code[3] if isinstance(code[3], (int, float)) else 4 for code in code_data]

# Theoretical efficiency metric (Error correction capability / Rate)
efficiency = [err / rate for err, rate in zip(error_capabilities, code_rates)]

# Create a comparison visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot code rate vs error correction capability
ax1.scatter(code_rates, error_capabilities, s=200, c=np.arange(len(code_names)), cmap="viridis", alpha=0.8, edgecolors="black")

# Add code names as labels
for i, name in enumerate(code_names):
    ax1.annotate(name, (code_rates[i], error_capabilities[i]), xytext=(5, 5), textcoords="offset points", fontsize=10)

# Add a theoretical bound line
x = np.linspace(0.1, 1, 100)
y = 0.5 * (1 - x) / x  # Simplified approximation of theoretical bound
ax1.plot(x, y, "r--", alpha=0.5, label="Theoretical Bound")

ax1.set_xlabel("Code Rate (k/n)", fontsize=14)
ax1.set_ylabel("Error Correction Capability (t)", fontsize=14)
ax1.set_title("Error Correction vs. Code Rate", fontsize=16, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot efficiency comparison
bars = ax2.bar(code_names, efficiency, color=plt.cm.viridis(np.linspace(0, 1, len(code_names))))

# Add value labels
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{eff:.2f}", ha="center", va="bottom", fontsize=10)

# Highlight repetition codes
for i, name in enumerate(code_names):
    if "Repetition" in name:
        bars[i].set_color("#e74c3c")  # Highlight repetition codes in red

ax2.set_xlabel("Error Correction Code", fontsize=14)
ax2.set_ylabel("Efficiency (t/Rate)", fontsize=14)
ax2.set_title("Error Correction Efficiency Comparison", fontsize=16, fontweight="bold")

# Fix: Set xticks before xticklabels to avoid FixedFormatter warning
ax2.set_xticks(np.arange(len(code_names)))
ax2.set_xticklabels(code_names, rotation=45, ha="right")

ax2.grid(True, axis="y", alpha=0.3)

# Add annotation explaining efficiency
ax2.text(
    0.5,
    -0.2,
    "Efficiency = Error Correction Capability / Code Rate\n" "Higher values indicate better trade-off between redundancy and protection.\n" "Repetition codes (in red) are simple but less efficient than more advanced codes.",
    ha="center",
    va="center",
    transform=ax2.transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# %%
# Interactive Decoder Visualization
# -------------------------------------------------------------------------
# Let's create an interactive visualization that shows how majority decoding works
# for the repetition code.


def visualize_majority_decoder(rep=5, p_error=0.2):
    """Create a visualization of the majority decoding process."""
    # Generate a single bit of data
    data_bit = torch.tensor([1])

    # Encode with repetition
    encoded = encode_repetition(data_bit, rep)

    # Create several error patterns to demonstrate majority voting
    patterns = []
    votes = []

    # No errors
    patterns.append(encoded.clone())
    votes.append(1)

    # One error
    if rep > 1:
        temp = encoded.clone()
        temp[0] = 1 - temp[0]  # Flip one bit
        patterns.append(temp)
        votes.append(1)

    # Two errors (if possible)
    if rep > 3:
        temp = encoded.clone()
        temp[0] = 1 - temp[0]  # Flip first bit
        temp[1] = 1 - temp[1]  # Flip second bit
        patterns.append(temp)
        votes.append(1)

    # Too many errors (if possible)
    if rep > 2:
        temp = encoded.clone()
        # Flip majority of bits
        for i in range((rep // 2) + 1):
            temp[i] = 1 - temp[i]
        patterns.append(temp)
        votes.append(0)  # Incorrect decoding

    # Create the visualization
    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 3 * len(patterns)), gridspec_kw={"hspace": 0.4})

    # Ensure axes is always an array to make it iterable
    axes = np.atleast_1d(axes)

    for i, (pattern, vote, ax) in enumerate(zip(patterns, votes, axes)):
        # Count ones and zeros
        num_ones = pattern.sum().item()
        num_zeros = rep - num_ones

        # Determine majority
        majority = "1" if num_ones > num_zeros else "0"
        correct = vote == data_bit.item()

        # Title based on error pattern
        if i == 0:
            title = "No Errors"
        elif i == 1:
            title = "One Error"
        elif i == 2:
            title = "Two Errors"
        else:
            title = f"{(rep // 2) + 1} Errors (Too Many)"

        ax.set_title(f"{title} - Decoded as {majority}", fontsize=14, fontweight="bold")

        # Plot the bits
        ax.bar(np.arange(rep), pattern.numpy(), color=[original_color if bit == 1 else "white" for bit in pattern], edgecolor="black", linewidth=1.5)

        # Mark errors
        for j in range(rep):
            if pattern[j] != encoded[j]:
                ax.text(j, pattern[j] + 0.05, "✗", color=error_color, fontsize=16, ha="center", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

        # Show the vote counts
        ax.text(rep + 0.5, 0.5, f"Count: {num_zeros} zeros / {num_ones} ones", ha="left", va="center", fontsize=12)

        # Show the decision
        result_text = f"Majority vote: {majority} → " + ("Correct" if correct else "Incorrect")
        box_color = "lightgreen" if correct else "salmon"

        ax.text(rep + 0.5, 0.2, result_text, ha="left", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7))

        # Set axes limits
        ax.set_ylim(-0.1, 1.3)
        ax.set_xlim(-0.5, rep + 4)
        ax.set_yticks([0, 1])
        ax.set_xticks(np.arange(rep))
        ax.set_xticklabels([f"bit {j}" for j in range(rep)])

    # Add a title for the entire figure
    fig.suptitle(f"Majority Decoding of {rep}-Repetition Code", fontsize=16, fontweight="bold", y=0.98)

    # Add an explanation of majority voting
    text = f"Majority voting decodes to the value that appears most frequently.\n" f"With {rep}-repetition, it can correct up to {rep//2} errors.\n" f"If more than {rep//2} bits are flipped, the decoding will be incorrect."

    fig.text(0.5, 0.01, text, ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Replace tight_layout with subplots_adjust
    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.1, hspace=0.4)
    plt.show()


# Demonstrate the majority decoder
visualize_majority_decoder(rep=5)

# %%
# Conclusion
# ----------------------
# In this example, we've visually explored how forward error correction works
# with a simple repetition code:
#
# - We saw how the encoding process adds redundancy to protect against errors
# - We visualized how errors are introduced in the transmission channel
# - We demonstrated how the decoding process can recover from these errors
# - We analyzed the error correction capabilities of different code parameters
# - We compared repetition codes with more advanced error correction techniques
#
# Key takeaways:
# - Forward error correction allows receivers to correct errors without retransmission
# - The repetition code is simple but inefficient compared to more advanced codes
# - The error correction capability depends on the code parameters and channel characteristics
# - Visual animations help understand the encoding-decoding process
#
# References:
# - :cite:`lin2004error` - Provides comprehensive coverage of error control coding
# - :cite:`moon2005error` - Explains mathematical methods for error correction
# - :cite:`macwilliams1977theory` - Classic text on the theory of error-correcting codes
