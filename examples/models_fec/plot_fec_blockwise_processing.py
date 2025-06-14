"""
====================================================================
Block-wise Processing for FEC
====================================================================

This example demonstrates how to perform block-wise processing of data for forward
error correction (FEC) using the `apply_blockwise` utility function. Block-wise
processing is essential in many coding schemes like block codes, systematic codes,
and interleaved coding.
"""

import numpy as np
import torch

from kaira.models.fec.utils import apply_blockwise

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# %%
# Setting up
# ----------------------
# Block-wise Processing Configuration
# ==================================

# First, we set a random seed to ensure reproducibility of results.
torch.manual_seed(42)
np.random.seed(42)

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

# Comment: Display original binary data for reference
print(f"Original binary data: {binary_data}")

# Apply NOT operation (1 -> 0, 0 -> 1) to each block of size 2
inverted_data = apply_blockwise(binary_data, block_size, lambda block: 1 - block)

# Comment: Show the result of block-wise NOT operation
print(f"Data after block-wise NOT operation: {inverted_data}")

# %%
# Visualizing Block-wise Operations
# -------------------------------------------------------------------------
# We can visualize how the block-wise operation transforms the data:

# Create blocks for visualization
input_blocks = [binary_data[i : i + block_size] for i in range(0, len(binary_data), block_size)]
output_blocks = [inverted_data[i : i + block_size] for i in range(0, len(inverted_data), block_size)]

fig = PlottingUtils.plot_blockwise_operation(input_blocks, output_blocks, "Block-wise NOT Operation")
fig.show()

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
# Comment: Display original data for reference
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

# Comment: Display data with parity bits added
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

# Comment: Show original encoded data and error positions
print(f"Original encoded data: {encoded_data}")
print(f"Introducing errors at positions: {error_positions}")

for pos in error_positions:
    corrupted_data[pos] = 1 - corrupted_data[pos]  # Flip the bit

# Comment: Display corrupted data
print(f"Corrupted data: {corrupted_data}")

# Reshape into blocks for parity checking
block_size_with_parity = block_size + 1
corrupted_blocks = corrupted_data.view(-1, block_size_with_parity)

# Check parity for each block
for i, block in enumerate(corrupted_blocks):
    is_valid = check_parity(block)
    # Comment: Display parity check results for each block
    print(f"Block {i+1}: {block}, Parity valid: {is_valid}")

# %%
# Visualizing Error Detection
# -------------------------------------------------------------
# Let's visualize the original, encoded, and corrupted data, highlighting where
# errors were introduced and which blocks had parity violations.

# Create syndrome and error pattern for visualization
syndrome = torch.zeros(len(corrupted_blocks))
for i, block in enumerate(corrupted_blocks):
    syndrome[i] = 1 if not check_parity(block) else 0

error_pattern = torch.zeros_like(corrupted_data)
for pos in error_positions:
    error_pattern[pos] = 1

fig = PlottingUtils.plot_parity_check_visualization(syndrome, error_pattern, "Parity Check Analysis")
fig.show()

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

# Comment: Display original blocks for reference
print("Original blocks:")
for i, block in enumerate(test_data):
    print(f"Block {i+1}: {block}")

# Comment: Show processed blocks and error detection results
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

# Comment: Display original data blocks
print("Original 4-bit data blocks:")
for i, block in enumerate(data_blocks):
    print(f"Block {i+1}: {block}")

# Comment: Show encoded Hamming codewords
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
# Comment: Display detailed results of Hamming decoding and error correction
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

# Select one example block to visualize in detail
block_idx = 0  # We'll use the first block
original_data = data_blocks[block_idx]
encoded = encoded_blocks[block_idx]
corrupted = corrupted_blocks[block_idx]
corrected, decoded, error_pos = results[block_idx]
actual_error_pos = error_positions[block_idx]

# Create simple generator and parity check matrices for visualization
# For Hamming(7,4), create simplified matrices
generator_matrix = torch.tensor([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)

parity_check_matrix = torch.tensor([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=torch.float32)

fig = PlottingUtils.plot_hamming_code_visualization(generator_matrix, parity_check_matrix, "Hamming Code Structure")
fig.show()

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
#   - Simple parity-based error detection
#   - Complex operations that return multiple values
#   - Implementation of a (7,4) Hamming code for error correction
#
# These techniques form the foundation for more advanced error correction codes
# like BCH, Reed-Solomon, and LDPC codes, which can correct multiple errors per block.
#
# References:
# - :cite:`lin2004error` - Provides detailed treatments of block coding and parity checks
# - :cite:`moon2005error` - Covers mathematical methods for various error correction codes
# - :cite:`golay1949notes` - Historical paper on efficient coding techniques
# - :cite:`richardson2008modern` - Modern approaches to coding theory
