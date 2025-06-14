"""
FEC Encoders Tutorial
=========================================

This tutorial demonstrates how to use various Forward Error Correction (FEC) encoders
from the kaira.models.fec.encoders module.

FEC codes add redundancy to transmitted data, allowing receivers to detect and
correct errors without retransmission.

We'll explore:

- Basic block codes (Repetition, Single Parity Check)
- Linear block codes (Hamming)
- Cyclic codes and BCH codes
- Reed-Solomon codes
- Advanced features and performance evaluation
"""

# %%
# First, let's import the necessary modules
import numpy as np
import torch

from examples.example_utils.plotting import (
    plot_code_structure_comparison,
    plot_hamming_code_visualization,
    setup_plotting_style,
)

# Import encoders from kaira
from kaira.models.fec.encoders import (
    BCHCodeEncoder,
    CyclicCodeEncoder,
    GolayCodeEncoder,
    HammingCodeEncoder,
    LinearBlockCodeEncoder,
    ReedSolomonCodeEncoder,
    RepetitionCodeEncoder,
    SingleParityCheckCodeEncoder,
    SystematicLinearBlockCodeEncoder,
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
setup_plotting_style()

# %%
# Helper Functions
# ------------------------------
#
# Let's define some helper functions to display code parameters and visualize codewords.


def print_code_parameters(encoder, name: str) -> None:
    """Print the parameters of a code encoder."""
    print(f"\n{name}:")
    print(f"  - Code length (n): {encoder.code_length}")
    print(f"  - Message length (k): {encoder.code_dimension}")
    print(f"  - Rate (k/n): {encoder.code_dimension/encoder.code_length:.3f}")
    if hasattr(encoder, "error_correction_capability"):
        print(f"  - Error correction capability (t): {encoder.error_correction_capability}")
    if hasattr(encoder, "generator_matrix"):
        print("  - Has generator matrix: Yes")
    if hasattr(encoder, "parity_check_matrix"):
        print("  - Has parity check matrix: Yes")


def visualize_codeword(message: torch.Tensor, codeword: torch.Tensor, name: str) -> None:
    """Visualize a message and its corresponding codeword."""
    plot_hamming_code_visualization(message=message.numpy(), codeword=codeword.numpy(), title=f"{name} - Message ({len(message)} bits) vs Codeword ({len(codeword)} bits)")


# %%
# Part 1: Basic Block Codes
# ------------------------------------------------
#
# Let's start with the simplest FEC codes: repetition codes and single parity check codes.

print("\n=========== Part 1: Basic Block Codes ===========")

# %%
# Repetition Code
# ------------------------
#
# A repetition code simply repeats each bit of the message multiple times.
# For a 3x repetition code, each message bit is encoded as 3 identical bits.

rep_encoder = RepetitionCodeEncoder(repetitions=3)
print_code_parameters(rep_encoder, "Repetition Code (3x)")

# Create a simple message
message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
print(f"Original message: {message.int().tolist()}")

# Encode using repetition code
codeword = rep_encoder(message)
print(f"Encoded codeword: {codeword.int().tolist()}")

# Visualize the encoding
visualize_codeword(message, codeword, "Repetition Code (3x)")

# %%
# Single Parity Check Code
# -------------------------------------------
#
# A single parity check code adds one parity bit to the message to ensure
# the total number of 1s in the codeword is even (even parity).

spc_encoder = SingleParityCheckCodeEncoder(dimension=4)
print_code_parameters(spc_encoder, "Single Parity Check Code")

# Encode using single parity check code
message = torch.tensor([1.0, 1.0, 0.0, 1.0])
codeword = spc_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"Parity bit: {codeword[-1].int().item()}")  # Should be 1 to make even parity

# %%
# Part 2: Linear Block Codes
# ------------------------------------------------
#
# Linear block codes are more sophisticated and offer better error correction
# capabilities while maintaining good code rates.

print("\n=========== Part 2: Linear Block Codes ===========")

# %%
# Hamming Code
# ---------------------
#
# Hamming codes are perfect codes that can correct one error.
# The Hamming(7,4) code encodes 4 message bits into 7 code bits.

hamming_encoder = HammingCodeEncoder(mu=3)  # Hamming(7,4) code
print_code_parameters(hamming_encoder, "Hamming(7,4) Code")

# Create a message
message = torch.tensor([1.0, 0.0, 1.0, 1.0])
print(f"Original message: {message.int().tolist()}")

# Encode using Hamming code
codeword = hamming_encoder(message)
print(f"Encoded codeword: {codeword.int().tolist()}")

# Display the generator matrix
print("Generator Matrix G:")
if hasattr(hamming_encoder, "generator_matrix"):
    print(hamming_encoder.generator_matrix.int().numpy())

# %%
# Custom Linear Block Code
# -------------------------------------------
#
# We can create custom linear block codes by defining our own generator matrix.

# Define a custom generator matrix for a (6,3) linear block code
G = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

custom_encoder = LinearBlockCodeEncoder(generator_matrix=G)
print_code_parameters(custom_encoder, "Custom (6,3) Linear Block Code")

message = torch.tensor([1.0, 1.0, 0.0])
codeword = custom_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")

# %%
# Part 3: Cyclic Codes and BCH Codes
# ------------------------------------------------
#
# Cyclic codes have the property that any cyclic shift of a codeword is also a codeword.
# BCH codes are a class of cyclic codes with excellent error correction capabilities.

print("\n=========== Part 3: Cyclic Codes and BCH Codes ===========")

# %%
# Cyclic Code
# --------------------
#
# Let's use a standard Hamming(7,4) code, which is a cyclic code.

cyclic_encoder = CyclicCodeEncoder.create_standard_code("Hamming(7,4)")
print_code_parameters(cyclic_encoder, "Cyclic Code (7,4)")

message = torch.tensor([1.0, 0.0, 1.0, 1.0])
codeword = cyclic_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")

# %%
# BCH Code
# ------------
#
# BCH codes are powerful cyclic codes that can correct multiple errors.
# Here we create a BCH(15,7) code that can correct up to 2 errors.

bch_encoder = BCHCodeEncoder(mu=4, delta=5)  # GF(2^4), minimum distance 5
print_code_parameters(bch_encoder, "BCH(15,7) Code")

message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
codeword = bch_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")

# %%
# Golay Code
# --------------
#
# The binary Golay code is a perfect code that can correct up to 3 errors.

golay_encoder = GolayCodeEncoder()
print_code_parameters(golay_encoder, "Binary Golay(23,12) Code")

message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
codeword = golay_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")

# %%
# Part 4: Reed-Solomon Codes
# ------------------------------------------------
#
# Reed-Solomon codes are particularly good at correcting burst errors.

print("\n=========== Part 4: Reed-Solomon Codes ===========")

# %%
# Reed-Solomon Code
# -------------------------------
#
# Reed-Solomon codes operate on symbols rather than bits, making them
# excellent for burst error correction.

# RS(15,9) code over GF(2^4)
# - mu=4 since the code length is n=2^mu-1=15
# - The redundancy is r=n-k=15-9=6
# - The design distance is delta=r+1=7
rs_encoder = ReedSolomonCodeEncoder(mu=4, delta=7)
print_code_parameters(rs_encoder, "Reed-Solomon(15,9) Code")

# Create a message for RS encoding
message = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])  # 9 symbols
codeword = rs_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"RS can correct up to {rs_encoder.error_correction_capability} symbol errors")

# %%
# Part 5: Advanced Features
# ------------------------------------------------
#
# Now let's explore some advanced features of FEC codes.

print("\n=========== Part 5: Advanced Features ===========")

# %%
# Systematic Encoding
# ---------------------------------
#
# A systematic code preserves the original message bits in the codeword,
# making it easier to extract the message without full decoding.

# Create a systematic linear block code from a parity submatrix
# For a systematic code, we need the parity submatrix P rather than the full generator matrix
# For our (6,3) code, the first 3 columns are identity and remaining 3 are parity
G = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.1], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

# Extract the parity submatrix (last 3 columns)
P = G[:, 3:]

# Initialize the systematic encoder with the parity submatrix
systematic_encoder = SystematicLinearBlockCodeEncoder(parity_submatrix=P)
print_code_parameters(systematic_encoder, "Systematic (6,3) Linear Block Code")

message = torch.tensor([1.0, 1.0, 0.0])
codeword = systematic_encoder(message)
print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print("Note how the first 3 bits of the codeword match the message (systematic property)")

# %%
# Batch Processing
# ------------------------
#
# All encoders support batch processing to encode multiple messages at once.

# Encode multiple messages at once using batch dimensions
messages = torch.tensor([[1.0, 0.0, 1.0, 1.1], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])  # First message  # Second message  # Third message

# Encode all messages at once with the Hamming encoder
codewords = hamming_encoder(messages)
print(f"Batch of messages shape: {messages.shape}")
print(f"Batch of codewords shape: {codewords.shape}")

for i in range(len(messages)):
    print(f"Message {i+1}: {messages[i].int().tolist()} → Codeword: {codewords[i].int().tolist()}")

# %%
# Part 6: Performance Evaluation
# ------------------------------------------------
#
# Finally, let's compare the performance characteristics of different codes.

print("\n=========== Part 6: Performance Evaluation ===========")

# %%
# Comparing Code Rates
# ----------------------------------
#
# The code rate (k/n) represents the efficiency of the code.
# Higher rates mean less redundancy but typically weaker error correction.

encoders = {
    "Repetition (3x)": RepetitionCodeEncoder(repetition_factor=3),
    "Single Parity Check": SingleParityCheckCodeEncoder(dimension=4),
    "Hamming(7,4)": HammingCodeEncoder(mu=3),
    "BCH(15,7)": BCHCodeEncoder(mu=4, delta=5),
    "RS(15,9)": ReedSolomonCodeEncoder(mu=4, delta=7),
    "Golay(23,12)": GolayCodeEncoder(),
}

# Compute and plot code rates
names = []
rates = []
min_distances = []

for name, encoder in encoders.items():
    rate = encoder.code_dimension / encoder.code_length
    rates.append(rate)
    names.append(name)

    # Get minimum distance if available
    if hasattr(encoder, "minimum_distance"):
        min_distances.append(encoder.minimum_distance)
    elif hasattr(encoder, "design_distance"):
        min_distances.append(encoder.design_distance)
    else:
        min_distances.append(None)

# Generate code comparison visualization
plot_code_structure_comparison(names=names, rates=rates, min_distances=min_distances, title="Comparison of FEC Code Performance")

# Code Rates and Minimum Distances Summary:
print("\nCode Rates and Minimum Distances:")
for i, name in enumerate(names):
    print(f"{name}: Rate = {rates[i]:.3f}, Min Distance = {min_distances[i]}")

# %%
# Conclusion
# --------------
#
# This tutorial demonstrated various FEC encoders from the kaira library.
#
# The choice of encoder depends on the specific requirements of the application:
#
# - Repetition codes are simple but inefficient
# - Hamming codes are efficient for single-error correction
# - BCH and Reed-Solomon codes provide strong error correction for burst errors
# - Systematic codes preserve the original message in the codeword
#
# For error correction, pair these encoders with their corresponding decoders.
