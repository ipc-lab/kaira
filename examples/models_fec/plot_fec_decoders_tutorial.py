"""
FEC Decoders Tutorial
=========================================

This tutorial demonstrates how to use various Forward Error Correction (FEC) decoders
from the kaira.models.fec.decoders module.

FEC decoders recover original messages from possibly corrupted codewords that have
been transmitted over noisy channels.

We'll explore:

- Basic concepts in FEC decoding
- Hard-decision vs. soft-decision decoding
- Syndrome-based decoding
- Advanced algebraic decoders
- Maximum likelihood decoding
- Performance evaluation and error correction capabilities
"""

# %%
# First, let's import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import torch

# For simulating noisy channels
from kaira.channels import AWGNChannel
from kaira.models.fec.decoders import (
    BerlekampMasseyDecoder,
    BruteForceMLDecoder,
    SyndromeLookupDecoder,
    WagnerSoftDecisionDecoder,
)

# Import encoders and decoders from kaira
from kaira.models.fec.encoders import (
    BCHCodeEncoder,
    GolayCodeEncoder,
    HammingCodeEncoder,
    LinearBlockCodeEncoder,
    RepetitionCodeEncoder,
    SingleParityCheckCodeEncoder,
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Helper Functions
# ---------------------------
#
# Let's define some helper functions to display decoder information and visualize
# the error correction process.


def print_decoder_info(decoder, name: str) -> None:
    """Print information about a decoder."""
    print(f"\n{name}:")
    print(f"  - Decoder type: {type(decoder).__name__}")
    if hasattr(decoder, "encoder"):
        print(f"  - Associated encoder: {type(decoder.encoder).__name__}")
        print(f"  - Code length (n): {decoder.encoder.code_length}")
        print(f"  - Message length (k): {decoder.encoder.code_dimension}")
        print(f"  - Code rate: {decoder.encoder.code_dimension/decoder.encoder.code_length:.3f}")


def visualize_error_correction(message: torch.Tensor, codeword: torch.Tensor, received: torch.Tensor, decoded: torch.Tensor, name: str) -> None:
    """Visualize the encoding, transmission with errors, and decoding process."""
    plt.figure(figsize=(12, 3))

    # Plot message bits
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(message.view(1, -1), cmap="binary", aspect="auto")
    ax1.set_title(f"Original Message\n({len(message)} bits)")
    ax1.set_yticks([])

    # Plot codeword bits
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(codeword.view(1, -1), cmap="binary", aspect="auto")
    ax2.set_title(f"Encoded Codeword\n({len(codeword)} bits)")
    ax2.set_yticks([])

    # Plot received bits with errors
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(received.view(1, -1), cmap="binary", aspect="auto")
    ax3.set_title(f"Received with Errors\n({(codeword != received).sum().item()} errors)")
    ax3.set_yticks([])

    # Plot decoded message
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(decoded.view(1, -1), cmap="binary", aspect="auto")
    is_correct = torch.all(message == decoded)
    ax4.set_title(f"Decoded Message\n({'Correct' if is_correct else 'Incorrect'})")
    ax4.set_yticks([])

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()


def introduce_errors(codeword: torch.Tensor, error_positions: list[int] | None = None, num_errors: int | None = None, error_prob: float | None = None) -> torch.Tensor:
    """Introduce errors to a codeword by flipping bits at specific positions, a random number of
    bits, or using a probabilistic model."""
    received = codeword.clone()

    if error_positions is not None:
        # Flip bits at specific positions
        for pos in error_positions:
            received[pos] = 1 - received[pos]
    elif num_errors is not None:
        # Flip a specific number of random bits
        if num_errors > 0:
            positions = torch.randperm(len(codeword))[:num_errors]
            for pos in positions:
                received[pos] = 1 - received[pos]
    elif error_prob is not None:
        # Flip each bit with probability error_prob
        flip_mask = torch.rand(codeword.shape) < error_prob
        received[flip_mask] = 1 - received[flip_mask]

    return received


# %%
# Introduction to FEC Decoding
# -----------------------------------------------------------------------------------
#
# Forward Error Correction (FEC) allows receivers to correct errors in transmitted data
# without requesting retransmission. This is achieved by adding redundancy during encoding.
# The decoder's job is to exploit this redundancy to recover the original message.

# %%
# Part 1: Basic FEC Decoding Concepts
# -----------------------------------------------------------------------------------
#
# Let's start with basic decoding approaches for simple codes.

# %%
# Repetition Code with Majority Logic Decoding
# -----------------------------------------------------------------------------------
#
# For repetition codes, we can use majority logic decoding to recover the original message.
rep_encoder = RepetitionCodeEncoder(repetitions=3)
message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
codeword = rep_encoder(message)

# Simulate errors in transmission (flip some bits)
error_positions = [2, 7, 12]  # Error in 1st bit of 2nd message bit, 1st bit of 3rd message bit, etc.
received = introduce_errors(codeword, error_positions=error_positions)

# Original message: [1, 0, 1, 1, 0]
# Encoded codeword: [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
# Received with errors: [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0] (errors at positions [2, 7, 12])


# Majority logic decoding for repetition code
def repetition_decode(received, repetitions=3):
    """Simple majority logic decoder for repetition codes."""
    n = len(received)
    k = n // repetitions
    decoded = torch.zeros(k, dtype=received.dtype)  # Ensure matching dtype

    for i in range(k):
        # Get the repeated bits for message bit i
        repeated = received[i * repetitions : (i + 1) * repetitions]
        # Majority vote
        if repeated.sum() > repetitions / 2:
            decoded[i] = 1.0  # Use explicit float value for consistency

    return decoded


decoded = repetition_decode(received, repetitions=3)
# Decoded message: [1, 0, 1, 1, 0]
# Correct?: True

visualize_error_correction(message, codeword, received, decoded, "Repetition Code (3x) with Majority Logic Decoding")

# %%
# Single Parity Check Decoding
# ---------------------------------------------------
#
# Single parity check codes can only detect single errors, but with soft-decision decoding,
# we can make educated guesses about the most likely correct message.

spc_encoder = SingleParityCheckCodeEncoder(dimension=4)  # Specify the dimension parameter
message = torch.tensor([1.0, 0.0, 1.0, 1.0])
codeword = spc_encoder(message)

# Simulate a single error
received = introduce_errors(codeword, error_positions=[1])

print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"Received with errors: {received.int().tolist()}")

# Instantiate the soft-decision decoder for single parity check codes
spc_decoder = WagnerSoftDecisionDecoder(spc_encoder)

# Convert to bipolar form for soft decision decoding (-1 for 0, +1 for 1)
soft_received = 2.0 * received - 1.0  # Use explicit float values

# Apply soft-decision decoding
soft_decoded = spc_decoder(soft_received)

# Convert back to binary
decoded = (soft_decoded > 0).float()

# Decoded message: [1, 0, 1, 1]
# Correct?: True
#
# Note: Single parity check codes can only detect, not correct errors in general.
# However, with soft information and the Wagner decoder, we can make error correction more likely.

# %%
# Part 2: Hard-Decision vs. Soft-Decision Decoding
# -----------------------------------------------------------------------------------
#
# Now we'll explore the difference between hard-decision and soft-decision decoding.

# %%
# Soft-Decision Decoding with Wagner's Algorithm
# ----------------------------------------------------------------------------------------
#
# Wagner's algorithm is a soft-decision decoder for single parity check codes.

# First, create a simple single parity check code
spc_encoder = SingleParityCheckCodeEncoder(dimension=5)  # Specify the dimension parameter
message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
codeword = spc_encoder(message)

# Simulate transmission over an AWGN channel
awgn_channel = AWGNChannel(snr_db=3.0)  # Signal-to-noise ratio in dB

# Pass the codeword through the AWGN channel
# Convert binary {0,1} to bipolar {-1,+1} for AWGN channel
bipolar_codeword = 2.0 * codeword - 1.0  # Use explicit float values
received_soft = awgn_channel(bipolar_codeword)

# Print the received soft values
print(f"Original message: {message.int().tolist()}")
print(f"Original codeword: {codeword.int().tolist()}")
print(f"Received soft values: {received_soft.tolist()}")

# Create the soft-decision decoder
wagner_decoder = WagnerSoftDecisionDecoder(spc_encoder)

# Decode using soft information
soft_decoded = wagner_decoder(received_soft)

# Convert back to binary for the final decoded message
# Note: Wagner decoder returns soft values for the message part only
decoded = (soft_decoded > 0).float()

print(f"Decoded message: {decoded.int().tolist()}")
print(f"Correct?: {torch.all(message == decoded).item()}")

print("\nAdvantage of soft-decision decoding:")
print("- Uses reliability (confidence) information from the channel")
print("- Typically provides 2-3 dB coding gain over hard-decision decoding")
print("- Better performance in Gaussian noise channels (AWGN)")

# %%
# Part 3: Syndrome-Based Decoding
# -----------------------------------------------------------------------------------
#
# Syndrome-based decoding is a common technique for linear block codes.

print("\n=========== Part 3: Syndrome-Based Decoding ===========")

# %%
# Hamming Code with Syndrome Lookup Decoding
# --------------------------------------------------------------------------------
#
# Syndrome decoding uses a lookup table to map error syndromes to error patterns.

print("\n3.1 Hamming Code with Syndrome Lookup Table Decoding")

# Create a Hamming(7,4) code
hamming_encoder = HammingCodeEncoder(mu=3)  # mu=3 creates a (7,4) Hamming code
message = torch.tensor([1.0, 0.0, 1.0, 1.0])
codeword = hamming_encoder(message)

# Create a syndrome lookup decoder for this code
syndrome_decoder = SyndromeLookupDecoder(hamming_encoder)

# Print information about the decoder
print_decoder_info(syndrome_decoder, "Hamming(7,4) Syndrome Decoder")

# Introduce errors (Hamming codes can correct 1 error)
received_1_error = introduce_errors(codeword, error_positions=[2])
received_2_errors = introduce_errors(codeword, error_positions=[2, 5])

print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"Received (1 error): {received_1_error.int().tolist()}")
print(f"Received (2 errors): {received_2_errors.int().tolist()}")

# Decode both received words
decoded_1_error = syndrome_decoder(received_1_error)
decoded_2_errors = syndrome_decoder(received_2_errors)

print(f"Decoded (1 error): {decoded_1_error.int().tolist()} (Correct: {torch.all(message == decoded_1_error).item()})")
print(f"Decoded (2 errors): {decoded_2_errors.int().tolist()} (Correct: {torch.all(message == decoded_2_errors).item()})")

# Show how syndrome decoding works
if hasattr(syndrome_decoder, "parity_check_matrix") and hasattr(syndrome_decoder, "syndrome_table"):
    H = syndrome_decoder.parity_check_matrix
    print("\nParity Check Matrix H:")
    print(H.int().numpy())

    print("\nSyndrome Calculation for 1-error case:")
    syndrome = torch.matmul(received_1_error, H.t()) % 2
    print(f"Syndrome = {syndrome.int().tolist()}")

    print("\nSyndrome Table (mapping from syndrome to error pattern):")
    for syn, error_pattern in list(syndrome_decoder.syndrome_table.items())[:5]:
        print(f"Syndrome {syn} -> Error pattern {error_pattern}")
    print("... (more entries)")

# Visualize the error correction process
visualize_error_correction(message, codeword, received_1_error, decoded_1_error, "Hamming(7,4) with Syndrome Decoding (1 error)")
visualize_error_correction(message, codeword, received_2_errors, decoded_2_errors, "Hamming(7,4) with Syndrome Decoding (2 errors - fails)")

# %%
# Part 4: Advanced Algebraic Decoders
# -----------------------------------------------------------------------------------
#
# More powerful codes require more sophisticated decoding algorithms.

print("\n=========== Part 4: Advanced Algebraic Decoders ===========")

# %%
# BCH Code with Berlekamp-Massey Algorithm
# --------------------------------------------------------------------------
#
# The Berlekamp-Massey algorithm is used to decode BCH and Reed-Solomon codes.

print("\n4.1 BCH Code with Berlekamp-Massey Algorithm")

# Create a BCH(15,7) code that can correct up to 2 errors
bch_encoder = BCHCodeEncoder(mu=4, delta=5)  # GF(2^4), minimum distance 5
message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
codeword = bch_encoder(message)

# Create a Berlekamp-Massey decoder for this code
bm_decoder = BerlekampMasseyDecoder(bch_encoder)
print_decoder_info(bm_decoder, "BCH(15,7) with Berlekamp-Massey Decoder")

# Introduce errors (up to 2 for BCH(15,7))
received_2_errors = introduce_errors(codeword, error_positions=[3, 10])

print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"Received (2 errors): {received_2_errors.int().tolist()}")

# Decode the received word
decoded = bm_decoder(received_2_errors)
print(f"Decoded: {decoded.int().tolist()}")
print(f"Correct?: {torch.all(message == decoded).item()}")

visualize_error_correction(message, codeword, received_2_errors, decoded, "BCH(15,7) with Berlekamp-Massey Decoding (2 errors)")

# %%
# BCH Code with Berlekamp-Massey Algorithm (Extended Example)
# -----------------------------------------------------------------------------------------------------------------
#
# Let's try a different BCH code with different error correction capability.

print("\n4.2 BCH Code with Berlekamp-Massey Algorithm (Extended Example)")

# For a 9-bit message with mu=4, we need to adjust message length
bch_encoder2 = BCHCodeEncoder(mu=4, delta=3)  # Creates BCH(15,11) code
# Extend message to 11 bits to match the code dimension
message = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # 11-bit message
codeword = bch_encoder2(message)

# Create a Berlekamp-Massey decoder for this code
bm_decoder2 = BerlekampMasseyDecoder(bch_encoder2)
print_decoder_info(bm_decoder2, "BCH(15,11) Decoder")

# Introduce errors (BCH(15,11) can correct up to 1 error)
received_3_errors = introduce_errors(codeword, error_positions=[2, 7, 12])

print(f"Original message: {message.int().tolist()}")
print(f"Encoded codeword: {codeword.int().tolist()}")
print(f"Received (3 errors): {received_3_errors.int().tolist()}")

# Decode the received word
decoded = bm_decoder2(received_3_errors)
print(f"Decoded: {decoded.int().tolist()}")
print(f"Correct?: {torch.all(message == decoded).item()}")

visualize_error_correction(message, codeword, received_3_errors, decoded, "BCH(15,11) with Berlekamp-Massey Decoding (3 errors)")

print("\nAdvantages of BCH codes:")
print("- Excellent for burst error correction (common in storage and communications)")
print("- Support a wide range of block lengths and code rates")
print("- Widely used in digital communications, data storage, and satellite systems")

# %%
# Part 5: Maximum Likelihood Decoding
# -----------------------------------------------------------------------------------
#
# Maximum likelihood decoding finds the most likely transmitted codeword.

print("\n=========== Part 5: Maximum Likelihood Decoding ===========")

# %%
# Brute Force Maximum Likelihood Decoder
# ------------------------------------------------------------------------
#
# For small codes, we can exhaustively check all possible codewords.

print("\n5.1 Maximum Likelihood Decoding with Brute Force Search")

# Create a small custom code for demonstration
G = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
custom_encoder = LinearBlockCodeEncoder(generator_matrix=G)
message = torch.tensor([1.0, 1.0, 0.0])
codeword = custom_encoder(message)

# Create maximum likelihood decoder
ml_decoder = BruteForceMLDecoder(custom_encoder)
print_decoder_info(ml_decoder, "Small (6,3) Code with ML Decoder")

# Simulate transmission over an AWGN channel
awgn_channel = AWGNChannel(snr_db=1.0)  # Low SNR to make it challenging
bipolar_codeword = 2.0 * codeword - 1.0  # Use explicit float values
received_soft = awgn_channel(bipolar_codeword)

print(f"Original message: {message.int().tolist()}")
print(f"Original codeword: {codeword.int().tolist()}")
print(f"Received soft values: {received_soft.tolist()}")

# Decode using ML decoder
decoded = ml_decoder(received_soft)
print(f"Decoded: {decoded.int().tolist()}")
print(f"Correct?: {torch.all(message == decoded).item()}")

print("\nAdvantages of Maximum Likelihood decoding:")
print("- Optimal performance (minimizes probability of word error)")
print("- Can use soft information from the channel")
print("- Works with any code structure")
print("\nDisadvantages:")
print("- Exponential complexity in code dimension (2^k potential messages)")
print("- Only practical for small codes")

# %%
# Part 6: Performance Evaluation
# -----------------------------------------------------------------------------------
#
# Let's compare the performance of different decoders in noisy channels.

print("\n=========== Part 6: Performance Evaluation ===========")

# %%
# Performance Comparison in AWGN Channel
# ------------------------------------------------------------------------
#
# We'll test different codes and decoders across various SNR levels.

print("\n6.1 Performance Comparison in AWGN Channel")

# Define several code and decoder pairs to compare
code_decoders = {
    "Repetition(3)": (RepetitionCodeEncoder(repetitions=3), lambda enc, x: repetition_decode(x, repetitions=3)),
    "Hamming(7,4)": (HammingCodeEncoder(mu=3), lambda enc, x: SyndromeLookupDecoder(enc)(x)),
    "BCH(15,7)": (BCHCodeEncoder(mu=4, delta=5), lambda enc, x: BerlekampMasseyDecoder(enc)(x)),
    "Golay(23,12)": (GolayCodeEncoder(), lambda enc, x: SyndromeLookupDecoder(enc)(x)),
}

# Test message for each code (appropriately sized)
test_messages = {"Repetition(3)": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0]), "Hamming(7,4)": torch.tensor([1.0, 0.0, 1.0, 1.0]), "BCH(15,7)": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]), "Golay(23,12)": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])}

# SNR range to test (in dB)
snr_range = np.arange(0, 11, 1)

# Number of Monte Carlo trials per SNR point
num_trials = 100

# Store results
results = {}

print("Running simulations... (this may take a moment)")

# Type-safe dictionary access and tensor creation
for name, (encoder, decoder_fn) in code_decoders.items():
    message = test_messages[name]  # Access dictionary with string key
    codeword = encoder(message)
    bipolar_codeword = 2.0 * codeword - 1.0  # Use explicit float values

    ber_values = []  # Bit Error Rate
    fer_values = []  # Frame Error Rate

    # Test each SNR point
    for snr_db in snr_range:
        awgn_channel = AWGNChannel(snr_db=float(snr_db))  # Explicit float conversion
        bit_errors = 0
        frame_errors = 0

        # Run Monte Carlo trials
        for _ in range(num_trials):
            # Transmit over AWGN channel
            received_soft = awgn_channel(bipolar_codeword)

            # Hard decision for binary decoders
            received_hard = (received_soft > 0).float()

            # Decode
            decoded = decoder_fn(encoder, received_hard)

            # Count errors
            if not torch.all(message == decoded):
                frame_errors += 1
                bit_errors += (message != decoded).sum().item()

        # Calculate error rates - ensure float division
        ber = bit_errors / (num_trials * len(message))
        fer = frame_errors / float(num_trials)  # Explicit float conversion

        ber_values.append(ber)
        fer_values.append(fer)

    results[name] = {"ber": ber_values, "fer": fer_values}

# %%
# Plot BER results
# --------------------------
#
# Let's visualize the bit error rate performance.

plt.figure(figsize=(10, 6))
for name, data in results.items():
    plt.semilogy(snr_range, data["ber"], marker="o", label=name)

plt.grid(True, which="both", ls="--")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Decoder Performance in AWGN Channel")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot FER results
# --------------------------
#
# Now let's visualize the frame error rate performance.

plt.figure(figsize=(10, 6))
for name, data in results.items():
    plt.semilogy(snr_range, data["fer"], marker="o", label=name)

plt.grid(True, which="both", ls="--")
plt.xlabel("SNR (dB)")
plt.ylabel("Frame Error Rate (FER)")
plt.title("Decoder Performance in AWGN Channel")
plt.legend()
plt.tight_layout()
plt.show()

# Print results at SNR=5dB
print("\nDecoder Performance at 5dB SNR:")
snr_idx = np.where(snr_range == 5)[0][0]
for name, data in results.items():
    print(f"{name}: BER = {data['ber'][snr_idx]:.4e}, FER = {data['fer'][snr_idx]:.4e}")

# %%
# Conclusion
# -----------------
#
# This tutorial demonstrated various FEC decoders from the kaira library.
#
# Key takeaways:
#
# 1. Different decoders are optimized for different code structures and error patterns
# 2. The choice of decoder depends on the code used and computational constraints
# 3. Soft-decision decoding generally outperforms hard-decision decoding
# 4. More powerful codes (BCH, Reed-Solomon, Golay) offer better error correction
# 5. There is always a trade-off between code rate, error correction, and complexity
#
# Pair these decoders with their corresponding encoders from kaira.models.fec.encoders
# for a complete Forward Error Correction solution.
