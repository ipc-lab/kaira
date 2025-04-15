"""
=================================================================================
Channel Code Model with Modulation/Demodulation
=================================================================================

This example demonstrates how to use the ChannelCodeModel for binary data transmission
over a noisy channel. The model applies channel coding, modulation, channel simulation,
demodulation, and decoding to transmit data reliably over noisy channels.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# -------------------------------
# First, we import necessary modules and set random seeds for reproducibility.
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.constraints.power import TotalPowerConstraint

# Import BER metric from kaira.metrics
from kaira.metrics import BER
from kaira.models import ChannelCodeModel
from kaira.models.binary import MajorityVoteDecoder, RepetitionEncoder
from kaira.modulations import BPSKDemodulator, BPSKModulator

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Creating Binary Test Data
# --------------------------------------------
# We'll create some binary test data to transmit through our channel.

# Create random binary data
batch_size = 100
message_length = 16
x = torch.randint(0, 2, (batch_size, message_length)).float()

print(f"Input data shape: {x.shape}")
print(f"Example message: {x[0].int().tolist()}")

# %%
# Building the Channel Code Model
# ------------------------------------------------------
# We'll create a simple channel code model using repetition coding.

# Repetition factor (each bit is repeated this many times)
rep_factor = 3

# Create model components
encoder = RepetitionEncoder(repetition_factor=rep_factor)
constraint = TotalPowerConstraint(total_power=1.0)
modulator = BPSKModulator()
channel = AWGNChannel(snr_db=5.0)  # Initialize with a default SNR value
demodulator = BPSKDemodulator()  # BPSKDemodulator automatically handles soft output when noise_var is provided
decoder = MajorityVoteDecoder(repetition_factor=rep_factor)

# Build the channel code model
model = ChannelCodeModel(encoder=encoder, constraint=constraint, modulator=modulator, channel=channel, demodulator=demodulator, decoder=decoder)

# %%
# Simulating Transmission at Various SNRs
# -------------------------------------------------------------------------
# We'll simulate transmission at different Signal-to-Noise Ratios (SNR).

# SNR values in dB
snr_values = np.arange(0, 11, 1)
bit_error_rates = []

# Initialize the BER metric with mean reduction
ber_metric = BER()

for snr in snr_values:
    # Update the channel's SNR parameter
    model.channel.snr_db = snr

    # Pass the binary data through our model without passing snr as a parameter
    with torch.no_grad():
        output = model(x)
        # Extract the decoded output from the model's return dictionary
        decoded = output["final_output"]

    # Calculate bit error rate using the BER metric
    ber = ber_metric(decoded.flatten(), x.flatten()).item()
    bit_error_rates.append(ber)
    print(f"SNR: {snr} dB, Bit Error Rate: {ber:.4f}")

# %%
# Visualizing Bit Error Rate vs SNR
# -------------------------------------------------------------
# Let's plot the bit error rate as a function of SNR.

plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, bit_error_rates, "o-", linewidth=2)
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Bit Error Rate vs. SNR for Repetition Code (R=1/3)")
plt.ylim([1e-3, 1])

# %%
# Visualizing Transmission Example
# ------------------------------------------------------------
# Let's see an example of a message being transmitted through the channel.

# Take a single message
test_message = x[0:1]
print(f"Original message: {test_message[0].int().tolist()}")

# Show the encoded message (repetition coding)
with torch.no_grad():
    # Get the encoded message by accessing model's encoder
    encoded = encoder(test_message)
    print(f"Encoded message: {encoded[0].int().tolist()}")

    # Get the modulated signal
    modulated = modulator(constraint(encoded))

    # Show transmission at different SNRs
    plt.figure(figsize=(12, 8))

    for i, snr in enumerate([0, 3, 6, 10]):
        # Update the channel's SNR parameter
        channel.snr_db = snr

        # Pass through channel at this SNR
        received = channel(modulated)

        # Demodulate
        demodulated = demodulator(received)

        # Decode
        decoded = decoder(demodulated)

        # Calculate errors for this example
        errors = (decoded != test_message).sum().item()

        # Plot the signals
        plt.subplot(4, 1, i + 1)

        # Extract real parts for plotting to avoid ComplexWarning
        modulated_plot = modulated[0].real if torch.is_complex(modulated[0]) else modulated[0]
        received_plot = received[0].real if torch.is_complex(received[0]) else received[0]

        # Plot transmitted signal
        plt.stem(np.arange(len(modulated_plot)), modulated_plot.numpy(), markerfmt="bo", linefmt="b-", basefmt="b-", label="Transmitted")

        # Plot received signal
        plt.plot(np.arange(len(received_plot)), received_plot.numpy(), "r.", alpha=0.7, label="Received")

        plt.title(f"SNR = {snr} dB, Errors: {errors}/{message_length}")
        plt.grid(True, alpha=0.3)

        if i == 0:
            plt.legend()

        if i == 3:
            plt.xlabel("Symbol Index")

    plt.tight_layout()

# %%
# Comparing Different Coding Schemes
# -------------------------------------------------------------
# In this section, we compare the performance of different coding schemes.


# Function to evaluate a model across different SNRs
def evaluate_model(model, test_data, snr_values):
    """Evaluate a channel code model over a range of SNRs.

    Args:
        model (ChannelCodeModel): The channel code model to evaluate.
        test_data (torch.Tensor): The input binary data for testing.
        snr_values (list or np.ndarray): A list or array of SNR values in dB.

    Returns:
        list: A list of Bit Error Rates (BER) corresponding to each SNR value.
    """
    ber_values = []
    # Create BER metric with mean reduction
    ber_metric = BER(reduction="mean")

    for snr in snr_values:
        # Update the model's channel SNR parameter
        model.channel.snr_db = snr

        with torch.no_grad():
            output = model(test_data)
            decoded = output["final_output"]

        # Calculate bit error rate using the BER metric
        ber = ber_metric(decoded, test_data).item()
        ber_values.append(ber)

    return ber_values


# Example of how you would compare different coding schemes
# (not executed in this example for simplicity)
# # Create models with different repetition factors
# models = {
#     "No coding": create_uncoded_model(),
#     "Rep-3": create_repetition_model(3),
#     "Rep-5": create_repetition_model(5),
#     "Hamming(7,4)": create_hamming_model()
# }
#
# # Evaluate each model
# results = {}
# for name, model in models.items():
#     results[name] = evaluate_model(model, x, snr_values)
#
# # Plot comparison
# plt.figure(figsize=(10, 6))
# for name, ber in results.items():
#     plt.semilogy(snr_values, ber, '-o', label=name)
#
# plt.grid(True, which="both", linestyle='--', alpha=0.7)
# plt.xlabel('SNR (dB)')
# plt.ylabel('Bit Error Rate (BER)')
# plt.title('Comparison of Different Coding Schemes')
# plt.legend()
# plt.ylim([1e-5, 1])

# %%
# Conclusion
# --------------------
# This example demonstrated how to use the ChannelCodeModel for reliable
# transmission of binary data over noisy channels. We showed how to:
#
# 1. Set up a complete channel coding pipeline with encoding, modulation,
#    channel simulation, demodulation, and decoding
# 2. Evaluate the performance in terms of bit error rate at different SNRs
# 3. Visualize the transmission process
#
# In practice, you would use more sophisticated coding schemes like LDPC codes,
# Turbo codes, or Polar codes for better error correction performance.
