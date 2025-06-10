"""
=================================================================================================
UplinkMACChannel Integration Example
=================================================================================================

This example demonstrates how to use the UplinkMACChannel in an end-to-end communication system.
The UplinkMACChannel handles per-user channel effects and signal combining for uplink Multiple
Access Channel scenarios.

Since UplinkMACChannel expects separate user signals (before combining), we create a custom
model that properly integrates encoders, UplinkMACChannel, and decoders.

Key Features Demonstrated:
- Using UplinkMACChannel for per-user channel modeling
- Comparing different channel configurations (shared vs per-user channels)
- Analyzing performance with varying numbers of users
- Demonstrating dynamic parameter updates during transmission
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Import necessary components
from kaira.channels import AWGNChannel, FlatFadingChannel, RayleighFadingChannel, UplinkMACChannel
from kaira.constraints import AveragePowerConstraint
from kaira.models.base import BaseModel
from kaira.models.components.mlp import MLPDecoder, MLPEncoder


class UplinkMACModel(BaseModel):
    """Custom model that properly integrates UplinkMACChannel with encoders and decoders.

    This model passes individual user signals to the UplinkMACChannel before combining, which is
    the intended usage pattern for UplinkMACChannel.
    """

    def __init__(self, encoders, decoder, uplink_channel, power_constraint, num_users):
        super().__init__()
        self.encoders = encoders
        self.decoder = decoder
        self.uplink_channel = uplink_channel
        self.power_constraint = power_constraint
        self.num_users = num_users

    def forward(self, user_messages, **kwargs):
        """Forward pass through the UplinkMACModel.

        Args:
            user_messages (List[torch.Tensor]): List of input messages, one per user.
                Each tensor should have shape (batch_size, message_dim).
            **kwargs: Additional keyword arguments passed to the channel and decoder.

        Returns:
            torch.Tensor: Reconstructed messages for all users combined.
                Shape: (batch_size, num_users * message_dim).
        """
        # 1. Encode each user's message
        encoded_signals = []
        for i in range(self.num_users):
            encoded_signal = self.encoders[i](user_messages[i])
            encoded_signals.append(encoded_signal)

        # 2. Apply power constraint to each user's signal separately
        constrained_signals = []
        for signal in encoded_signals:
            constrained_signal = self.power_constraint(signal)
            constrained_signals.append(constrained_signal)

        # 3. Pass separate user signals through UplinkMACChannel
        # This is where UplinkMACChannel applies per-user channel effects and combines
        received_signal = self.uplink_channel(constrained_signals, **kwargs)

        # 4. Convert complex signal to real if necessary (for MLP decoder compatibility)
        if received_signal.dtype.is_complex:
            # Convert complex to real by concatenating real and imaginary parts
            real_part = received_signal.real
            imag_part = received_signal.imag
            received_signal = torch.cat([real_part, imag_part], dim=-1)

        # 5. Decode the received combined signal
        reconstructed = self.decoder(received_signal, **kwargs)

        return reconstructed


# %%
# Setup Parameters
# --------------------------------------------------------------------------
print("=== UplinkMACChannel Integration Example ===\n")

# System parameters
num_users = 3
message_dim = 8
code_dim = 16
batch_size = 64

# Channel parameters
coherence_time = 10
avg_noise_power = 0.1
snr_db = 12.0

print("System Configuration:")
print(f"- Number of users: {num_users}")
print(f"- Message dimension: {message_dim}")
print(f"- Code dimension: {code_dim}")
print(f"- Batch size: {batch_size}")
print(f"- SNR: {snr_db} dB")
print(f"- Coherence time: {coherence_time}")
print(f"- Average noise power: {avg_noise_power}\n")

# %%
# Scenario 1: Shared Channel Configuration
# --------------------------------------------------------------------------
print("--- Scenario 1: Shared Channel Configuration ---")

# Create a shared Rayleigh fading channel for all users
shared_channel = RayleighFadingChannel(coherence_time=coherence_time, avg_noise_power=avg_noise_power)

# Create UplinkMACChannel with shared channel
uplink_mac_shared = UplinkMACChannel(user_channels=shared_channel, num_users=num_users, combine_method="sum")  # Single channel shared by all users

# Create encoders and decoder for the MAC model
encoders = nn.ModuleList([MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[32, 24]) for _ in range(num_users)])

joint_decoder = MLPDecoder(in_features=code_dim * 2, out_features=message_dim * num_users, hidden_dims=[48, 32])  # Double size to handle real + imaginary parts

# Create power constraint
power_constraint = AveragePowerConstraint(average_power=1.0)

# Create the integrated MAC model
mac_model_shared = UplinkMACModel(encoders=encoders, decoder=joint_decoder, uplink_channel=uplink_mac_shared, power_constraint=power_constraint, num_users=num_users)

# Generate test messages
user_messages = [torch.randn(batch_size, message_dim) for _ in range(num_users)]

# Test the integrated model
with torch.no_grad():
    reconstructed_tensor = mac_model_shared(user_messages, snr=snr_db)

# Calculate and display performance
reconstructed_reshaped = reconstructed_tensor.view(batch_size, num_users, message_dim)
reconstructed_messages = [reconstructed_reshaped[:, i, :] for i in range(num_users)]

mse_loss = nn.MSELoss()
total_mse_shared = 0
for i in range(num_users):
    user_mse = mse_loss(reconstructed_messages[i], user_messages[i]).item()
    total_mse_shared += user_mse
    print(f"User {i+1} MSE (Shared Channel): {user_mse:.6f}")

avg_mse_shared = total_mse_shared / num_users
print(f"Average MSE (Shared Channel): {avg_mse_shared:.6f}\n")

# %%
# Scenario 2: Per-User Channel Configuration
# --------------------------------------------------------------------------
print("--- Scenario 2: Per-User Channel Configuration ---")

# Create individual channels for each user with different characteristics
per_user_channels = [
    FlatFadingChannel(fading_type="rayleigh", coherence_time=coherence_time, avg_noise_power=avg_noise_power),
    FlatFadingChannel(fading_type="rician", coherence_time=coherence_time, avg_noise_power=avg_noise_power, k_factor=3.0),
    AWGNChannel(snr_db=snr_db),  # One user has AWGN channel
]

# Create UplinkMACChannel with per-user channels
uplink_mac_per_user = UplinkMACChannel(user_channels=per_user_channels, num_users=num_users, combine_method="sum")

# Create new encoders and decoder for this scenario
encoders_per_user = nn.ModuleList([MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[32, 24]) for _ in range(num_users)])

joint_decoder_per_user = MLPDecoder(in_features=code_dim * 2, out_features=message_dim * num_users, hidden_dims=[48, 32])  # Double size to handle real + imaginary parts

# Create the integrated MAC model with per-user channels
mac_model_per_user = UplinkMACModel(encoders=encoders_per_user, decoder=joint_decoder_per_user, uplink_channel=uplink_mac_per_user, power_constraint=power_constraint, num_users=num_users)

# Test the per-user channel model
with torch.no_grad():
    reconstructed_tensor_per_user = mac_model_per_user(user_messages, snr=snr_db)

# Calculate and display performance
reconstructed_reshaped_per_user = reconstructed_tensor_per_user.view(batch_size, num_users, message_dim)
reconstructed_messages_per_user = [reconstructed_reshaped_per_user[:, i, :] for i in range(num_users)]

total_mse_per_user = 0
for i in range(num_users):
    user_mse = mse_loss(reconstructed_messages_per_user[i], user_messages[i]).item()
    total_mse_per_user += user_mse
    print(f"User {i+1} MSE (Per-User Channel): {user_mse:.6f}")

avg_mse_per_user = total_mse_per_user / num_users
print(f"Average MSE (Per-User Channel): {avg_mse_per_user:.6f}\n")

# %%
# Scenario 3: Combining Methods Comparison
# --------------------------------------------------------------------------
print("--- Scenario 3: Combining Methods Comparison ---")

combining_methods = ["sum", "weighted_sum"]
combine_results = {}

for method in combining_methods:
    print(f"Testing combining method: {method}")

    # Create channels for this test
    test_channels = [FlatFadingChannel(fading_type="rayleigh", coherence_time=coherence_time, avg_noise_power=avg_noise_power) for _ in range(num_users)]

    # Create UplinkMACChannel with specific combining method
    uplink_test = UplinkMACChannel(user_channels=test_channels, num_users=num_users, combine_method=method)

    # Create MAC model
    encoders_test = nn.ModuleList([MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[32, 24]) for _ in range(num_users)])

    decoder_test = MLPDecoder(in_features=code_dim * 2, out_features=message_dim * num_users, hidden_dims=[48, 32])  # Double size to handle real + imaginary parts

    mac_test = UplinkMACModel(encoders=encoders_test, decoder=decoder_test, uplink_channel=uplink_test, power_constraint=power_constraint, num_users=num_users)

    # Test performance
    with torch.no_grad():
        reconstructed_test = mac_test(user_messages, snr=snr_db)

    # Calculate MSE
    reconstructed_test_reshaped = reconstructed_test.view(batch_size, num_users, message_dim)
    total_mse_test = 0
    for i in range(num_users):
        user_reconstructed = reconstructed_test_reshaped[:, i, :]
        user_mse = mse_loss(user_reconstructed, user_messages[i]).item()
        total_mse_test += user_mse

    avg_mse_test = total_mse_test / num_users
    combine_results[method] = avg_mse_test
    print(f"  Average MSE with {method}: {avg_mse_test:.6f}")

# %%
# Scenario 4: Dynamic Parameter Updates
# --------------------------------------------------------------------------
print("\n--- Scenario 4: Dynamic Parameter Updates ---")

# Create UplinkMACChannel for dynamic updates
dynamic_channels = [FlatFadingChannel(fading_type="rayleigh", coherence_time=coherence_time, avg_noise_power=avg_noise_power) for _ in range(num_users)]

uplink_mac_dynamic = UplinkMACChannel(user_channels=dynamic_channels, num_users=num_users, combine_method="sum")

# Create MAC model for dynamic scenario
encoders_dynamic = nn.ModuleList([MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[32, 24]) for _ in range(num_users)])

decoder_dynamic = MLPDecoder(in_features=code_dim * 2, out_features=message_dim * num_users, hidden_dims=[48, 32])  # Double size to handle real + imaginary parts

mac_model_dynamic = UplinkMACModel(encoders=encoders_dynamic, decoder=decoder_dynamic, uplink_channel=uplink_mac_dynamic, power_constraint=power_constraint, num_users=num_users)

# Simulate transmission with changing conditions
snr_values = [5.0, 10.0, 15.0, 20.0]
mse_results_dynamic = []

for snr in snr_values:
    print(f"Testing at SNR = {snr} dB...")

    # Update individual channel parameters dynamically
    new_noise_power = 0.1 * (10 ** (-snr / 10))  # Adjust noise based on SNR

    # Update each user channel's parameters if they support it
    for i, channel in enumerate(uplink_mac_dynamic.user_channels):
        if hasattr(channel, "avg_noise_power"):
            channel.avg_noise_power = new_noise_power
        # For channels that have update methods, you could call them here
        # For simplicity, we'll just update the noise power directly

    # Generate new test messages
    test_messages = [torch.randn(batch_size, message_dim) for _ in range(num_users)]

    with torch.no_grad():
        reconstructed_dynamic = mac_model_dynamic(test_messages, snr=snr)

    # Calculate performance
    reconstructed_dynamic_reshaped = reconstructed_dynamic.view(batch_size, num_users, message_dim)
    total_mse_dynamic = 0
    for i in range(num_users):
        user_reconstructed = reconstructed_dynamic_reshaped[:, i, :]
        user_mse = mse_loss(user_reconstructed, test_messages[i]).item()
        total_mse_dynamic += user_mse

    avg_mse_dynamic = total_mse_dynamic / num_users
    mse_results_dynamic.append(avg_mse_dynamic)
    print(f"  Average MSE: {avg_mse_dynamic:.6f}")

# %%
# Plotting Results
# --------------------------------------------------------------------------
plt.figure(figsize=(12, 8))

# Plot 1: Shared vs Per-User Channels
plt.subplot(2, 2, 1)
configs = ["Shared Channel", "Per-User Channels"]
mse_values = [avg_mse_shared, avg_mse_per_user]
plt.bar(configs, mse_values, color=["blue", "orange"], alpha=0.7)
plt.ylabel("Average MSE")
plt.title("Shared vs Per-User Channel Configuration")
plt.yscale("log")
plt.grid(True, alpha=0.3, axis="y")

# Plot 2: Dynamic SNR Performance
plt.subplot(2, 2, 2)
plt.plot(snr_values, mse_results_dynamic, "o-", linewidth=2, markersize=8, color="green")
plt.xlabel("SNR (dB)")
plt.ylabel("Average MSE")
plt.title("Performance vs SNR (Dynamic Updates)")
plt.grid(True, alpha=0.3)
plt.yscale("log")

# Plot 3: Combining Methods Comparison
plt.subplot(2, 2, 3)
methods = list(combine_results.keys())
mse_combine_values = list(combine_results.values())
plt.bar(methods, mse_combine_values, color=["blue", "red"], alpha=0.7)
plt.xlabel("Combining Method")
plt.ylabel("Average MSE")
plt.title("Performance by Combining Method")
plt.yscale("log")
plt.grid(True, alpha=0.3, axis="y")

# Plot 4: Signal Power Analysis
plt.subplot(2, 2, 4)
# Generate test signals for power analysis
test_signals = [torch.randn(batch_size, code_dim) for _ in range(num_users)]
with torch.no_grad():
    output = uplink_mac_shared(test_signals, snr=snr_db)
    input_power = [torch.mean(torch.abs(sig) ** 2).item() for sig in test_signals]
    output_power = torch.mean(torch.abs(output) ** 2).item()

plt.bar(range(1, num_users + 1), input_power, alpha=0.7, label="Input Power", color="blue")
plt.axhline(y=output_power, color="red", linestyle="--", label="Output Power", linewidth=2)
plt.xlabel("User")
plt.ylabel("Signal Power")
plt.title("Signal Power Analysis")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Summary and Conclusions
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("INTEGRATION SUMMARY")
print("=" * 70)
print("Successfully demonstrated UplinkMACChannel integration:")
print("")
print("1. Shared Channel Configuration:")
print(f"   - Average MSE: {avg_mse_shared:.6f}")
print("   - Uses single channel instance for all users")
print("")
print("2. Per-User Channel Configuration:")
print(f"   - Average MSE: {avg_mse_per_user:.6f}")
print("   - Individual channels allow user-specific modeling")
print("")
print("3. Combining Methods:")
for method, mse in combine_results.items():
    print(f"   - {method}: MSE = {mse:.6f}")
print("")
print("4. Dynamic Parameter Updates:")
print(f"   - SNR range: {min(snr_values)}-{max(snr_values)} dB")
print("   - Real-time parameter adjustment capability")
print("")
print("Key Benefits of UplinkMACChannel:")
print("- Composition-based design allows flexible channel modeling")
print("- Per-user channel effects with proper signal combining")
print("- Support for both shared and per-user channel configurations")
print("- Dynamic parameter updates during operation")
print("- Comprehensive performance analysis capabilities")
print("=" * 70)
