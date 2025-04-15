"""
=================================================================================================
Multiple Access Channel Model for Joint Encoding
=================================================================================================

This example demonstrates how to use the MultipleAccessChannelModel for transmitting
information from multiple users over a shared channel. This model simulates
scenarios where multiple transmitters send signals simultaneously and a single
receiver tries to recover all messages.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Import necessary channel and constraint components
from kaira.channels import AWGNChannel
from kaira.constraints import AveragePowerConstraint
from kaira.models import MultipleAccessChannelModel
from kaira.models.components.mlp import MLPDecoder, MLPEncoder

# %%
# Model Setup
# --------------------------------------------------------------------------
# Define parameters for the simulation
num_users = 2  # Number of users transmitting simultaneously
message_dim = 10  # Dimensionality of each user's message vector
code_dim = 20  # Dimensionality of the signal transmitted over the channel (shared resource)
batch_size = 128  # Number of samples to process in parallel

# Create individual encoders for each user.
# Each encoder maps a user's message to a channel code.
encoders = nn.ModuleList([MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[50, 30]) for _ in range(num_users)])

# Create a single joint decoder.
# It receives the combined signal from the channel and attempts to decode messages for ALL users.
# Input dimension: code_dim (received signal dimension)
# Output dimension: message_dim * num_users (concatenated decoded messages)
joint_decoder = MLPDecoder(in_features=code_dim, out_features=message_dim * num_users, hidden_dims=[50, 30])

# Instantiate channel and power constraint
# The channel simulates the effects of the transmission medium (e.g., adding noise)
# Provide a default SNR value for initialization, it can be overridden in the forward pass
channel = AWGNChannel(snr_db=10.0)
# The power constraint ensures the transmitted signal adheres to power limits
power_constraint = AveragePowerConstraint(average_power=1.0)

# Instantiate the Multiple Access Channel Model, combining the encoders and the joint decoder.
# Add the channel and power_constraint arguments, and specify the decoder is shared.
mac_model = MultipleAccessChannelModel(
    encoders=encoders,
    decoder=joint_decoder,
    channel=channel,
    power_constraint=power_constraint,
    shared_decoder=True,  # Indicate that the single decoder instance is shared
)

# Generate some random messages for each user to simulate input data.
# This creates a list where each element is a batch of messages for one user.
user_messages = [torch.randn(batch_size, message_dim) for _ in range(num_users)]

# %%
# Single Transmission Example (Fixed SNR)
# --------------------------------------------------------------------------
# Simulate transmission at a fixed Signal-to-Noise Ratio (SNR).
fixed_snr = 10.0  # Example SNR in dB

# Perform the forward pass through the model.
# The model handles encoding, channel transmission (adding noise based on SNR), and decoding.
with torch.no_grad():  # Disable gradient calculation for inference
    reconstructed_messages_tensor = mac_model(user_messages, snr=fixed_snr)

# The decoder outputs a single tensor containing concatenated messages.
# Reshape the output tensor to separate messages for each user.
# Output shape: (batch_size, num_users * message_dim) -> (batch_size, num_users, message_dim)
reconstructed_messages_reshaped = reconstructed_messages_tensor.view(batch_size, num_users, message_dim)
# Split into a list, where each element corresponds to a user's reconstructed messages.
reconstructed_messages = [reconstructed_messages_reshaped[:, i, :] for i in range(num_users)]

# We'll use Mean Squared Error (MSE) to evaluate reconstruction quality.
mse_loss = nn.MSELoss(reduction="mean")  # Use mean reduction for average MSE per user

# Calculate MSE for each user at the fixed SNR
total_mse = 0
print(f"--- Results for SNR = {fixed_snr} dB ---")
for i in range(num_users):
    reconstructed_user_message = reconstructed_messages[i]
    # Calculate MSE between the original and reconstructed message for user i
    user_mse = mse_loss(reconstructed_user_message, user_messages[i]).item()  # Use .item() to get the scalar value
    total_mse += user_mse
    print(f"User {i+1} MSE: {user_mse:.6f}")

# Average MSE across all users
avg_mse = total_mse / num_users
print(f"Average MSE across users: {avg_mse:.6f}")
print("-" * (26 + len(str(fixed_snr))))


# %%
# Visualizing User Interference Effects (Optional - Uncomment to run)


def simulate_with_varying_users(base_model, max_users=4, snr=10.0, message_dim=10, code_dim=20, batch_size=128):
    """Simulates transmission with varying number of active users.

    Args:
        base_model (MultipleAccessChannelModel): The base MAC model with pre-defined components.
        max_users (int): The maximum number of users to simulate.
        snr (float): The fixed SNR (dB) for the simulation.
        message_dim (int): Dimensionality of each user's message.
        code_dim (int): Dimensionality of the channel code.
        batch_size (int): Number of samples per batch.

    Returns:
        list: A list of average MSE values for each number of active users.
    """
    mse_results = []
    original_encoders = base_model.encoders
    original_decoder = base_model.decoder
    # Get original channel and constraint from the base model
    original_channel = base_model.channel
    original_constraint = base_model.power_constraint
    len(original_encoders)

    for active_users in range(1, max_users + 1):
        print(f"Simulating with {active_users} active users...")
        # Select a subset of encoders
        current_encoders = nn.ModuleList(original_encoders[:active_users])

        # Adjust decoder output size (assuming it can handle variable output or is retrained/redefined)
        # For simplicity, let's assume the decoder structure is fixed but we only evaluate relevant outputs
        # A more realistic scenario might involve retraining or a flexible decoder architecture.
        # Here, we'll use the original decoder but only calculate loss for the active users' messages.

        # Create a temporary model with the subset of encoders and original components
        temp_model = MultipleAccessChannelModel(encoders=current_encoders, decoder=original_decoder, channel=original_channel, power_constraint=original_constraint)

        # Generate messages for the current number of active users
        user_subset = [torch.randn(batch_size, message_dim) for _ in range(active_users)]

        with torch.no_grad():
            # The model output will have shape (batch_size, original_num_users * message_dim)
            # We need to interpret the first 'active_users * message_dim' outputs
            reconstructed_all = temp_model(user_subset, snr=snr)

        # Calculate MSE for each active user
        total_mse = 0
        loss_fn = nn.MSELoss()
        for i in range(active_users):
            # Extract the portion of the output corresponding to user i
            start_idx = i * message_dim
            end_idx = (i + 1) * message_dim
            reconstructed_user_i = reconstructed_all[:, start_idx:end_idx]

            # Compare with the original message for user i
            user_mse = loss_fn(reconstructed_user_i, user_subset[i]).item()
            total_mse += user_mse

        avg_mse = total_mse / active_users
        mse_results.append(avg_mse)
        print(f"Average MSE for {active_users} users: {avg_mse:.6f}")

    return mse_results


# Uncomment the following block to run the user interference analysis:
#
# print("\n--- User Interference Analysis ---")
# # Simulate transmission at fixed SNR but varying number of users
# interference_results = simulate_with_varying_users(mac_model, max_users=4, snr=fixed_snr)
#
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 5), interference_results, "o-", linewidth=2)
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.xlabel("Number of Active Users")
# plt.ylabel("Average Mean Squared Error (MSE)")
# plt.title(f"Impact of Number of Users on Reconstruction (SNR={fixed_snr}dB)")
# plt.xticks(range(1, 5))
# plt.yscale("log")
# plt.tight_layout()
# plt.show()


# %%
# Training a Multiple Access Channel Model (Optional - Uncomment to run)
# --------------------------------------------------------------------------


# Modify train_mac_model to accept a fixed training_snr
def train_mac_model(model, optimizer, num_epochs=50, steps_per_epoch=100, batch_size=64, message_dim=10, num_users=2, training_snr=10.0):  # Use fixed training_snr
    """Trains the Multiple Access Channel model at a fixed SNR.

    Args:
        model (MultipleAccessChannelModel): The MAC model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        num_epochs (int): Number of training epochs.
        steps_per_epoch (int): Number of training steps per epoch.
        batch_size (int): Batch size for training.
        message_dim (int): Dimensionality of each user's message.
        num_users (int): Number of users in the model.
        training_snr (float): The fixed SNR (dB) to use during training.

    Returns:
        list: A list of average loss values for each epoch.
    """
    losses = []
    print(f"--- Starting Training at SNR = {training_snr} dB ---")
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            # Generate a batch of random messages
            batch_messages = [torch.randn(batch_size, message_dim) for _ in range(num_users)]

            # Forward pass using the fixed training SNR
            reconstructed = model(batch_messages, snr=training_snr)

            # Compute loss (MSE between original and reconstructed messages)
            loss = 0
            loss_fn = nn.MSELoss()
            for i in range(num_users):
                original_user_msg = batch_messages[i]
                # Extract the reconstructed message part for user i
                reconstructed_user_msg = reconstructed[:, i * message_dim : (i + 1) * message_dim]
                user_loss = loss_fn(reconstructed_user_msg, original_user_msg)
                loss += user_loss  # Sum losses from all users

            # Average loss across users for the backward pass
            loss = loss / num_users

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / steps_per_epoch
        losses.append(avg_epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

    print("Training finished.")
    print("-" * (26 + len(str(training_snr))))
    return losses


# Uncomment the following block to run the training:
#
# # Re-initialize model and optimizer for training
# encoders_train = nn.ModuleList([
#     MLPEncoder(in_features=message_dim, out_features=code_dim, hidden_dims=[50, 30])
#     for _ in range(num_users)
# ])
# joint_decoder_train = MLPDecoder(
#     in_features=code_dim,
#     out_features=message_dim * num_users,
#     hidden_dims=[50, 30]
# )
# # Instantiate channel and constraint for the training model
# # Provide a default SNR value for initialization
# channel_train = AWGNChannel(snr_db=10.0)
# power_constraint_train = AveragePowerConstraint(average_power=1.0)
#
# mac_model_train = MultipleAccessChannelModel(
#     encoders=encoders_train,
#     decoder=joint_decoder_train,
#     channel=channel_train,
#     power_constraint=power_constraint_train
# )
#
# # Set up optimizer
# optimizer = torch.optim.Adam(mac_model_train.parameters(), lr=0.001)
#
# # Define the fixed SNR for training
# fixed_training_snr = 15.0 # Example fixed SNR for training
#
# # Train the model using the fixed SNR
# training_losses = train_mac_model(
#     mac_model_train,
#     optimizer,
#     num_epochs=25, # Reduced epochs for quick example
#     steps_per_epoch=50,
#     batch_size=batch_size,
#     message_dim=message_dim,
#     num_users=num_users,
#     training_snr=fixed_training_snr # Pass the fixed SNR
# )
#
# # Plot training progress
# plt.figure(figsize=(10, 6))
# plt.plot(training_losses)
# plt.xlabel('Epoch')
# plt.ylabel('Average Training Loss')
# plt.title(f'Training Loss for MAC Model (Fixed SNR = {fixed_training_snr} dB)') # Update title
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show() # Ensure plot is displayed when this section is run


# %%
# Conclusion
# --------------------------------------------------------------------------
# This example demonstrated how to set up and use the MultipleAccessChannelModel.
# Key functionalities include:
# 1. Simulating transmission from multiple users over a shared channel using individual encoders
#    and a joint decoder at a fixed SNR.
# 2. Evaluating reconstruction performance using MSE.
# 3. (Optional) Analyzing the impact of an increasing number of users (interference) on performance
#    at a fixed SNR.
# 4. (Optional) Training the end-to-end system (encoders and decoder) jointly at a fixed SNR.
#
# The commented-out sections provide code for exploring user interference and training.
# You can uncomment them to run those experiments.

# Display any plots generated if running interactively (e.g., in Jupyter)
# This handles the case where the optional sections are not run but the initial setup might create figures.
if plt.get_fignums():
    plt.show()
