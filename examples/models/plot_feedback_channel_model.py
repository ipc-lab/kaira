"""
=================================================================================================
Feedback Channel Model for Interactive Communication
=================================================================================================

This example demonstrates how to use the FeedbackChannelModel for communication
systems where there's a feedback link from the receiver to the transmitter. Such
systems can adapt their transmission strategy based on the feedback, potentially
improving performance compared to one-way communication.
"""

# %%
# Imports and Setup
# -------------------------------
# First, we import necessary modules and set random seeds for reproducibility.
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from kaira.models import FeedbackChannelModel
from kaira.channels import AWGNChannel
from kaira.constraints.power import AveragePowerConstraint, TotalPowerConstraint
from kaira.models.components import MLPEncoder, MLPDecoder

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Creating Test Data
# -------------------------------
# We'll create some test data to transmit over our feedback channel system.

# Define parameters
batch_size = 32
message_dim = 10
code_dim = 8
feedback_dim = 4  # Dimension of the feedback information

# Create random messages to transmit
messages = torch.rand(batch_size, message_dim)

print(f"Message shape: {messages.shape}")
print(f"Example message: {messages[0][:5]}")  # Print first 5 elements

# %%
# Building the Feedback Channel Model
# ---------------------------------------------------------------
# We'll create a feedback channel model with encoders, decoders, and feedback mechanisms.

# Forward encoder (transmitter)
forward_encoder = MLPEncoder(
    in_features=message_dim,  # Change to only take message as input, matching what's provided
    out_features=code_dim,
    hidden_dims=[20, 15]
)

# Forward constraint
forward_constraint = TotalPowerConstraint(total_power=1.0)

# Forward channel
forward_channel = AWGNChannel(snr_db=10.0)  # Using 10 dB SNR

# Decoder (receiver)
decoder = MLPDecoder(
    in_features=code_dim,
    out_features=message_dim,
    hidden_dims=[15, 20]
)

# Feedback encoder (at the receiver)
feedback_encoder = MLPEncoder(
    in_features=code_dim,  # Takes received signal as input
    out_features=feedback_dim,
    hidden_dims=[12, 8]
)

# Feedback constraint
feedback_constraint = TotalPowerConstraint(total_power=0.5)  # Lower power for feedback channel

# Feedback channel
feedback_channel = AWGNChannel(snr_db=10.0)

# Build the feedback channel model
feedback_model = FeedbackChannelModel(
    encoder=forward_encoder,
    forward_channel=forward_channel,
    decoder=decoder,
    feedback_generator=feedback_encoder,
    feedback_channel=feedback_channel,
    feedback_processor=feedback_encoder,  # Using feedback_encoder as the processor for simplicity
    max_iterations=3  # Adding explicit iteration count
)

# %%
# Simulating Interactive Communication
# ---------------------------------------------------------------
# We'll simulate transmission at different forward and feedback SNR levels.

# Define SNR levels for both forward and feedback channels
forward_snr_values = [0, 5, 10, 15, 20]
feedback_snr = 10  # Fixed feedback SNR for this analysis

mse_per_snr = []
mse_loss = nn.MSELoss()

for forward_snr in forward_snr_values:
    # Pass the messages through our model with the current SNRs
    with torch.no_grad():
        reconstructed = feedback_model(messages, forward_snr=forward_snr, feedback_snr=feedback_snr)
    
    # Calculate MSE
    error = mse_loss(reconstructed, messages).item()
    mse_per_snr.append(error)
    print(f"Forward SNR: {forward_snr} dB, Feedback SNR: {feedback_snr} dB, MSE: {error:.6f}")

# %%
# Visualizing Reconstruction Quality with Feedback
# ------------------------------------------------------------------------------------
# Let's plot the reconstruction quality as a function of forward SNR.

plt.figure(figsize=(10, 6))
plt.plot(forward_snr_values, mse_per_snr, 'o-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Forward Channel SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'Reconstruction Error vs. Forward SNR (Feedback SNR = {feedback_snr}dB)')
plt.yscale('log')
plt.tight_layout()

# %%
# Effect of Feedback Channel Quality
# -------------------------------------------------------------
# Let's examine how the quality of the feedback channel affects overall performance.

# Fix forward SNR and vary feedback SNR
forward_snr = 10  # Fixed forward SNR for this analysis
feedback_snr_values = [0, 5, 10, 15, 20]

mse_with_varying_feedback = []

for feedback_snr in feedback_snr_values:
    # Pass the messages through our model with the current SNRs
    with torch.no_grad():
        reconstructed = feedback_model(messages, forward_snr=forward_snr, feedback_snr=feedback_snr)
    
    # Calculate MSE
    error = mse_loss(reconstructed, messages).item()
    mse_with_varying_feedback.append(error)
    print(f"Forward SNR: {forward_snr} dB, Feedback SNR: {feedback_snr} dB, MSE: {error:.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(feedback_snr_values, mse_with_varying_feedback, 'o-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Feedback Channel SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'Effect of Feedback Channel Quality (Forward SNR = {forward_snr}dB)')
plt.yscale('log')
plt.tight_layout()

# %%
# Comparison With and Without Feedback
# ---------------------------------------------------------------
# Let's compare our feedback model with a similar system without feedback.

# For this comparison, we'll create a simplified model without feedback
# It will have the same structure but without the feedback mechanism

# Simple encoder without feedback input
standard_encoder = MLPEncoder(
    in_features=message_dim,
    out_features=code_dim,
    hidden_dims=[20, 15]
)

standard_constraint = AveragePowerConstraint(average_power=1.0)
standard_channel = AWGNChannel(snr_db=10.0)
standard_decoder = decoder  # Use the same decoder for fair comparison

# Function for direct transmission without feedback
def direct_transmission(messages, encoder, constraint, channel, decoder, snr):
    code = encoder(messages)
    constrained_code = constraint(code)
    received = channel(constrained_code, snr=snr)
    reconstructed = decoder(received)
    return reconstructed

# Compare performance at different SNRs
feedback_errors = []
direct_errors = []

# Use the first set of SNRs for comparison
for forward_snr in forward_snr_values:
    # With feedback
    with torch.no_grad():
        feedback_reconstructed = feedback_model(
            messages, forward_snr=forward_snr, feedback_snr=feedback_snr)
        feedback_error = mse_loss(feedback_reconstructed, messages).item()
        feedback_errors.append(feedback_error)
    
    # Without feedback
    with torch.no_grad():
        direct_reconstructed = direct_transmission(
            messages, standard_encoder, standard_constraint, 
            standard_channel, standard_decoder, forward_snr)
        direct_error = mse_loss(direct_reconstructed, messages).item()
        direct_errors.append(direct_error)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(forward_snr_values, feedback_errors, 'o-', linewidth=2, label='With Feedback')
plt.plot(forward_snr_values, direct_errors, 's--', linewidth=2, label='Without Feedback')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Forward SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Performance Comparison: With vs. Without Feedback')
plt.yscale('log')
plt.legend()
plt.tight_layout()

# %%
# Visualizing Feedback Information
# ------------------------------------------------------
# Let's examine what kind of information is being sent back through the feedback channel.

# Function to get feedback signal for different messages
def get_feedback_signal(model, messages, forward_snr, feedback_snr):
    # First encode and send the message
    code = model.forward_encoder(torch.cat([messages, torch.zeros(batch_size, feedback_dim)], dim=1))
    constrained_code = model.forward_constraint(code)
    received = model.forward_channel(constrained_code, snr=forward_snr)
    
    # Generate feedback
    feedback = model.feedback_encoder(received)
    
    return feedback

# Get feedback for some example messages
example_feedback = get_feedback_signal(
    feedback_model, messages[:5], forward_snr=10, feedback_snr=10)

# Visualize the first dimension of feedback for the first few messages
plt.figure(figsize=(10, 6))
for i in range(min(5, feedback_dim)):
    plt.plot(example_feedback[:, i].detach().numpy(), 'o-', label=f'Dimension {i+1}')
plt.grid(True, alpha=0.3)
plt.xlabel('Message Index')
plt.ylabel('Feedback Signal Value')
plt.title('Feedback Signal Patterns')
plt.legend()
plt.tight_layout()

# %%
# Training a Feedback Channel Model
# ------------------------------------------------------------
# In practice, you would train your feedback model to minimize reconstruction error.
# Here's an example of how you could set up the training process:

def train_feedback_model(model, optimizer, num_epochs=50, batch_size=32,
                       message_dim=10, forward_snr_range=(0, 20), feedback_snr_range=(5, 15)):
    """Example training loop for a feedback channel model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        # Generate random messages
        messages = torch.rand(batch_size, message_dim)
        
        # Generate random SNRs within the given ranges
        forward_snr = torch.FloatTensor(1).uniform_(forward_snr_range[0], forward_snr_range[1]).item()
        feedback_snr = torch.FloatTensor(1).uniform_(feedback_snr_range[0], feedback_snr_range[1]).item()
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(messages, forward_snr=forward_snr, feedback_snr=feedback_snr)
        
        # Compute loss
        loss = mse_loss(reconstructed, messages)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        losses.append(loss.item())
    
    return losses

# Example of how you would train the model
# (not executed in this example for simplicity)
"""
# Set up optimizer
optimizer = torch.optim.Adam(feedback_model.parameters(), lr=0.001)

# Train the model
training_losses = train_feedback_model(feedback_model, optimizer)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Feedback Channel Model')
plt.grid(True)
"""

# %%
# Conclusion
# --------------------
# This example demonstrated how to use the FeedbackChannelModel for interactive communication
# where a feedback channel exists from the receiver back to the transmitter. Key insights include:
#
# 1. Feedback can improve the performance of communication systems by allowing adaptation
# 2. The quality of the feedback channel significantly impacts the overall performance
# 3. Systems with feedback can outperform standard one-way communication systems
#
# In practical applications, feedback communication is valuable for scenarios like:
# - ARQ (Automatic Repeat reQuest) protocols where the receiver requests retransmissions
# - Rate adaptation in wireless systems based on channel state information
# - Power control and beam steering in MIMO systems
# - Adaptive coding and modulation schemes
