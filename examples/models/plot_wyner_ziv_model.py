"""
=================================================================================================
Wyner-Ziv Model for Source Coding with Side Information
=================================================================================================

This example demonstrates how to use the WynerZivModel for source coding with side information,
a scenario where the decoder has access to additional correlated information (side information)
that is not available to the encoder, allowing for more efficient compression.
"""

# %%
# Imports and Setup
# -------------------------------
# First, we import necessary modules and set random seeds for reproducibility.
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from kaira.models import WynerZivModel
from kaira.channels import AWGNChannel
from kaira.constraints.power import AveragePowerConstraint, TotalPowerConstraint
from kaira.models.components import MLPEncoder, MLPDecoder

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Creating Source and Side Information
# ----------------------------------------------------------------
# Let's generate correlated source and side information.

# Define parameters
batch_size = 64
source_dim = 10
code_dim = 5  # Compressed representation dimension

# Create correlated source and side information
# We'll model correlation using a simple Gaussian model: y = x + noise
source_data = torch.randn(batch_size, source_dim)
correlation_noise = 0.5 * torch.randn(batch_size, source_dim)  # Noise level determines correlation
side_info = source_data + correlation_noise

print(f"Source data shape: {source_data.shape}")
print(f"Side information shape: {side_info.shape}")

# Visualize correlation between source and side information
plt.figure(figsize=(6, 6))
plt.scatter(source_data[:, 0].numpy(), side_info[:, 0].numpy(), alpha=0.5)
plt.xlabel("Source (First Dimension)")
plt.ylabel("Side Information (First Dimension)")
plt.title("Correlation between Source and Side Information")
plt.grid(True)

# %%
# Building the Wyner-Ziv Model
# ----------------------------------------------------
# Now we'll create the Wyner-Ziv model components.

# Source encoder (compresses the source without knowledge of side information)
encoder = MLPEncoder(
    in_features=source_dim,
    out_features=code_dim,
    hidden_dims=[20, 15]
)

# Power constraint for the encoded signal
constraint = TotalPowerConstraint(total_power=1.0)

# Channel for transmitting the compressed representation
channel = AWGNChannel(snr_db=10.0)

# Decoder with access to both the received code and side information
# Input features = code_dim (from received code) + source_dim (from side info)
decoder = MLPDecoder(
    in_features=code_dim + source_dim,
    out_features=source_dim,
    hidden_dims=[24, 20]
)

# Build the Wyner-Ziv model
wz_model = WynerZivModel(
    encoder=encoder,
    constraint=constraint,
    channel=channel,
    decoder=decoder
)

# %%
# Simulating Transmission with Side Information
# ----------------------------------------------------------------------------------
# We'll simulate transmission at different SNR levels and observe reconstruction quality.

snr_values = [0, 5, 10, 15, 20]
mse_per_snr = []

# We'll use MSE to evaluate reconstruction quality
mse_loss = nn.MSELoss()

for snr in snr_values:
    # Pass the source data through our model with the current SNR and side information
    with torch.no_grad():
        reconstructed = wz_model(source_data, side_info)
    
    # Calculate MSE
    error = mse_loss(reconstructed, source_data).item()
    mse_per_snr.append(error)
    print(f"SNR: {snr} dB, MSE: {error:.6f}")

# %%
# Visualizing Reconstruction Quality vs SNR
# -------------------------------------------------------------------------
# Let's plot the reconstruction quality as a function of SNR.

plt.figure(figsize=(10, 6))
plt.plot(snr_values, mse_per_snr, 'o-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Reconstruction Error vs. SNR for Wyner-Ziv Coding')
plt.yscale('log')
plt.tight_layout()

# %%
# Examining Effect of Side Information Quality
# ---------------------------------------------------------------------------------
# Let's examine how the quality of side information affects reconstruction.

def simulate_with_varying_correlation(model, source_data, noise_levels, snr=10.0):
    """Simulate transmission with varying correlation between source and side info."""
    results = []
    
    for noise in noise_levels:
        # Generate side information with different correlation levels
        side_info = source_data + noise * torch.randn_like(source_data)
        
        # Transmit through the model
        with torch.no_grad():
            reconstructed = model(source_data, side_info)
        
        # Calculate error
        error = mse_loss(reconstructed, source_data).item()
        results.append(error)
    
    return results

# Define correlation levels by varying noise
noise_levels = [0.1, 0.5, 1.0, 2.0, 4.0]
fixed_snr = 10.0

# Simulate with different correlation levels
correlation_results = simulate_with_varying_correlation(
    wz_model, source_data, noise_levels, fixed_snr)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, correlation_results, 'o-', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Side Information Noise Level')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'Effect of Side Information Quality (SNR={fixed_snr}dB)')
plt.yscale('log')

# %%
# Comparison with Direct Transmission
# ---------------------------------------------------------------
# Let's compare the Wyner-Ziv scheme with direct transmission without side information.

# For this comparison, we'll create a simplified model without side info
# It will have the same structure but won't use side information

# Simple encoder-decoder without side info
standard_encoder = MLPEncoder(
    in_features=source_dim,
    out_features=code_dim,
    hidden_dims=[20, 15]
)

standard_constraint = AveragePowerConstraint(average_power=1.0)
standard_channel = AWGNChannel(snr_db=10.0)

# Decoder without side information
standard_decoder = MLPDecoder(
    in_features=code_dim,
    out_features=source_dim,
    hidden_dims=[15, 20]
)

# Function for direct transmission without side info
def direct_transmission(source, encoder, constraint, channel, decoder, snr):
    code = encoder(source)
    constrained_code = constraint(code)
    received = channel(constrained_code)
    reconstructed = decoder(received)
    return reconstructed

# Compare performance at different SNRs
wz_errors = []
direct_errors = []

for snr in snr_values:
    # Wyner-Ziv with side info
    with torch.no_grad():
        wz_reconstructed = wz_model(source_data, side_info)
        wz_error = mse_loss(wz_reconstructed, source_data).item()
        wz_errors.append(wz_error)
    
    # Direct transmission without side info
    with torch.no_grad():
        direct_reconstructed = direct_transmission(
            source_data, standard_encoder, standard_constraint, 
            standard_channel, standard_decoder, snr)
        direct_error = mse_loss(direct_reconstructed, source_data).item()
        direct_errors.append(direct_error)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(snr_values, wz_errors, 'o-', linewidth=2, label='Wyner-Ziv with Side Info')
plt.plot(snr_values, direct_errors, 's--', linewidth=2, label='Direct Transmission')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Performance Comparison: Wyner-Ziv vs Direct Transmission')
plt.yscale('log')
plt.legend()
plt.tight_layout()

# %%
# Training a Wyner-Ziv Model
# --------------------------------------------
# In practice, you would train your Wyner-Ziv model to minimize reconstruction error.
# Here's how you could set up the training loop:

def train_wyner_ziv_model(model, optimizer, num_epochs=50, batch_size=64, 
                       source_dim=10, noise_range=(0.1, 1.0),
                       snr_range=(0, 20)):
    """Example training loop for a Wyner-Ziv model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        # Generate random source data
        source_data = torch.randn(batch_size, source_dim)
        
        # Generate random noise level for side information
        noise_level = torch.FloatTensor(1).uniform_(noise_range[0], noise_range[1]).item()
        side_info = source_data + noise_level * torch.randn_like(source_data)
        
        # Generate random SNR within the given range
        snr = torch.FloatTensor(1).uniform_(snr_range[0], snr_range[1]).item()
        
        # Set the channel's SNR for this iteration
        model.channel.snr_db = snr
        
        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(source_data, side_info)
        
        # Compute loss
        loss = mse_loss(reconstructed, source_data)
        
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
optimizer = torch.optim.Adam(wz_model.parameters(), lr=0.001)

# Train the model
training_losses = train_wyner_ziv_model(wz_model, optimizer)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss for Wyner-Ziv Model')
plt.grid(True)
"""

# %%
# Conclusion
# --------------------
# This example demonstrated how to use the WynerZivModel for source coding with side information.
# Key insights include:
#
# 1. Side information at the decoder can improve reconstruction quality
# 2. The quality of side information significantly impacts performance
# 3. Wyner-Ziv coding outperforms direct transmission when good side information is available
#
# In practical applications, Wyner-Ziv coding is valuable for scenarios like:
# - Distributed video coding where key frames are used as side information
# - Sensor networks where correlated measurements are available
# - Multi-view image coding where different camera perspectives are correlated
