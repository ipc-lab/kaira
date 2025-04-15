"""
=================================================================================================
Sequential Model for Modular Neural Network Design
=================================================================================================

This example demonstrates how to use the SequentialModel as a foundation for building
modular neural network architectures. The SequentialModel allows you to compose multiple
modules together, similar to PyTorch's nn.Sequential but with additional features for
communication system modeling.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# -------------------------------
# First, we import necessary modules and set random seeds for reproducibility.
import numpy as np
import torch
from torch import nn

from kaira.channels import AWGNChannel
from kaira.constraints.power import TotalPowerConstraint
from kaira.models.components import MLPDecoder, MLPEncoder
from kaira.models.generic import SequentialModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Creating Test Data
# -------------------------------
# Let's create some test data for our demonstration.

# Define parameters
batch_size = 32
input_dim = 10
hidden_dim = 16
output_dim = 10

# Create random input data
x = torch.rand(batch_size, input_dim)

print(f"Input data shape: {x.shape}")
print(f"Example input: {x[0][:5]}")  # Print first 5 elements

# %%
# Building a Basic Sequential Model
# ------------------------------------------------------------
# Let's start with a simple sequential model that chains multiple operations.

# Create individual components
encoder = MLPEncoder(in_features=input_dim, out_features=hidden_dim, hidden_dims=[20])

constraint = TotalPowerConstraint(total_power=1.0)
channel = AWGNChannel(snr_db=10.0)  # Using 10 dB SNR

decoder = MLPDecoder(in_features=hidden_dim, out_features=output_dim, hidden_dims=[20])

# Create a sequential model
sequential_model = SequentialModel(steps=[encoder, constraint, channel, decoder])

# Check model structure
print("Sequential model structure:")
for i, step in enumerate(sequential_model.steps):
    print(f"  {i}: {step.__class__.__name__}")

# %%
# Using the Sequential Model
# -------------------------------------------
# Let's see how to use this model with different parameters.

# Passing data through the model at different SNR levels
snr_values = [0, 5, 10, 15, 20]
mse_per_snr = []

mse_loss = nn.MSELoss()

for snr in snr_values:
    # Pass the data through our model with the current SNR
    with torch.no_grad():
        output = sequential_model(x, snr=snr)

    # Calculate MSE
    error = mse_loss(output, x).item()
    mse_per_snr.append(error)
    print(f"SNR: {snr} dB, MSE: {error:.6f}")

# %%
# Visualizing Performance
# ----------------------------------------
# Let's plot the reconstruction error as a function of SNR.

plt.figure(figsize=(10, 6))
plt.plot(snr_values, mse_per_snr, "o-", linewidth=2)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Reconstruction Error vs. SNR")
plt.yscale("log")
plt.tight_layout()

# %%
# Advanced Sequential Model with Module Replacement
# ------------------------------------------------------------------------------------------
# One advantage of SequentialModel is that we can easily swap components.
# Let's create a new model with a different encoder.

# Create a new encoder with different architecture
new_encoder = MLPEncoder(in_features=input_dim, out_features=hidden_dim, hidden_dims=[32, 24])  # Deeper network

# Create a new sequential model with the updated encoder
enhanced_model = SequentialModel(steps=[new_encoder, constraint, channel, decoder])

# Compare performance of original and enhanced models
original_errors = []
enhanced_errors = []

for snr in snr_values:
    # Original model
    with torch.no_grad():
        original_output = sequential_model(x, snr=snr)
        original_error = mse_loss(original_output, x).item()
        original_errors.append(original_error)

    # Enhanced model
    with torch.no_grad():
        enhanced_output = enhanced_model(x, snr=snr)
        enhanced_error = mse_loss(enhanced_output, x).item()
        enhanced_errors.append(enhanced_error)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(snr_values, original_errors, "o-", linewidth=2, label="Original Model")
plt.plot(snr_values, enhanced_errors, "s--", linewidth=2, label="Enhanced Model")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Performance Comparison: Original vs Enhanced Model")
plt.yscale("log")
plt.legend()
plt.tight_layout()

# %%
# Inspecting Intermediate Outputs
# -----------------------------------------------------
# The sequential model also allows us to inspect intermediate outputs.

# First, let's define a custom snr value for analysis
analysis_snr = 10.0

# Initialize inputs, will be updated at each stage
current_output = x

# Process and visualize each stage
intermediate_outputs = []
output_stats = []

for i, module in enumerate(sequential_model.steps):
    # Process current module
    # Note: AWGNChannel doesn't accept snr in its forward method
    # The SNR must be set during channel initialization
    current_output = module(current_output)

    # Store intermediate output (use first sample for visualization)
    intermediate_outputs.append(current_output[0].detach().cpu().numpy())

    # Calculate statistics
    output_stats.append({"module": module.__class__.__name__, "mean": current_output.mean().item(), "std": current_output.std().item(), "min": current_output.min().item(), "max": current_output.max().item()})

# Create a grid to visualize intermediate outputs
fig, axes = plt.subplots(len(sequential_model.steps), 1, figsize=(12, 10))
module_names = [module.__class__.__name__ for module in sequential_model.steps]

for i, (ax, output, name) in enumerate(zip(axes, intermediate_outputs, module_names)):
    ax.stem(np.arange(len(output)), output)
    ax.set_title(f"After {name}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Print statistics
print(f"\nIntermediate output statistics at SNR = {analysis_snr} dB:")
for stats in output_stats:
    print(f"{stats['module']}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, " f"min={stats['min']:.3f}, max={stats['max']:.3f}")

# %%
# Creating a Custom Process Flow
# ----------------------------------------------------
# The SequentialModel can be extended to create custom processing flows.
# Here's an example of creating a residual connection in a sequential model.


class ResidualSequentialModel(SequentialModel):
    """Example of a custom sequential model with a residual connection."""

    def forward(self, x, **kwargs):
        # Save input for residual connection
        original_input = x

        # Process through modules
        current = x
        for i, module in enumerate(self.steps):
            if hasattr(module, "supports_snr") and module.supports_snr and "snr" in kwargs:
                current = module(current, snr=kwargs["snr"])
            else:
                current = module(current)

        # Add residual connection (assuming input and output dimensions match)
        if original_input.shape == current.shape:
            return current + 0.1 * original_input
        else:
            return current


# Create residual model with compatible dimensions
residual_model = ResidualSequentialModel(steps=[encoder, constraint, channel, decoder])

# Compare standard and residual models
standard_errors = []
residual_errors = []

for snr in snr_values:
    # Standard sequential model
    with torch.no_grad():
        # Update channel SNR in standard model
        for module in sequential_model.steps:
            if isinstance(module, AWGNChannel):
                module.snr_db = snr

        standard_output = sequential_model(x)
        standard_error = mse_loss(standard_output, x).item()
        standard_errors.append(standard_error)

    # Residual model
    with torch.no_grad():
        # Update channel SNR in residual model
        for module in residual_model.steps:
            if isinstance(module, AWGNChannel):
                module.snr_db = snr

        residual_output = residual_model(x)
        residual_error = mse_loss(residual_output, x).item()
        residual_errors.append(residual_error)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(snr_values, standard_errors, "o-", linewidth=2, label="Standard Sequential")
plt.plot(snr_values, residual_errors, "s--", linewidth=2, label="Residual Sequential")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Performance Comparison: Standard vs Residual Sequential Model")
plt.yscale("log")
plt.legend()
plt.tight_layout()

# %%
# Training Sequential Models
# -------------------------------------------
# Here's how you can train a sequential model:


def train_sequential_model(model, optimizer, num_epochs=50, batch_size=32, input_dim=10, snr_range=(0, 20)):
    """Example training loop for a sequential model."""
    model.train()
    losses = []

    for epoch in range(num_epochs):
        # Generate random input data
        inputs = torch.rand(batch_size, input_dim)

        # Generate random SNR within the given range
        snr = torch.FloatTensor(1).uniform_(snr_range[0], snr_range[1]).item()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, snr=snr)

        # Compute loss (reconstruction error)
        loss = nn.MSELoss()(outputs, inputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        losses.append(loss.item())

    return losses


# Example of how you would train the model
# (not executed in this example for simplicity)
# # Set up optimizer
# optimizer = torch.optim.Adam(sequential_model.parameters(), lr=0.001)
#
# # Train the model
# training_losses = train_sequential_model(sequential_model, optimizer)
#
# # Plot training loss
# plt.figure(figsize=(10, 6))
# plt.plot(training_losses)
# plt.xlabel("Training Epoch")
# plt.ylabel("MSE Loss")
# plt.title("Sequential Model Training Loss")
# plt.grid(True)
# plt.show()

# %%
# Conclusion
# --------------------
# This example demonstrated how to use the SequentialModel to build modular
# neural network architectures for communication systems. Key insights include:
#
# 1. Sequential models provide a clean way to chain operations in a communication pipeline
# 2. Components can be easily replaced for experimentation and improvement
# 3. Intermediate outputs can be inspected for analysis
# 4. The basic sequential structure can be extended for custom processing flows
#
# Sequential models are ideal for communication systems where data flows through
# distinct processing stages and adaptability to different components is important.
