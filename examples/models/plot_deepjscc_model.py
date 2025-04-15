"""
=================================================================================================
Deep Joint Source-Channel Coding (DeepJSCC) Model
=================================================================================================

This example demonstrates how to use the DeepJSCC model for image transmission
over a noisy channel. DeepJSCC is an end-to-end approach that jointly optimizes
source compression and channel coding using deep neural networks, providing
robust performance in varying channel conditions.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# -------------------------------
# First, we import necessary modules and set random seeds for reproducibility.
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.constraints.power import AveragePowerConstraint
from kaira.models import DeepJSCCModel
from kaira.models.components import ConvDecoder, ConvEncoder

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Creating Synthetic Data
# ------------------------------------------
# For this example, we'll create a synthetic image dataset.

# Create sample image data (3 channels, 32x32 resolution)
batch_size = 4
image_size = 32
n_channels = 3
x = torch.randn(batch_size, n_channels, image_size, image_size)

# Normalize images to [0, 1] range for better visualization
x = (x - x.min()) / (x.max() - x.min())

# %%
# Visualizing Sample Images
# --------------------------------------------
# Let's visualize one of our sample images.

plt.figure(figsize=(4, 4))
plt.imshow(x[0].permute(1, 2, 0).numpy())
plt.title("Sample Original Image")
plt.axis("off")
plt.tight_layout()

# %%
# Building the DeepJSCC Model
# ---------------------------------------------------
# Now we'll create the components needed for our DeepJSCC model.

# Define model parameters
feature_dim = 256
compression_ratio = 1 / 6  # Channel bandwidth / Source bandwidth
code_length = int(image_size * image_size * n_channels * compression_ratio)

# Create encoder, decoder and other components
encoder = ConvEncoder(in_channels=n_channels, out_features=code_length, hidden_dims=[16, 32, 64])

decoder = ConvDecoder(in_features=code_length, out_channels=n_channels, output_size=(image_size, image_size), hidden_dims=[64, 32, 16])

constraint = AveragePowerConstraint(average_power=1.0)
channel = AWGNChannel(snr_db=10.0)

# Build the DeepJSCC model
model = DeepJSCCModel(encoder=encoder, constraint=constraint, channel=channel, decoder=decoder)

# %%
# Simulating Transmission
# ------------------------------------------
# We'll simulate transmission over channels with different noise levels (SNRs).

snr_values = [0, 5, 10, 15, 20]  # SNR in dB
results = []

# We'll use the first image from our batch for visualization
test_image = x[0:1]

for snr in snr_values:
    # Pass the image through our model with the current SNR
    with torch.no_grad():
        received = model(test_image, snr=snr)

    # Store the result
    results.append(received[0].detach().cpu())

# %%
# Visualizing Results
# ---------------------------------
# Let's visualize the original image and the received images at different SNRs.

plt.figure(figsize=(12, 3))

# Original image
plt.subplot(1, len(snr_values) + 1, 1)
plt.imshow(test_image[0].permute(1, 2, 0).numpy())
plt.title("Original")
plt.axis("off")

# Received images at different SNRs
for i, (snr, result) in enumerate(zip(snr_values, results)):
    plt.subplot(1, len(snr_values) + 1, i + 2)
    plt.imshow(result.permute(1, 2, 0).numpy().clip(0, 1))
    plt.title(f"SNR = {snr} dB")
    plt.axis("off")

plt.tight_layout()

# %%
# Training a DeepJSCC Model
# --------------------------------------------
# In practice, you would train your DeepJSCC model using a loss function.
# Here's how you could set up the training loop:


def train_deepjscc_model(model, train_loader, optimizer, criterion, epochs=5, snr_range=(0, 20)):
    """Example training loop for a DeepJSCC model."""
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, images in enumerate(train_loader):
            # Generate random SNR within the given range
            snr = torch.FloatTensor(1).uniform_(snr_range[0], snr_range[1])

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, snr=snr)

            # Compute loss
            loss = criterion(outputs, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return losses


# Example of how you would use the training function
# (not executed in this example for simplicity)
# # Set up data loader, optimizer, etc.
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# from kaira.losses.image import MSELoss
# criterion = MSELoss()
#
# # Train the model
# training_losses = train_deepjscc_model(model, train_loader, optimizer, criterion)
#
# # Plot training loss
# plt.figure(figsize=(10, 6))
# plt.plot(training_losses)
# plt.xlabel("Training Epoch")
# plt.ylabel("MSE Loss")
# plt.title("DeepJSCC Model Training Loss")
# plt.grid(True)
# plt.show()

# %%
# Conclusion
# --------------------
# This example demonstrated how to set up and use a DeepJSCC model for joint source-channel
# coding in image transmission. The model effectively handles different channel qualities
# and provides graceful degradation as the SNR decreases.
#
# For practical applications, you would:
# 1. Use real image datasets
# 2. Train the model for longer with proper hyperparameter tuning
# 3. Evaluate the model using appropriate metrics like PSNR or SSIM
# 4. Compare with traditional separate source and channel coding approaches
