#!/usr/bin/env python

"""Basic Usage of Kaira
=======================
Basic usage example of Kaira for image transmission over a noisy channel. This example demonstrates how to create a simple JSCC system for image transmission.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import kaira
from kaira.channels import AWGNChannel
from kaira.metrics import PSNR
from kaira.models.image import DeepJSCCQ2Decoder, DeepJSCCQ2Encoder
from kaira.pipelines import DeepJSCCPipeline
from kaira.utils import snr_db_to_linear, to_tensor


# Load and preprocess an image
def load_image(path, size=(256, 256)):
    """Load and preprocess an image for transmission."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )
    return transform(img).unsqueeze(0)  # Add batch dimension


# Create JSCC model components
encoder = DeepJSCCQ2Encoder(
    in_channels=3,  # RGB image
    latent_dim=16,  # Compressed representation dimension
    compression_ratio=1 / 6,  # Compression factor
)

decoder = DeepJSCCQ2Decoder(out_channels=3, latent_dim=16)  # RGB image

# Create a channel with 10dB SNR
channel = AWGNChannel(snr=10.0)

# Create pipeline
pipeline = DeepJSCCPipeline(encoder=encoder, decoder=decoder, channel=channel)


# Example usage
def main():
    """Run JSCC system for image transmission."""
    # Create a sample image for demonstration
    # This will serve as both the example image and provide the thumbnail
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create RGB components for a visually distinct image
    r = np.sin(5 * np.pi * x_grid) * np.cos(5 * np.pi * y_grid)
    g = np.sin(7 * np.pi * x_grid) * np.cos(7 * np.pi * y_grid)
    b = np.sin(9 * np.pi * x_grid) * np.cos(9 * np.pi * y_grid)
    
    # Normalize to [0, 1] range
    r = (r + 1) / 2
    g = (g + 1) / 2
    b = (b + 1) / 2
    
    # Create RGB image
    rgb_image = np.stack([r, g, b], axis=2)
    
    # Save the sample image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title("Sample Image for Transmission")
    plt.tight_layout()
    plt.savefig("sample_image.jpg")
    
    # Create example thumbnail
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb_image)
    plt.title("Kaira JSCC Example")
    plt.axis('off')
    plt.tight_layout()
    
    # Convert to tensor for processing
    image = torch.tensor(rgb_image).permute(2, 0, 1).unsqueeze(0).float()

    # Transmit through noisy channel
    with torch.no_grad():
        reconstructed = pipeline(image)

    # Calculate PSNR
    psnr_metric = PSNR()
    quality = psnr_metric(reconstructed, image)

    print(f"PSNR: {quality:.2f} dB")

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed.squeeze(0).permute(1, 2, 0).numpy())
    axes[1].set_title(f"Reconstructed (PSNR: {quality:.2f} dB)")
    axes[1].axis('off')
    
    plt.suptitle("Image Transmission over a Noisy Channel")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
