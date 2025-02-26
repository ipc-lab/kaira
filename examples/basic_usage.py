#!/usr/bin/env python

"""
Basic Usage of Kaira
===================

This example demonstrates how to create a simple JSCC system for image transmission over a noisy channel using Kaira.
"""

import os
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
from kaira.constraints import AveragePowerConstraint


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


def create_sample_image(size=(256, 256)):
    """Create a sample image for demonstration."""
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
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
    
    return rgb_image


# Example usage
def main():
    """Run JSCC system for image transmission."""
    # Create JSCC model components
    encoder = DeepJSCCQ2Encoder(
        N=64,
        M=32
    )

    decoder = DeepJSCCQ2Decoder(M=32, N=64)  # RGB image

    # Create a channel with 10dB SNR
    channel = AWGNChannel(avg_noise_power=1.0)
    
    # Set the average power constraint
    constraint = AveragePowerConstraint(1.0)

    # Create pipeline
    pipeline = DeepJSCCPipeline(encoder=encoder, decoder=decoder, channel=channel, constraint=constraint)
    
    # Create a sample image for demonstration
    rgb_image = create_sample_image()
    
    # Create example thumbnail
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb_image)
    plt.title("Sample Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
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
