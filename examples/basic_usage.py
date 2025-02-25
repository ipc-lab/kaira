#!/usr/bin/env python

"""Basic usage example of Kaira for image transmission over a noisy channel.

This example demonstrates how to create a simple JSCC system for image transmission.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import kaira
from kaira.channels import GaussianChannel
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
channel = GaussianChannel(snr=10.0)

# Create pipeline
pipeline = DeepJSCCPipeline(encoder=encoder, decoder=decoder, channel=channel)


# Example usage
def main():
    """The following code snippet demonstrates how to use Kaira to create a simple JSCC system for
    image transmission.

    The system consists of an image encoder, a channel, and an image decoder. The encoder
    compresses the input image into a latent representation, which is then transmitted over a noisy
    channel. The decoder reconstructs the image from the received latent representation.
    """

    # Load image
    image = load_image("sample_image.jpg")

    # Transmit through noisy channel
    with torch.no_grad():
        reconstructed = pipeline(image)

    # Calculate PSNR
    psnr_metric = PSNR()
    quality = psnr_metric(reconstructed, image)

    print(f"PSNR: {quality:.2f} dB")

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.squeeze(0).permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed.squeeze(0).permute(1, 2, 0))
    axes[1].set_title(f"Reconstructed (PSNR: {quality:.2f} dB)")
    plt.savefig("reconstruction_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
