"""Quick start example for using the kaira.metrics module."""

import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.append(os.path.abspath("../.."))

from kaira.metrics import (
    LPIPS,
    PSNR,
    SSIM,
    compute_multiple_metrics,
    create_image_quality_metrics,
    format_metric_results,
)


def generate_sample_images(batch_size=4, channels=3, height=64, width=64, noise_level=0.1):
    """Generate sample images for demonstration.

    Args:
        batch_size (int): Number of images to generate
        channels (int): Number of channels in images
        height (int): Height of images
        width (int): Width of images
        noise_level (float): Standard deviation of noise to add

    Returns:
        tuple: Original images and noisy images
    """
    # Create random "original" images (normalized to [0, 1])
    originals = torch.rand(batch_size, channels, height, width)

    # Add noise to create "reconstructed" images
    noise = noise_level * torch.randn_like(originals)
    noisy = torch.clamp(originals + noise, 0, 1)

    return originals, noisy


def visualize_images(originals, noisy):
    """Visualize original and noisy images.

    Args:
        originals (torch.Tensor): Original images
        noisy (torch.Tensor): Noisy/reconstructed images
    """
    batch_size = originals.shape[0]
    fig, axes = plt.subplots(2, batch_size, figsize=(3 * batch_size, 6))

    for i in range(batch_size):
        # Plot original image
        img = originals[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        # Plot noisy image
        img = noisy[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Noisy {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Run the quick start example."""
    print("Kaira Metrics Quick Start Example\n")

    # Generate sample images
    print("Generating sample images...")
    originals, noisy = generate_sample_images(batch_size=4, noise_level=0.1)

    # Visualize images
    print("Displaying sample images...")
    visualize_images(originals, noisy)

    # Individual metrics
    print("\nComputing individual metrics:")
    psnr = PSNR(data_range=1.0)
    ssim = SSIM(data_range=1.0)
    lpips = LPIPS(net_type="alex")

    psnr_value = psnr(noisy, originals)
    ssim_value = ssim(noisy, originals)
    lpips_value = lpips(noisy, originals)

    print(f"PSNR: {psnr_value.mean():.4f} ± {psnr_value.std():.4f}")
    print(f"SSIM: {ssim_value.mean():.4f} ± {ssim_value.std():.4f}")
    print(f"LPIPS: {lpips_value.mean():.4f} ± {lpips_value.std():.4f}")

    # Using factory to create metrics
    print("\nUsing factory to create multiple metrics:")
    metrics = create_image_quality_metrics(data_range=1.0, lpips_net_type="alex")

    # Computing multiple metrics at once
    print("\nComputing multiple metrics at once:")
    results = compute_multiple_metrics(metrics, noisy, originals)

    # Format results
    formatted = format_metric_results(results)
    print(f"Results: {formatted}")

    print("\nDone!")


if __name__ == "__main__":
    main()
