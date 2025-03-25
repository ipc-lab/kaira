"""
============================
Image Quality Metrics
============================

This example demonstrates the image quality metrics available in Kaira, including
PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), MS-SSIM
(Multi-Scale SSIM), and LPIPS (Learned Perceptual Image Patch Similarity).

These metrics are particularly useful for:
* Evaluating image compression algorithms
* Assessing deep learning-based image processing
* Quality control in image transmission systems
"""

# %%
# First, let's import the necessary modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from kaira.metrics import (
    PSNR, SSIM, MultiScaleSSIM, LPIPS,
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity
)
from kaira.metrics.registry import MetricRegistry
from kaira.metrics.utils import visualize_metrics_comparison

# %%
# Load test images
# ---------------
# We'll use some images from CIFAR-10 dataset for demonstration

transform = T.Compose([
    T.ToTensor(),
])

dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get a batch of images
images, _ = next(iter(dataloader))

# %%
# Create different types of distortions
# -----------------------------------
# We'll create different types of distortions to compare how various metrics
# assess them

def add_gaussian_noise(image, std=0.1):
    """Add Gaussian noise to image."""
    return image + torch.randn_like(image) * std

def add_salt_pepper_noise(image, prob=0.05):
    """Add salt and pepper noise to image."""
    mask = torch.rand_like(image)
    image = image.clone()
    image[mask < prob/2] = 0  # salt
    image[mask > 1 - prob/2] = 1  # pepper
    return image

def blur_image(image, kernel_size=3):
    """Apply Gaussian blur to image."""
    return T.GaussianBlur(kernel_size)(image)

def compress_image(image, quality=10):
    """Simulate JPEG compression artifacts."""
    to_pil = T.ToPILImage()
    to_tensor = T.ToTensor()
    return to_tensor(to_pil(image).convert('RGB').save('temp.jpg', quality=quality))

# Create distorted versions
noisy_images = torch.stack([add_gaussian_noise(img) for img in images])
sp_noisy_images = torch.stack([add_salt_pepper_noise(img) for img in images])
blurred_images = torch.stack([blur_image(img) for img in images])
compressed_images = torch.stack([compress_image(img) for img in images])

# %%
# Initialize metrics
# ----------------
# We'll use Kaira's metric registry to create our metrics

metrics_dict = MetricRegistry.create_image_quality_metrics(data_range=1.0)

# Or initialize metrics individually:
psnr = PeakSignalNoiseRatio(data_range=1.0)  # or PSNR()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)  # or SSIM()
ms_ssim = MultiScaleSSIM(data_range=1.0)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')  # or LPIPS()

# %%
# Evaluate metrics on different distortions
# ---------------------------------------

def evaluate_all_metrics(original, distorted):
    """Evaluate all metrics between original and distorted images."""
    return {
        'PSNR': psnr(distorted, original),
        'SSIM': ssim(distorted, original),
        'MS-SSIM': ms_ssim(distorted, original),
        'LPIPS': lpips(distorted, original)
    }

# Evaluate metrics for each type of distortion
gaussian_metrics = evaluate_all_metrics(images, noisy_images)
sp_metrics = evaluate_all_metrics(images, sp_noisy_images)
blur_metrics = evaluate_all_metrics(images, blurred_images)
compress_metrics = evaluate_all_metrics(images, compressed_images)

# %%
# Visualize results
# ---------------
# First, let's look at the distorted images

plt.figure(figsize=(15, 8))
titles = ['Original', 'Gaussian Noise', 'Salt & Pepper', 'Blur', 'Compressed']
all_images = [images, noisy_images, sp_noisy_images, blurred_images, compressed_images]

for i, (title, imgs) in enumerate(zip(titles, all_images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(imgs[0].permute(1, 2, 0).clip(0, 1))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

# %%
# Now let's compare how different metrics evaluate each distortion type

all_metrics = [
    gaussian_metrics,
    sp_metrics,
    blur_metrics,
    compress_metrics
]

labels = ['Gaussian', 'Salt & Pepper', 'Blur', 'Compression']

visualize_metrics_comparison(
    all_metrics,
    labels,
    figsize=(12, 6),
    title='Image Quality Metrics Comparison'
)

# %%
# Interpreting the Results
# -----------------------
#
# The results show how different metrics capture various aspects of image quality:
#
# * **PSNR** is a simple pixel-level metric that measures absolute differences
#   * Higher values indicate better quality
#   * More sensitive to noise than blurring
#   * May not align well with human perception
#
# * **SSIM** considers structural information
#   * Values range from -1 to 1 (higher is better)
#   * More tolerant of uniform changes
#   * Better correlation with human perception than PSNR
#
# * **MS-SSIM** evaluates structural similarity at multiple scales
#   * Similar to SSIM but captures both local and global structures
#   * Often preferred for high-resolution images
#   * Better at detecting blur than basic SSIM
#
# * **LPIPS** uses deep features to measure perceptual similarity
#   * Lower values indicate better perceptual quality
#   * Trained on human perceptual judgments
#   * Often best matches human quality assessment
#
# Different distortions affect these metrics differently:
#
# * Gaussian noise heavily impacts PSNR but less so SSIM
# * Blur might maintain good PSNR but show poor SSIM/MS-SSIM
# * LPIPS often identifies perceptually significant distortions
#   that other metrics might miss
#
# For practical applications:
#
# * Use multiple metrics for comprehensive evaluation
# * Consider the specific requirements of your application
# * LPIPS is recommended when perceptual quality is critical
# * PSNR/SSIM are good for optimization objectives due to
#   their mathematical properties
