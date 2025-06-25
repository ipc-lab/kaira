"""
=================================================================================================
Image Compressors Comparison
=================================================================================================

This example demonstrates how to use all available image compressors in Kaira, including
traditional image compression formats (JPEG, PNG, WebP, etc.) and neural compression models.
We'll compare their performance in terms of compression ratio and image quality.

This example covers:

* Traditional image compressors (JPEG, PNG, WebP, JPEG 2000)
* Advanced compressors (BPG, JPEG XL)
* Neural network-based compressors (optional)
* Performance comparison and visualization
* Quality vs compression trade-off analysis
"""

import warnings
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.data.sample_data import SampleImagesDataset
from kaira.models.image.compressors import (
    BPGCompressor,
    JPEG2000Compressor,
    JPEGCompressor,
    JPEGXLCompressor,
    NeuralCompressor,
    PNGCompressor,
    WebPCompressor,
)

# %%
# Imports and Setup
# -------------------------------


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# %%
# Loading Sample Images
# ---------------------------------
# Load sample images for compression testing
# Using 128x128 size for better PNG compression results

print("Loading sample images...")
dataset = SampleImagesDataset(n_samples=4, target_size=(128, 128))
print(f"Loaded {len(dataset)} images")

# Extract images and names from dataset
images = []
image_names = []
for i in range(len(dataset)):
    sample = dataset[i]
    # Convert from numpy (C, H, W) to torch tensor
    image_tensor = torch.from_numpy(sample["image"])
    images.append(image_tensor)
    image_names.append(sample["filename"])

images = torch.stack(images)
print(f"Images shape: {images.shape}")
print(f"Image names: {image_names}")

# Display sample images
plt.figure(figsize=(15, 4))
for i in range(len(images)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i].permute(1, 2, 0).detach().cpu().numpy())
    plt.title(f"{image_names[i].title()}")
    plt.axis("off")
plt.suptitle("Sample Test Images (128x128)", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Traditional Image Compressors
# ----------------------------------------
# Let's start with traditional image compression formats

print("\n" + "=" * 50)
print("Testing Traditional Image Compressors")
print("=" * 50)

# Initialize traditional compressors
traditional_compressors = {
    "JPEG": JPEGCompressor(quality=75, collect_stats=True, return_bits=True),
    "JPEG 2000": JPEG2000Compressor(quality=75, collect_stats=True, return_bits=True),
    "PNG": PNGCompressor(quality=9, collect_stats=True, return_bits=True),  # PNG: 0-9 compression level (lossless)
    "WebP": WebPCompressor(quality=75, collect_stats=True, return_bits=True),
}

print("Note: PNG uses lossless compression, so it will typically show higher bit counts")
print("but maintains perfect image quality. We're using 128x128 images for more reasonable")
print("PNG file sizes while still demonstrating the compression characteristics.")
print()

# Test each traditional compressor
traditional_results: Dict[str, Optional[Dict[str, Any]]] = {}
for name, compressor in traditional_compressors.items():
    print(f"\nTesting {name} Compressor...")
    try:
        # Compress images
        compressed_images, bits_per_image = compressor(images)
        stats = compressor.get_stats()

        traditional_results[name] = {"compressed_images": compressed_images, "bits_per_image": bits_per_image, "avg_bits": np.mean(bits_per_image), "compression_ratio": stats.get("avg_compression_ratio", 0), "stats": stats}

        # Calculate compression ratio manually if not provided
        original_size = images.shape[1] * images.shape[2] * images.shape[3] * 8  # RGB image in bits
        result = traditional_results[name]
        if result is not None and result["compression_ratio"] == 0:
            result["compression_ratio"] = original_size / result["avg_bits"]

        if result is not None:
            print(f"  ✓ Average bits per image: {result['avg_bits']:.0f}")
            print(f"  ✓ Average compression ratio: {result['compression_ratio']:.2f}:1")

    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        traditional_results[name] = None

# %%
# Advanced Compressors (BPG and JPEG XL)
# -----------------------------------------------
# Test more advanced compression formats

print("\n" + "=" * 50)
print("Testing Advanced Image Compressors")
print("=" * 50)

# BPG Compressor (if available)
print("\nTesting BPG Compressor...")
try:
    bpg_compressor = BPGCompressor(quality=30, collect_stats=True, return_bits=True)
    compressed_images_bpg, bits_per_image_bpg = bpg_compressor(images)
    bpg_stats = bpg_compressor.get_stats()

    traditional_results["BPG"] = {"compressed_images": compressed_images_bpg, "bits_per_image": bits_per_image_bpg, "avg_bits": np.mean(bits_per_image_bpg), "compression_ratio": bpg_stats.get("avg_compression_ratio", 0), "stats": bpg_stats}

    # Calculate compression ratio manually if not provided
    original_size = images.shape[1] * images.shape[2] * images.shape[3] * 8  # RGB image in bits
    result = traditional_results["BPG"]
    if result is not None and result["compression_ratio"] == 0:
        result["compression_ratio"] = original_size / result["avg_bits"]

    if result is not None:
        print(f"  ✓ Average bits per image: {result['avg_bits']:.0f}")
        print(f"  ✓ Average compression ratio: {result['compression_ratio']:.2f}:1")

except Exception as e:
    print(f"  ✗ BPG not available: {str(e)}")
    traditional_results["BPG"] = None

# JPEG XL Compressor (if available)
print("\nTesting JPEG XL Compressor...")
try:
    jpegxl_compressor = JPEGXLCompressor(quality=75, collect_stats=True, return_bits=True)
    compressed_images_jxl, bits_per_image_jxl = jpegxl_compressor(images)
    jxl_stats = jpegxl_compressor.get_stats()

    traditional_results["JPEG XL"] = {"compressed_images": compressed_images_jxl, "bits_per_image": bits_per_image_jxl, "avg_bits": np.mean(bits_per_image_jxl), "compression_ratio": jxl_stats.get("avg_compression_ratio", 0), "stats": jxl_stats}

    # Calculate compression ratio manually if not provided
    original_size = images.shape[1] * images.shape[2] * images.shape[3] * 8  # RGB image in bits
    result = traditional_results["JPEG XL"]
    if result is not None and result["compression_ratio"] == 0:
        result["compression_ratio"] = original_size / result["avg_bits"]

    if result is not None:
        print(f"  ✓ Average bits per image: {result['avg_bits']:.0f}")
        print(f"  ✓ Average compression ratio: {result['compression_ratio']:.2f}:1")

except Exception as e:
    print(f"  ✗ JPEG XL not available: {str(e)}")
    traditional_results["JPEG XL"] = None

# %%
# Neural Compressors
# ----------------------------
# Test neural network-based compression models
# The example images are 128x128, which needs to be resized for neural compressors
print(f"Input images shape: {images.shape}")

# Neural compressors typically expect larger images (256x256 or more)
# We'll resize for neural compression but note the size difference
neural_images = torch.nn.functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
print(f"Resized for neural compression: {neural_images.shape}")

# Use only one neural compression method to avoid downloading all models
neural_methods = [
    "bmshj2018_factorized",  # Most common and well-tested method
]

neural_results: Dict[str, Optional[Dict[str, Any]]] = {}

for method in neural_methods:
    print(f"\nTesting Neural Compressor: {method}")
    try:
        # Test with a middle-range quality
        neural_compressor = NeuralCompressor(method=method, quality=4, collect_stats=True, return_bits=True)  # Middle quality level

        # Compress images (resize to 256x256 for neural compression)
        compressed_images_neural, bits_per_image_neural = neural_compressor(neural_images)
        neural_stats = neural_compressor.get_stats()

        neural_results[method] = {"compressed_images": compressed_images_neural, "bits_per_image": bits_per_image_neural.detach().cpu().numpy(), "avg_bits": float(bits_per_image_neural.detach().mean().cpu()), "compression_ratio": neural_stats.get("avg_compression_ratio", 0), "stats": neural_stats}

        # Calculate compression ratio manually if not provided (based on 256x256 size)
        original_neural_size = neural_images.shape[1] * neural_images.shape[2] * neural_images.shape[3] * 8
        result = neural_results[method]
        if result is not None and result["compression_ratio"] == 0:
            result["compression_ratio"] = original_neural_size / result["avg_bits"]

        if result is not None:
            print(f"  ✓ Average bits per image: {result['avg_bits']:.0f}")
            print(f"  ✓ Average compression ratio: {result['compression_ratio']:.2f}:1")

    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}...")
        neural_results[method] = None

# %%
# Compression Results Visualization
# ----------------------------------------
# Create comprehensive visualizations of the compression results

# Prepare data for plotting
all_results = {**traditional_results, **neural_results}
valid_results = {k: v for k, v in all_results.items() if v is not None}

if valid_results:
    # Extract data for plotting
    compressor_names = list(valid_results.keys())
    avg_bits = [result["avg_bits"] for result in valid_results.values()]
    compression_ratios = [result["compression_ratio"] for result in valid_results.values()]

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Average bits per image
    bars1 = ax1.bar(compressor_names, avg_bits, alpha=0.7, color="skyblue", edgecolor="navy")
    ax1.set_title("Average Bits per Image", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Bits per Image")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, bits in zip(bars1, avg_bits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{bits:.0f}", ha="center", va="bottom", fontweight="bold")

    # Plot 2: Compression ratios
    bars2 = ax2.bar(compressor_names, compression_ratios, alpha=0.7, color="lightgreen", edgecolor="darkgreen")
    ax2.set_title("Compression Ratios", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Compression Ratio (X:1)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, ratio in zip(bars2, compression_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + height * 0.01, f"{ratio:.1f}:1", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.show()

# %%
# Compressed Images Visualization
# ----------------------------------------
# Show compressed images from different methods

# Select a few representative compressors for visual comparison
demo_compressors = []
demo_names = []

# Add the best traditional compressor
if "JPEG" in valid_results:
    demo_compressors.append(valid_results["JPEG"]["compressed_images"])
    demo_names.append("JPEG")

# Add BPG if available
if "BPG" in valid_results:
    demo_compressors.append(valid_results["BPG"]["compressed_images"])
    demo_names.append("BPG")

# Add a neural compressor if available
neural_demo = None
for method in ["bmshj2018_factorized", "bmshj2018_hyperprior"]:
    if method in valid_results:
        # Neural images need to be resized back to 128x128 for visualization
        neural_compressed = valid_results[method]["compressed_images"]
        resized_neural = torch.nn.functional.interpolate(neural_compressed, size=(128, 128), mode="bilinear", align_corners=False)
        demo_compressors.append(resized_neural)
        demo_names.append(f"Neural ({method})")
        neural_demo = method
        break

if demo_compressors:
    num_methods = len(demo_compressors)
    num_images = min(2, len(images))  # Show first 2 images

    fig, axes = plt.subplots(num_images, num_methods + 1, figsize=(4 * (num_methods + 1), 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for img_idx in range(num_images):
        # Show original image
        axes[img_idx, 0].imshow(images[img_idx].permute(1, 2, 0).detach().cpu().numpy())
        axes[img_idx, 0].set_title("Original")
        axes[img_idx, 0].axis("off")

        # Show compressed images
        for method_idx, (compressed_imgs, method_name) in enumerate(zip(demo_compressors, demo_names)):
            axes[img_idx, method_idx + 1].imshow(compressed_imgs[img_idx].permute(1, 2, 0).detach().cpu().numpy())
            axes[img_idx, method_idx + 1].set_title(method_name)
            axes[img_idx, method_idx + 1].axis("off")

    plt.suptitle("Compressed Image Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

# %%
# Performance Summary Table
# --------------------------------
# Create a summary table of all compression results

print("\n" + "=" * 80)
print("COMPRESSION PERFORMANCE SUMMARY")
print("=" * 80)
print(f"{'Compressor':<20} {'Avg Bits':<12} {'Compression':<15} {'Status':<10}")
print("-" * 80)

for name, result in all_results.items():
    if result is not None:
        avg_bits_str = f"{result['avg_bits']:.0f}"
        ratio_str = f"{result['compression_ratio']:.1f}:1"
        status_str = "✓ Success"
    else:
        avg_bits_str = "N/A"
        ratio_str = "N/A"
        status_str = "✗ Failed"

    print(f"{name:<20} {avg_bits_str:<12} {ratio_str:<15} {status_str:<10}")

print("-" * 80)

# Calculate original image size for reference
original_bits = images.shape[1] * images.shape[2] * images.shape[3] * 8  # 8 bits per channel (128x128 RGB)
print(f"Original image size: {original_bits} bits per image (128x128 RGB)")
print(f"Neural compressors used 256x256 images: {256*256*3*8} bits per image")

# %%
# Quality vs Compression Trade-off (JPEG Example)
# -------------------------------------------------------
# Demonstrate quality vs compression trade-off using JPEG

print("\n" + "=" * 50)
print("Quality vs Compression Trade-off Analysis")
print("=" * 50)

# Test JPEG at different quality levels
jpeg_qualities = [10, 25, 50, 75, 90, 95]
jpeg_trade_off_results = []

print("\nTesting JPEG at different quality levels...")
for quality in jpeg_qualities:
    try:
        jpeg_compressor = JPEGCompressor(quality=quality, collect_stats=True, return_bits=True)
        compressed_imgs, bits_list = jpeg_compressor(images)
        stats = jpeg_compressor.get_stats()

        avg_bits = np.mean(bits_list)
        compression_ratio = stats.get("avg_compression_ratio", 0)

        # Calculate compression ratio manually if not provided
        original_size = images.shape[1] * images.shape[2] * images.shape[3] * 8  # RGB image in bits
        if compression_ratio == 0:
            compression_ratio = original_size / avg_bits

        jpeg_trade_off_results.append({"quality": quality, "avg_bits": avg_bits, "compression_ratio": compression_ratio})

        print(f"  Quality {quality:2d}: {avg_bits:6.0f} bits, {compression_ratio:4.1f}:1 compression")

    except Exception as e:
        print(f"  Quality {quality:2d}: Failed - {str(e)}")

# Plot quality vs compression trade-off
if jpeg_trade_off_results:
    qualities = [r["quality"] for r in jpeg_trade_off_results]
    bits_values = [r["avg_bits"] for r in jpeg_trade_off_results]
    ratios = [r["compression_ratio"] for r in jpeg_trade_off_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Quality vs Bits
    ax1.plot(qualities, bits_values, "o-", linewidth=2, markersize=8, color="blue")
    ax1.set_xlabel("JPEG Quality")
    ax1.set_ylabel("Average Bits per Image")
    ax1.set_title("Quality vs File Size", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Quality vs Compression Ratio
    ax2.plot(qualities, ratios, "o-", linewidth=2, markersize=8, color="red")
    ax2.set_xlabel("JPEG Quality")
    ax2.set_ylabel("Compression Ratio (X:1)")
    ax2.set_title("Quality vs Compression Ratio", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
# Key Takeaways and Recommendations
# --------------------------------------

print("\n" + "=" * 50)
print("KEY TAKEAWAYS AND RECOMMENDATIONS")
print("=" * 50)

print(
    """
Based on this comprehensive comparison of image compressors:

1. **Traditional Compressors**:
   • JPEG: Good balance of compression and compatibility
   • PNG: Lossless compression, higher file sizes but perfect quality
   • WebP: Modern format with better compression than JPEG
   • JPEG 2000: Advanced features but limited adoption

2. **Advanced Compressors**:
   • BPG: Excellent compression ratios (if available)
   • JPEG XL: Next-generation format with superior performance

3. **Neural Compressors**:
   • State-of-the-art compression ratios
   • Require specialized hardware for optimal performance
   • Different methods optimized for different scenarios

4. **Recommendations**:
   • For web/general use: WebP or JPEG
   • For maximum compression: BPG or Neural compressors
   • For research/experimentation: Neural compressors
   • For archival/lossless: PNG or JPEG 2000 lossless mode

Note: PNG shows higher bit counts because it's lossless compression -
it preserves perfect image quality at the cost of larger file sizes.

Choose the compressor based on your specific requirements for:
- Compression ratio vs quality trade-off
- Compatibility requirements
- Processing time constraints
- Hardware availability
"""
)

print("\nExample completed successfully! ✓")
