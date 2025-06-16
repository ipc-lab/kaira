"""Test script for JPEG and PNG compressors."""

import torch
from PIL import Image


# Test the new compressors
def test_jpeg_png_compressors():
    """Test JPEG and PNG compressors with sample data."""

    # Import the new compressors
    from kaira.models.image.compressors import JPEGCompressor, PNGCompressor

    # Create a sample batch of random images
    batch_size = 2
    channels = 3
    height = 64
    width = 64

    # Create random tensor with values in [0, 1]
    x = torch.rand(batch_size, channels, height, width)

    print("Testing JPEG Compressor...")
    print("=" * 50)

    # Test JPEG with fixed quality
    jpeg_compressor = JPEGCompressor(quality=85, collect_stats=True, return_bits=True)
    jpeg_result, jpeg_bits = jpeg_compressor(x)

    print(f"JPEG - Input shape: {x.shape}")
    print(f"JPEG - Output shape: {jpeg_result.shape}")
    print(f"JPEG - Bits per image: {jpeg_bits}")
    print(f"JPEG - Stats: {jpeg_compressor.get_stats()}")
    print()

    # Test JPEG with bit constraint
    jpeg_constrained = JPEGCompressor(max_bits_per_image=5000, collect_stats=True, return_bits=True)
    jpeg_result_const, jpeg_bits_const = jpeg_constrained(x)

    print(f"JPEG Constrained - Bits per image: {jpeg_bits_const}")
    print(f"JPEG Constrained - Stats: {jpeg_constrained.get_stats()}")
    print()

    print("Testing PNG Compressor...")
    print("=" * 50)

    # Test PNG with fixed compression level
    png_compressor = PNGCompressor(quality=6, collect_stats=True, return_bits=True)
    png_result, png_bits = png_compressor(x)

    print(f"PNG - Input shape: {x.shape}")
    print(f"PNG - Output shape: {png_result.shape}")
    print(f"PNG - Bits per image: {png_bits}")
    print(f"PNG - Stats: {png_compressor.get_stats()}")
    print()

    # Test PNG with bit constraint
    png_constrained = PNGCompressor(max_bits_per_image=20000, collect_stats=True, return_bits=True)
    png_result_const, png_bits_const = png_constrained(x)

    print(f"PNG Constrained - Bits per image: {png_bits_const}")
    print(f"PNG Constrained - Stats: {png_constrained.get_stats()}")
    print()

    # Test direct compression/decompression methods
    print("Testing direct compression methods...")
    print("=" * 50)

    # Create a simple test image
    test_image = Image.new("RGB", (32, 32), color="red")

    # Test JPEG direct methods
    jpeg_simple = JPEGCompressor(quality=90)
    jpeg_data = jpeg_simple.compress(test_image)
    jpeg_recovered = jpeg_simple.decompress(jpeg_data)

    print(f"JPEG Direct - Original size: {test_image.size}")
    print(f"JPEG Direct - Compressed size: {len(jpeg_data)} bytes")
    print(f"JPEG Direct - Recovered size: {jpeg_recovered.size}")
    print()

    # Test PNG direct methods
    png_simple = PNGCompressor(quality=9)
    png_data = png_simple.compress(test_image)
    png_recovered = png_simple.decompress(png_data)

    print(f"PNG Direct - Original size: {test_image.size}")
    print(f"PNG Direct - Compressed size: {len(png_data)} bytes")
    print(f"PNG Direct - Recovered size: {png_recovered.size}")
    print()

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_jpeg_png_compressors()
