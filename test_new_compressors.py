"""Test script for JPEG, PNG, JPEG XL, JPEG2000, and WebP compressors."""

import torch
from PIL import Image


# Test the new compressors
def test_all_compressors():
    """Test all compressors with sample data."""

    # Import all the compressors
    from kaira.models.image.compressors import (
        JPEG2000Compressor,
        JPEGCompressor,
        JPEGXLCompressor,
        PNGCompressor,
        WebPCompressor,
    )

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

    print("Testing JPEG XL Compressor...")
    print("=" * 50)

    # Test JPEG XL with fixed quality
    jpegxl_compressor = JPEGXLCompressor(quality=85, collect_stats=True, return_bits=True)
    jpegxl_result, jpegxl_bits = jpegxl_compressor(x)

    print(f"JPEG XL - Input shape: {x.shape}")
    print(f"JPEG XL - Output shape: {jpegxl_result.shape}")
    print(f"JPEG XL - Bits per image: {jpegxl_bits}")
    print(f"JPEG XL - Stats: {jpegxl_compressor.get_stats()}")
    print()

    # Test JPEG XL with lossless mode
    jpegxl_lossless = JPEGXLCompressor(quality=100, lossless=True, collect_stats=True, return_bits=True)
    jpegxl_result_lossless, jpegxl_bits_lossless = jpegxl_lossless(x)

    print(f"JPEG XL Lossless - Bits per image: {jpegxl_bits_lossless}")
    print(f"JPEG XL Lossless - Stats: {jpegxl_lossless.get_stats()}")
    print()

    print("Testing JPEG 2000 Compressor...")
    print("=" * 50)

    # Test JPEG 2000 with fixed quality
    jpeg2000_compressor = JPEG2000Compressor(quality=85, collect_stats=True, return_bits=True)
    jpeg2000_result, jpeg2000_bits = jpeg2000_compressor(x)

    print(f"JPEG 2000 - Input shape: {x.shape}")
    print(f"JPEG 2000 - Output shape: {jpeg2000_result.shape}")
    print(f"JPEG 2000 - Bits per image: {jpeg2000_bits}")
    print(f"JPEG 2000 - Stats: {jpeg2000_compressor.get_stats()}")
    print()

    # Test JPEG 2000 with bit constraint
    jpeg2000_constrained = JPEG2000Compressor(max_bits_per_image=4000, collect_stats=True, return_bits=True)
    jpeg2000_result_const, jpeg2000_bits_const = jpeg2000_constrained(x)

    print(f"JPEG 2000 Constrained - Bits per image: {jpeg2000_bits_const}")
    print(f"JPEG 2000 Constrained - Stats: {jpeg2000_constrained.get_stats()}")
    print()

    print("Testing WebP Compressor...")
    print("=" * 50)

    # Test WebP with fixed quality
    webp_compressor = WebPCompressor(quality=85, collect_stats=True, return_bits=True)
    webp_result, webp_bits = webp_compressor(x)

    print(f"WebP - Input shape: {x.shape}")
    print(f"WebP - Output shape: {webp_result.shape}")
    print(f"WebP - Bits per image: {webp_bits}")
    print(f"WebP - Stats: {webp_compressor.get_stats()}")
    print()

    # Test WebP with lossless mode
    webp_lossless = WebPCompressor(lossless=True, collect_stats=True, return_bits=True)
    webp_result_lossless, webp_bits_lossless = webp_lossless(x)

    print(f"WebP Lossless - Bits per image: {webp_bits_lossless}")
    print(f"WebP Lossless - Stats: {webp_lossless.get_stats()}")
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

    # Test JPEG XL direct methods
    jpegxl_simple = JPEGXLCompressor(quality=90)
    jpegxl_data = jpegxl_simple.compress(test_image)
    jpegxl_recovered = jpegxl_simple.decompress(jpegxl_data)

    print(f"JPEG XL Direct - Original size: {test_image.size}")
    print(f"JPEG XL Direct - Compressed size: {len(jpegxl_data)} bytes")
    print(f"JPEG XL Direct - Recovered size: {jpegxl_recovered.size}")
    print()

    # Test JPEG 2000 direct methods
    jpeg2000_simple = JPEG2000Compressor(quality=90)
    jpeg2000_data = jpeg2000_simple.compress(test_image)
    jpeg2000_recovered = jpeg2000_simple.decompress(jpeg2000_data)

    print(f"JPEG 2000 Direct - Original size: {test_image.size}")
    print(f"JPEG 2000 Direct - Compressed size: {len(jpeg2000_data)} bytes")
    print(f"JPEG 2000 Direct - Recovered size: {jpeg2000_recovered.size}")
    print()

    # Test WebP direct methods
    webp_simple = WebPCompressor(quality=90)
    webp_data = webp_simple.compress(test_image)
    webp_recovered = webp_simple.decompress(webp_data)

    print(f"WebP Direct - Original size: {test_image.size}")
    print(f"WebP Direct - Compressed size: {len(webp_data)} bytes")
    print(f"WebP Direct - Recovered size: {webp_recovered.size}")
    print()

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_all_compressors()
