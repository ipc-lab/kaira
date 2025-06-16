"""Unit tests for JPEG XL, JPEG2000, and WebP compressors."""

import unittest

import torch
from PIL import Image

from kaira.models.image.compressors import (
    JPEG2000Compressor,
    JPEGXLCompressor,
    WebPCompressor,
)


class TestJPEGXLCompressor(unittest.TestCase):
    """Test cases for JPEG XL compressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_tensor = torch.rand(2, 3, 32, 32)  # Small test images
        self.test_pil_image = Image.new("RGB", (32, 32), color="red")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = JPEGXLCompressor(quality=85)
        self.assertEqual(compressor.quality, 85)
        self.assertEqual(compressor.effort, 7)
        self.assertFalse(compressor.lossless)

    def test_init_with_lossless(self):
        """Test initialization with lossless mode."""
        compressor = JPEGXLCompressor(quality=90, lossless=True)
        self.assertEqual(compressor.quality, 100)  # Quality gets overridden to 100 in lossless mode
        self.assertTrue(compressor.lossless)

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = JPEGXLCompressor(max_bits_per_image=3000)
        self.assertEqual(compressor.max_bits_per_image, 3000)
        self.assertIsNone(compressor.quality)

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with self.assertRaises(ValueError):
            JPEGXLCompressor()

    def test_effort_validation(self):
        """Test effort parameter validation."""
        # Valid effort
        JPEGXLCompressor(quality=85, effort=5)

        # Invalid effort - too low
        with self.assertRaises(ValueError):
            JPEGXLCompressor(quality=85, effort=0)

        # Invalid effort - too high
        with self.assertRaises(ValueError):
            JPEGXLCompressor(quality=85, effort=10)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = JPEGXLCompressor(quality=50)
        compressor._validate_quality(85)

        with self.assertRaises(ValueError):
            compressor._validate_quality(0)

        with self.assertRaises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = JPEGXLCompressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        self.assertEqual(min_q, 1)
        self.assertEqual(max_q, 100)

    def test_compress_decompress_single_image(self):
        """Test compression and decompression of a single image."""
        compressor = JPEGXLCompressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(self.test_pil_image, 90)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        self.assertEqual(bits, len(compressed_data) * 8)

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_direct_compression_methods(self):
        """Test direct compress/decompress methods."""
        compressor = JPEGXLCompressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(self.test_pil_image)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)


class TestJPEG2000Compressor(unittest.TestCase):
    """Test cases for JPEG 2000 compressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_tensor = torch.rand(2, 3, 32, 32)  # Small test images
        self.test_pil_image = Image.new("RGB", (32, 32), color="blue")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = JPEG2000Compressor(quality=85)
        self.assertEqual(compressor.quality, 85)
        self.assertEqual(compressor.progression_order, "LRCP")
        self.assertEqual(compressor.num_resolutions, 6)

    def test_init_with_irreversible(self):
        """Test initialization with irreversible parameter."""
        compressor = JPEG2000Compressor(quality=90, irreversible=True)
        self.assertEqual(compressor.quality, 90)
        self.assertTrue(compressor.irreversible)

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = JPEG2000Compressor(max_bits_per_image=4000)
        self.assertEqual(compressor.max_bits_per_image, 4000)
        self.assertIsNone(compressor.quality)

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with self.assertRaises(ValueError):
            JPEG2000Compressor()

    def test_progression_order_validation(self):
        """Test progression order validation."""
        # Valid progression orders
        for order in ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]:
            JPEG2000Compressor(quality=85, progression_order=order)

        # Invalid progression order
        with self.assertRaises(ValueError):
            JPEG2000Compressor(quality=85, progression_order="INVALID")

    def test_num_resolutions_validation(self):
        """Test number of resolutions validation."""
        # Valid resolutions
        JPEG2000Compressor(quality=85, num_resolutions=3)
        JPEG2000Compressor(quality=85, num_resolutions=33)

        # Invalid resolutions - too low
        with self.assertRaises(ValueError):
            JPEG2000Compressor(quality=85, num_resolutions=0)

        # Invalid resolutions - too high
        with self.assertRaises(ValueError):
            JPEG2000Compressor(quality=85, num_resolutions=34)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = JPEG2000Compressor(quality=50)
        compressor._validate_quality(85)

        with self.assertRaises(ValueError):
            compressor._validate_quality(0)

        with self.assertRaises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = JPEG2000Compressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        self.assertEqual(min_q, 1)
        self.assertEqual(max_q, 100)

    def test_compress_decompress_single_image(self):
        """Test compression and decompression of a single image."""
        compressor = JPEG2000Compressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(self.test_pil_image, 90)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        self.assertEqual(bits, len(compressed_data) * 8)

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_direct_compression_methods(self):
        """Test direct compress/decompress methods."""
        compressor = JPEG2000Compressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(self.test_pil_image)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)


class TestWebPCompressor(unittest.TestCase):
    """Test cases for WebP compressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_tensor = torch.rand(2, 3, 32, 32)  # Small test images
        self.test_pil_image = Image.new("RGB", (32, 32), color="green")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = WebPCompressor(quality=85)
        self.assertEqual(compressor.quality, 85)
        self.assertEqual(compressor.method, 4)
        self.assertFalse(compressor.lossless)

    def test_init_with_lossless(self):
        """Test initialization with lossless mode."""
        compressor = WebPCompressor(lossless=True)
        self.assertTrue(compressor.lossless)

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = WebPCompressor(max_bits_per_image=3500)
        self.assertEqual(compressor.max_bits_per_image, 3500)
        self.assertIsNone(compressor.quality)

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with self.assertRaises(ValueError):
            WebPCompressor()

    def test_method_validation(self):
        """Test method parameter validation."""
        # Valid methods
        for method in range(7):
            WebPCompressor(quality=85, method=method)

        # Invalid method - too low
        with self.assertRaises(ValueError):
            WebPCompressor(quality=85, method=-1)

        # Invalid method - too high
        with self.assertRaises(ValueError):
            WebPCompressor(quality=85, method=7)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = WebPCompressor(quality=50)
        compressor._validate_quality(85)

        with self.assertRaises(ValueError):
            compressor._validate_quality(0)

        with self.assertRaises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = WebPCompressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        self.assertEqual(min_q, 1)
        self.assertEqual(max_q, 100)

    def test_compress_decompress_single_image(self):
        """Test compression and decompression of a single image."""
        compressor = WebPCompressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(self.test_pil_image, 90)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        self.assertEqual(bits, len(compressed_data) * 8)

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_lossless_compression(self):
        """Test lossless compression mode."""
        compressor = WebPCompressor(lossless=True)

        # Test that compress method works with lossless
        compressed_data = compressor.compress(self.test_pil_image)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)

        # Decompress and verify
        recovered_image = compressor.decompress(compressed_data)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_direct_compression_methods(self):
        """Test direct compress/decompress methods."""
        compressor = WebPCompressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(self.test_pil_image)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)


if __name__ == "__main__":
    unittest.main()
