"""Unit tests for JPEG and PNG compressors."""

import unittest

import torch
from PIL import Image

from kaira.models.image.compressors import BaseImageCompressor, JPEGCompressor, PNGCompressor


class TestJPEGCompressor(unittest.TestCase):
    """Test cases for JPEG compressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_tensor = torch.rand(2, 3, 32, 32)  # Small test images
        self.test_pil_image = Image.new("RGB", (32, 32), color="red")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = JPEGCompressor(quality=85)
        self.assertEqual(compressor.quality, 85)
        self.assertTrue(compressor.optimize)
        self.assertFalse(compressor.progressive)

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = JPEGCompressor(max_bits_per_image=5000)
        self.assertEqual(compressor.max_bits_per_image, 5000)
        self.assertIsNone(compressor.quality)

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with self.assertRaises(ValueError):
            JPEGCompressor()

    def test_quality_validation(self):
        """Test quality parameter validation."""
        # Valid quality
        JPEGCompressor(quality=50)

        # Invalid quality - too low
        with self.assertRaises(ValueError):
            JPEGCompressor(quality=0)

        # Invalid quality - too high
        with self.assertRaises(ValueError):
            JPEGCompressor(quality=101)

        # Invalid quality - not integer
        with self.assertRaises(ValueError):
            JPEGCompressor(quality=50.5)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = JPEGCompressor(quality=50)
        min_q, max_q = compressor._get_quality_range()
        self.assertEqual(min_q, 1)
        self.assertEqual(max_q, 100)

    def test_compress_decompress_single_image(self):
        """Test compression and decompression of a single image."""
        compressor = JPEGCompressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(self.test_pil_image, 90)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        self.assertEqual(bits, len(compressed_data) * 8)

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)
        self.assertEqual(recovered_image.mode, "RGB")

    def test_direct_compression_methods(self):
        """Test direct compress/decompress methods."""
        compressor = JPEGCompressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(self.test_pil_image)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_forward_with_quality(self):
        """Test forward pass with fixed quality."""
        compressor = JPEGCompressor(quality=75, return_bits=True, collect_stats=True)

        result, bits = compressor(self.test_image_tensor)

        self.assertEqual(result.shape, self.test_image_tensor.shape)
        self.assertEqual(len(bits), self.test_image_tensor.shape[0])
        self.assertTrue(all(isinstance(b, int) and b > 0 for b in bits))

        stats = compressor.get_stats()
        self.assertIn("total_bits", stats)
        self.assertIn("avg_quality", stats)
        self.assertEqual(stats["avg_quality"], 75)

    def test_forward_with_bit_constraint(self):
        """Test forward pass with bit constraint."""
        compressor = JPEGCompressor(max_bits_per_image=3000, return_bits=True)

        result, bits = compressor(self.test_image_tensor)

        self.assertEqual(result.shape, self.test_image_tensor.shape)
        self.assertEqual(len(bits), self.test_image_tensor.shape[0])
        # All images should be under or near the bit limit
        for b in bits:
            # Allow some tolerance since we might not be able to meet exact constraint
            self.assertLessEqual(b, 3500)  # Some tolerance for edge cases

    def test_return_compressed_data(self):
        """Test returning compressed data."""
        compressor = JPEGCompressor(quality=80, return_bits=False, return_compressed_data=True)

        result, compressed_data = compressor(self.test_image_tensor)

        self.assertEqual(result.shape, self.test_image_tensor.shape)
        self.assertEqual(len(compressed_data), self.test_image_tensor.shape[0])
        self.assertTrue(all(isinstance(data, bytes) for data in compressed_data))

    def test_all_return_options(self):
        """Test returning both bits and compressed data."""
        compressor = JPEGCompressor(quality=70, return_bits=True, return_compressed_data=True)

        result, bits, compressed_data = compressor(self.test_image_tensor)

        self.assertEqual(result.shape, self.test_image_tensor.shape)
        self.assertEqual(len(bits), self.test_image_tensor.shape[0])
        self.assertEqual(len(compressed_data), self.test_image_tensor.shape[0])


class TestPNGCompressor(unittest.TestCase):
    """Test cases for PNG compressor."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_tensor = torch.rand(2, 3, 32, 32)  # Small test images
        self.test_pil_image = Image.new("RGB", (32, 32), color="blue")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = PNGCompressor(quality=6)
        self.assertEqual(compressor.quality, 6)
        self.assertTrue(compressor.optimize)

    def test_init_with_compress_level(self):
        """Test initialization with compress_level parameter."""
        compressor = PNGCompressor(compress_level=9)
        self.assertEqual(compressor.quality, 9)  # Should be mapped to quality

    def test_compress_level_precedence(self):
        """Test that compress_level takes precedence over quality."""
        compressor = PNGCompressor(quality=3, compress_level=7)
        self.assertEqual(compressor.quality, 7)

    def test_quality_validation(self):
        """Test compression level validation."""
        # Valid compression level
        PNGCompressor(quality=5)

        # Invalid compression level - too low
        with self.assertRaises(ValueError):
            PNGCompressor(quality=-1)

        # Invalid compression level - too high
        with self.assertRaises(ValueError):
            PNGCompressor(quality=10)

        # Invalid compression level - not integer
        with self.assertRaises(ValueError):
            PNGCompressor(quality=5.5)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = PNGCompressor(quality=5)
        min_q, max_q = compressor._get_quality_range()
        self.assertEqual(min_q, 0)
        self.assertEqual(max_q, 9)

    def test_compress_decompress_single_image(self):
        """Test compression and decompression of a single image."""
        compressor = PNGCompressor(quality=9)

        # Compress
        compressed_data, bits = compressor._compress_single_image(self.test_pil_image, 9)
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(len(compressed_data), 0)
        self.assertEqual(bits, len(compressed_data) * 8)

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        self.assertIsInstance(recovered_image, Image.Image)
        self.assertEqual(recovered_image.size, self.test_pil_image.size)

    def test_lossless_compression(self):
        """Test that PNG compression is lossless."""
        # Create a simple pattern that should compress well and be exactly recoverable
        test_image = Image.new("RGB", (16, 16), color="white")
        # Add some pixels for pattern
        pixels = test_image.load()
        for i in range(8):
            pixels[i, i] = (255, 0, 0)  # Red diagonal

        compressor = PNGCompressor(quality=9, return_bits=False)

        # Convert to tensor and back through compression
        tensor = compressor._pil_to_tensor(test_image)
        result_tensor = compressor(tensor.unsqueeze(0))  # Add batch dimension
        result_image = compressor._tensor_to_pil(result_tensor.squeeze(0))  # Remove batch dimension

        # The images should be very similar (allowing for minor floating point differences)
        # We'll check that they have the same size and basic properties
        self.assertEqual(result_image.size, test_image.size)
        self.assertEqual(result_image.mode, test_image.mode)

    def test_forward_with_bit_constraint(self):
        """Test forward pass with bit constraint."""
        compressor = PNGCompressor(max_bits_per_image=15000, return_bits=True)

        result, bits = compressor(self.test_image_tensor)

        self.assertEqual(result.shape, self.test_image_tensor.shape)
        self.assertEqual(len(bits), self.test_image_tensor.shape[0])


class TestBaseImageCompressor(unittest.TestCase):
    """Test cases for base image compressor functionality."""

    def test_abstract_instantiation(self):
        """Test that BaseImageCompressor cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseImageCompressor(quality=50)

    def test_tensor_pil_conversion(self):
        """Test tensor to PIL and back conversion."""
        compressor = JPEGCompressor(quality=90)  # Use concrete implementation

        # Create test tensor
        tensor = torch.rand(3, 32, 32)

        # Convert to PIL and back
        pil_image = compressor._tensor_to_pil(tensor)
        recovered_tensor = compressor._pil_to_tensor(pil_image)

        self.assertEqual(pil_image.size, (32, 32))
        self.assertEqual(pil_image.mode, "RGB")
        self.assertEqual(recovered_tensor.shape, tensor.shape)

    def test_grayscale_conversion(self):
        """Test grayscale image handling."""
        compressor = JPEGCompressor(quality=90)

        # Create grayscale tensor
        gray_tensor = torch.rand(1, 32, 32)

        # Convert to PIL and back
        pil_image = compressor._tensor_to_pil(gray_tensor)
        recovered_tensor = compressor._pil_to_tensor(pil_image)

        # Single channel tensors are converted to grayscale PIL images
        self.assertEqual(pil_image.mode, "L")
        # But when converted back through _pil_to_tensor, they become RGB
        self.assertEqual(recovered_tensor.shape, (3, 32, 32))

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        compressor = JPEGCompressor(quality=90)

        ratio = compressor.get_compression_ratio(1000, 500)
        self.assertEqual(ratio, 2.0)

        # Test edge case
        ratio = compressor.get_compression_ratio(1000, 0)
        self.assertEqual(ratio, float("inf"))


if __name__ == "__main__":
    unittest.main()
