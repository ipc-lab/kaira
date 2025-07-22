"""Integration tests for JPEG, PNG, JPEG XL, JPEG2000, and WebP compressors."""

import pytest
import torch
from PIL import Image

from kaira.models.image.compressors import (
    JPEG2000Compressor,
    JPEGCompressor,
    JPEGXLCompressor,
    PNGCompressor,
    WebPCompressor,
)


@pytest.fixture
def sample_batch():
    """Create a sample batch of random images."""
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    return torch.rand(batch_size, channels, height, width)


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return Image.new("RGB", (32, 32), color="red")


class TestJPEGCompressorIntegration:
    """Integration tests for JPEG compressor."""

    def test_jpeg_with_quality(self, sample_batch):
        """Test JPEG compression with fixed quality."""
        jpeg_compressor = JPEGCompressor(quality=85, collect_stats=True, return_bits=True)
        jpeg_result, jpeg_bits = jpeg_compressor(sample_batch)

        assert jpeg_result.shape == sample_batch.shape
        assert isinstance(jpeg_bits, list)
        assert len(jpeg_bits) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpeg_bits)

        stats = jpeg_compressor.get_stats()
        assert stats is not None

    def test_jpeg_with_bit_constraint(self, sample_batch):
        """Test JPEG compression with bit constraint."""
        jpeg_constrained = JPEGCompressor(max_bits_per_image=5000, collect_stats=True, return_bits=True)
        jpeg_result_const, jpeg_bits_const = jpeg_constrained(sample_batch)

        assert jpeg_result_const.shape == sample_batch.shape
        assert isinstance(jpeg_bits_const, list)
        assert len(jpeg_bits_const) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpeg_bits_const)

        stats = jpeg_constrained.get_stats()
        assert stats is not None

    def test_jpeg_direct_methods(self, test_image):
        """Test JPEG direct compression methods."""
        jpeg_simple = JPEGCompressor(quality=90)
        jpeg_data = jpeg_simple.compress(test_image)
        jpeg_recovered = jpeg_simple.decompress(jpeg_data)

        assert isinstance(jpeg_data, bytes)
        assert len(jpeg_data) > 0
        assert jpeg_recovered.size == test_image.size


class TestPNGCompressorIntegration:
    """Integration tests for PNG compressor."""

    def test_png_with_quality(self, sample_batch):
        """Test PNG compression with fixed compression level."""
        png_compressor = PNGCompressor(quality=6, collect_stats=True, return_bits=True)
        png_result, png_bits = png_compressor(sample_batch)

        assert png_result.shape == sample_batch.shape
        assert isinstance(png_bits, list)
        assert len(png_bits) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in png_bits)

        stats = png_compressor.get_stats()
        assert stats is not None

    def test_png_with_bit_constraint(self, sample_batch):
        """Test PNG compression with bit constraint."""
        png_constrained = PNGCompressor(max_bits_per_image=20000, collect_stats=True, return_bits=True)
        png_result_const, png_bits_const = png_constrained(sample_batch)

        assert png_result_const.shape == sample_batch.shape
        assert isinstance(png_bits_const, list)
        assert len(png_bits_const) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in png_bits_const)

        stats = png_constrained.get_stats()
        assert stats is not None

    def test_png_direct_methods(self, test_image):
        """Test PNG direct compression methods."""
        png_simple = PNGCompressor(quality=9)
        png_data = png_simple.compress(test_image)
        png_recovered = png_simple.decompress(png_data)

        assert isinstance(png_data, bytes)
        assert len(png_data) > 0
        assert png_recovered.size == test_image.size


class TestJPEGXLCompressorIntegration:
    """Integration tests for JPEG XL compressor."""

    def test_jpegxl_with_quality(self, sample_batch):
        """Test JPEG XL compression with fixed quality."""
        jpegxl_compressor = JPEGXLCompressor(quality=85, collect_stats=True, return_bits=True)
        jpegxl_result, jpegxl_bits = jpegxl_compressor(sample_batch)

        assert jpegxl_result.shape == sample_batch.shape
        assert isinstance(jpegxl_bits, list)
        assert len(jpegxl_bits) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpegxl_bits)

        stats = jpegxl_compressor.get_stats()
        assert stats is not None

    def test_jpegxl_lossless_mode(self, sample_batch):
        """Test JPEG XL lossless compression."""
        jpegxl_lossless = JPEGXLCompressor(quality=100, lossless=True, collect_stats=True, return_bits=True)
        jpegxl_result_lossless, jpegxl_bits_lossless = jpegxl_lossless(sample_batch)

        assert jpegxl_result_lossless.shape == sample_batch.shape
        assert isinstance(jpegxl_bits_lossless, list)
        assert len(jpegxl_bits_lossless) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpegxl_bits_lossless)

        stats = jpegxl_lossless.get_stats()
        assert stats is not None

    def test_jpegxl_direct_methods(self, test_image):
        """Test JPEG XL direct compression methods."""
        jpegxl_simple = JPEGXLCompressor(quality=90)
        jpegxl_data = jpegxl_simple.compress(test_image)
        jpegxl_recovered = jpegxl_simple.decompress(jpegxl_data)

        assert isinstance(jpegxl_data, bytes)
        assert len(jpegxl_data) > 0
        assert jpegxl_recovered.size == test_image.size


class TestJPEG2000CompressorIntegration:
    """Integration tests for JPEG 2000 compressor."""

    def test_jpeg2000_with_quality(self, sample_batch):
        """Test JPEG 2000 compression with fixed quality."""
        jpeg2000_compressor = JPEG2000Compressor(quality=85, collect_stats=True, return_bits=True)
        jpeg2000_result, jpeg2000_bits = jpeg2000_compressor(sample_batch)

        assert jpeg2000_result.shape == sample_batch.shape
        assert isinstance(jpeg2000_bits, list)
        assert len(jpeg2000_bits) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpeg2000_bits)

        stats = jpeg2000_compressor.get_stats()
        assert stats is not None

    def test_jpeg2000_with_bit_constraint(self, sample_batch):
        """Test JPEG 2000 compression with bit constraint."""
        jpeg2000_constrained = JPEG2000Compressor(max_bits_per_image=4000, collect_stats=True, return_bits=True)
        jpeg2000_result_const, jpeg2000_bits_const = jpeg2000_constrained(sample_batch)

        assert jpeg2000_result_const.shape == sample_batch.shape
        assert isinstance(jpeg2000_bits_const, list)
        assert len(jpeg2000_bits_const) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in jpeg2000_bits_const)

        stats = jpeg2000_constrained.get_stats()
        assert stats is not None

    def test_jpeg2000_direct_methods(self, test_image):
        """Test JPEG 2000 direct compression methods."""
        jpeg2000_simple = JPEG2000Compressor(quality=90)
        jpeg2000_data = jpeg2000_simple.compress(test_image)
        jpeg2000_recovered = jpeg2000_simple.decompress(jpeg2000_data)

        assert isinstance(jpeg2000_data, bytes)
        assert len(jpeg2000_data) > 0
        assert jpeg2000_recovered.size == test_image.size


class TestWebPCompressorIntegration:
    """Integration tests for WebP compressor."""

    def test_webp_with_quality(self, sample_batch):
        """Test WebP compression with fixed quality."""
        webp_compressor = WebPCompressor(quality=85, collect_stats=True, return_bits=True)
        webp_result, webp_bits = webp_compressor(sample_batch)

        assert webp_result.shape == sample_batch.shape
        assert isinstance(webp_bits, list)
        assert len(webp_bits) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in webp_bits)

        stats = webp_compressor.get_stats()
        assert stats is not None

    def test_webp_lossless_mode(self, sample_batch):
        """Test WebP lossless compression."""
        webp_lossless = WebPCompressor(lossless=True, collect_stats=True, return_bits=True)
        webp_result_lossless, webp_bits_lossless = webp_lossless(sample_batch)

        assert webp_result_lossless.shape == sample_batch.shape
        assert isinstance(webp_bits_lossless, list)
        assert len(webp_bits_lossless) == sample_batch.shape[0]
        assert all(isinstance(b, int) and b > 0 for b in webp_bits_lossless)

        stats = webp_lossless.get_stats()
        assert stats is not None

    def test_webp_direct_methods(self, test_image):
        """Test WebP direct compression methods."""
        webp_simple = WebPCompressor(quality=90)
        webp_data = webp_simple.compress(test_image)
        webp_recovered = webp_simple.decompress(webp_data)

        assert isinstance(webp_data, bytes)
        assert len(webp_data) > 0
        assert webp_recovered.size == test_image.size


class TestAllCompressorsIntegration:
    """Integration tests for all compressors together."""

    def test_all_compressors_consistency(self, test_image):
        """Test that all compressors work consistently."""
        compressors = [
            JPEGCompressor(quality=90),
            PNGCompressor(quality=9),
            JPEGXLCompressor(quality=90),
            JPEG2000Compressor(quality=90),
            WebPCompressor(quality=90),
        ]

        for compressor in compressors:
            # Test direct compression/decompression
            compressed_data = compressor.compress(test_image)
            recovered_image = compressor.decompress(compressed_data)

            assert isinstance(compressed_data, bytes)
            assert len(compressed_data) > 0
            assert isinstance(recovered_image, Image.Image)
            assert recovered_image.size == test_image.size

    def test_compressor_statistics(self, sample_batch):
        """Test that all compressors can collect statistics."""
        compressors = [
            JPEGCompressor(quality=85, collect_stats=True, return_bits=True),
            PNGCompressor(quality=6, collect_stats=True, return_bits=True),
            JPEGXLCompressor(quality=85, collect_stats=True, return_bits=True),
            JPEG2000Compressor(quality=85, collect_stats=True, return_bits=True),
            WebPCompressor(quality=85, collect_stats=True, return_bits=True),
        ]

        for compressor in compressors:
            result, bits = compressor(sample_batch)

            assert result.shape == sample_batch.shape
            assert isinstance(bits, list)
            assert len(bits) == sample_batch.shape[0]
            assert all(isinstance(b, int) and b > 0 for b in bits)

            stats = compressor.get_stats()
            assert stats is not None
