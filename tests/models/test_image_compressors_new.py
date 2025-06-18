"""Unit tests for JPEG XL, JPEG2000, and WebP compressors."""

import pytest
import torch
from PIL import Image

from kaira.models.image.compressors import (
    JPEG2000Compressor,
    JPEGXLCompressor,
    WebPCompressor,
)


class TestJPEGXLCompressor:
    """Test cases for JPEG XL compressor."""

    @pytest.fixture
    def test_image_tensor(self):
        """Test image tensor fixture."""
        return torch.rand(2, 3, 32, 32)  # Small test images

    @pytest.fixture
    def test_pil_image(self):
        """Test PIL image fixture."""
        return Image.new("RGB", (32, 32), color="red")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = JPEGXLCompressor(quality=85)
        assert compressor.quality == 85
        assert compressor.effort == 7
        assert not compressor.lossless

    def test_init_with_lossless(self):
        """Test initialization with lossless mode."""
        compressor = JPEGXLCompressor(quality=90, lossless=True)
        assert compressor.quality == 100  # Quality gets overridden to 100 in lossless mode
        assert compressor.lossless

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = JPEGXLCompressor(max_bits_per_image=3000)
        assert compressor.max_bits_per_image == 3000
        assert compressor.quality is None

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with pytest.raises(ValueError):
            JPEGXLCompressor()

    def test_effort_validation(self):
        """Test effort parameter validation."""
        # Valid effort
        JPEGXLCompressor(quality=85, effort=5)

        # Invalid effort - too low
        with pytest.raises(ValueError):
            JPEGXLCompressor(quality=85, effort=0)

        # Invalid effort - too high
        with pytest.raises(ValueError):
            JPEGXLCompressor(quality=85, effort=10)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = JPEGXLCompressor(quality=50)
        compressor._validate_quality(85)

        with pytest.raises(ValueError):
            compressor._validate_quality(0)

        with pytest.raises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = JPEGXLCompressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        assert min_q == 1
        assert max_q == 100

    def test_compress_decompress_single_image(self, test_pil_image):
        """Test compression and decompression of a single image."""
        compressor = JPEGXLCompressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(test_pil_image, 90)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        assert bits == len(compressed_data) * 8

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size

    def test_direct_compression_methods(self, test_pil_image):
        """Test direct compress/decompress methods."""
        compressor = JPEGXLCompressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(test_pil_image)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size


class TestJPEG2000Compressor:
    """Test cases for JPEG 2000 compressor."""

    @pytest.fixture
    def test_image_tensor(self):
        """Test image tensor fixture."""
        return torch.rand(2, 3, 32, 32)  # Small test images

    @pytest.fixture
    def test_pil_image(self):
        """Test PIL image fixture."""
        return Image.new("RGB", (32, 32), color="blue")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = JPEG2000Compressor(quality=85)
        assert compressor.quality == 85
        assert compressor.progression_order == "LRCP"
        assert compressor.num_resolutions == 6

    def test_init_with_irreversible(self):
        """Test initialization with irreversible parameter."""
        compressor = JPEG2000Compressor(quality=90, irreversible=True)
        assert compressor.quality == 90
        assert compressor.irreversible

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = JPEG2000Compressor(max_bits_per_image=4000)
        assert compressor.max_bits_per_image == 4000
        assert compressor.quality is None

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with pytest.raises(ValueError):
            JPEG2000Compressor()

    def test_progression_order_validation(self):
        """Test progression order validation."""
        # Valid progression orders
        for order in ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]:
            JPEG2000Compressor(quality=85, progression_order=order)

        # Invalid progression order
        with pytest.raises(ValueError):
            JPEG2000Compressor(quality=85, progression_order="INVALID")

    def test_num_resolutions_validation(self):
        """Test number of resolutions validation."""
        # Valid resolutions
        JPEG2000Compressor(quality=85, num_resolutions=3)
        JPEG2000Compressor(quality=85, num_resolutions=33)

        # Invalid resolutions - too low
        with pytest.raises(ValueError):
            JPEG2000Compressor(quality=85, num_resolutions=0)

        # Invalid resolutions - too high
        with pytest.raises(ValueError):
            JPEG2000Compressor(quality=85, num_resolutions=34)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = JPEG2000Compressor(quality=50)
        compressor._validate_quality(85)

        with pytest.raises(ValueError):
            compressor._validate_quality(0)

        with pytest.raises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = JPEG2000Compressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        assert min_q == 1
        assert max_q == 100

    def test_compress_decompress_single_image(self, test_pil_image):
        """Test compression and decompression of a single image."""
        compressor = JPEG2000Compressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(test_pil_image, 90)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        assert bits == len(compressed_data) * 8

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size

    def test_direct_compression_methods(self, test_pil_image):
        """Test direct compress/decompress methods."""
        compressor = JPEG2000Compressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(test_pil_image)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size


class TestWebPCompressor:
    """Test cases for WebP compressor."""

    @pytest.fixture
    def test_image_tensor(self):
        """Test image tensor fixture."""
        return torch.rand(2, 3, 32, 32)  # Small test images

    @pytest.fixture
    def test_pil_image(self):
        """Test PIL image fixture."""
        return Image.new("RGB", (32, 32), color="green")

    def test_init_with_quality(self):
        """Test initialization with quality parameter."""
        compressor = WebPCompressor(quality=85)
        assert compressor.quality == 85
        assert compressor.method == 4
        assert not compressor.lossless

    def test_init_with_lossless(self):
        """Test initialization with lossless mode."""
        compressor = WebPCompressor(lossless=True)
        assert compressor.lossless

    def test_init_with_max_bits(self):
        """Test initialization with max_bits_per_image parameter."""
        compressor = WebPCompressor(max_bits_per_image=3500)
        assert compressor.max_bits_per_image == 3500
        assert compressor.quality is None

    def test_init_without_parameters(self):
        """Test that initialization fails without required parameters."""
        with pytest.raises(ValueError):
            WebPCompressor()

    def test_method_validation(self):
        """Test method parameter validation."""
        # Valid methods
        for method in range(7):
            WebPCompressor(quality=85, method=method)

        # Invalid method - too low
        with pytest.raises(ValueError):
            WebPCompressor(quality=85, method=-1)

        # Invalid method - too high
        with pytest.raises(ValueError):
            WebPCompressor(quality=85, method=7)

    def test_quality_validation(self):
        """Test quality validation."""
        compressor = WebPCompressor(quality=50)
        compressor._validate_quality(85)

        with pytest.raises(ValueError):
            compressor._validate_quality(0)

        with pytest.raises(ValueError):
            compressor._validate_quality(101)

    def test_quality_range(self):
        """Test quality range method."""
        compressor = WebPCompressor(quality=85)
        min_q, max_q = compressor._get_quality_range()
        assert min_q == 1
        assert max_q == 100

    def test_compress_decompress_single_image(self, test_pil_image):
        """Test compression and decompression of a single image."""
        compressor = WebPCompressor(quality=90)

        # Compress
        compressed_data, bits = compressor._compress_single_image(test_pil_image, 90)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        assert bits == len(compressed_data) * 8

        # Decompress
        recovered_image = compressor._decompress_single_image(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size

    def test_lossless_compression(self, test_pil_image):
        """Test lossless compression mode."""
        compressor = WebPCompressor(lossless=True)

        # Test that compress method works with lossless
        compressed_data = compressor.compress(test_pil_image)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

        # Decompress and verify
        recovered_image = compressor.decompress(compressed_data)
        assert recovered_image.size == test_pil_image.size

    def test_direct_compression_methods(self, test_pil_image):
        """Test direct compress/decompress methods."""
        compressor = WebPCompressor(quality=85)

        # Test compress method
        compressed_data = compressor.compress(test_pil_image)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

        # Test decompress method
        recovered_image = compressor.decompress(compressed_data)
        assert isinstance(recovered_image, Image.Image)
        assert recovered_image.size == test_pil_image.size
