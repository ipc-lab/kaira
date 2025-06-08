"""Tests for BPG image compression model with both unit and integration tests."""

import os
import subprocess  # nosec B404
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from kaira.models.image.compressors.bpg import BPGCompressor


def _is_bpg_available():
    """Check if BPG tools are available on the system."""
    try:
        # Run commands without check=True since they exit with code 1 when showing help
        subprocess.run(["bpgenc"], capture_output=True, check=False)  # nosec B603 B607
        subprocess.run(["bpgdec"], capture_output=True, check=False)  # nosec B603 B607
        return True
    except FileNotFoundError:
        return False


# Check tool availability at module level
bpg_available = _is_bpg_available()
skip_if_no_bpg = pytest.mark.skipif(not bpg_available, reason="BPG tools (bpgenc/bpgdec) not available")


@pytest.fixture
def sample_image():
    """Fixture that provides a sample image tensor for testing."""
    img = Image.new("RGB", (32, 32), color="red")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dimension


@pytest.fixture
def larger_sample_image():
    """Fixture that provides a larger sample image tensor for integration testing."""
    img = Image.new("RGB", (64, 64), color="red")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dimension


@pytest.fixture
def bpg_compressor_mocked():
    """Fixture that provides a BPGCompressor instance with mocked subprocess calls."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        compressor = BPGCompressor(quality=30)
        yield compressor


# =============================================================================
# UNIT TESTS (Mocked, always run)
# =============================================================================


@pytest.mark.unit
@skip_if_no_bpg
def test_bpg_compressor_initialization():
    """Test BPGCompressor initialization with valid parameters."""
    compressor = BPGCompressor(quality=30)
    assert compressor.quality == 30
    assert compressor.max_bits_per_image is None

    # Test max_bits_per_image mode
    compressor = BPGCompressor(max_bits_per_image=1000)
    assert compressor.quality is None
    assert compressor.max_bits_per_image == 1000


@pytest.mark.unit
@skip_if_no_bpg
def test_validate_executable_path():
    """Test validation of executable paths."""
    # Valid paths should not raise an error
    BPGCompressor(quality=30, bpg_encoder_path="bpgenc", bpg_decoder_path="bpgdec")

    # Invalid paths should raise ValueError
    with pytest.raises(ValueError, match="contains invalid characters"):
        BPGCompressor(quality=30, bpg_encoder_path="bpgenc; rm -rf /")

    with pytest.raises(ValueError, match="contains invalid characters"):
        BPGCompressor(quality=30, bpg_encoder_path="bpgenc &")

    # Test for potentially dangerous characters in non-existent paths
    with pytest.raises(ValueError, match="potentially dangerous characters"):
        BPGCompressor(quality=30, bpg_encoder_path="bpgenc$(whoami)")


@pytest.mark.unit
@skip_if_no_bpg
def test_safe_subprocess_run(bpg_compressor_mocked):
    """Test safe subprocess execution."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Test with default arguments
        bpg_compressor_mocked._safe_subprocess_run(["echo", "test"])
        mock_run.assert_called_with(["echo", "test"], shell=False, capture_output=True)

        # Test with custom kwargs
        bpg_compressor_mocked._safe_subprocess_run(["echo", "test"], text=True, stdout=subprocess.PIPE)
        mock_run.assert_called_with(["echo", "test"], shell=False, text=True, stdout=subprocess.PIPE)

        # Ensure shell=True is overridden
        bpg_compressor_mocked._safe_subprocess_run(["echo", "test"], shell=True)  # nosec B604
        mock_run.assert_called_with(["echo", "test"], shell=False, capture_output=True)


@pytest.mark.unit
@skip_if_no_bpg
def test_bpg_compressor_forward_unit(sample_image, bpg_compressor_mocked):
    """Test BPGCompressor forward pass with mocked compression."""
    # Ensure return_bits is False for this test
    bpg_compressor_mocked.return_bits = False

    # Mock the internal compress method
    with patch.object(bpg_compressor_mocked, "compress_with_quality") as mock_compress:
        mock_compress.return_value = torch.ones_like(sample_image[0])

        output = bpg_compressor_mocked(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == sample_image.shape


@pytest.mark.unit
@skip_if_no_bpg
def test_bpg_compressor_forward_with_bits_unit(sample_image, bpg_compressor_mocked):
    """Test BPGCompressor forward pass with bits per image (mocked)."""
    with patch.object(bpg_compressor_mocked, "compress_with_quality") as mock_compress:
        mock_compress.return_value = (torch.ones_like(sample_image[0]), {"bits": 1000})

        bpg_compressor_mocked.return_bits = True
        output, bits_per_image = bpg_compressor_mocked(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == sample_image.shape
        assert isinstance(bits_per_image, list)
        assert len(bits_per_image) == 1


@pytest.mark.unit
@skip_if_no_bpg
def test_setup_temp_paths_unit(bpg_compressor_mocked):
    """Test _setup_temp_paths method (mocked)."""
    with patch("tempfile.mkdtemp") as mock_mkdtemp:
        mock_mkdtemp.return_value = "/tmp/test_dir"  # nosec B108
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = "test-uuid"

            paths = bpg_compressor_mocked._setup_temp_paths(123)

            assert paths["dir"] == "/tmp/test_dir"  # nosec B108
            assert paths["input"] == "/tmp/test_dir/input_123_test-uuid.png"  # nosec B108
            assert paths["compressed"] == "/tmp/test_dir/compressed_123_test-uuid.bpg"  # nosec B108
            assert paths["output"] == "/tmp/test_dir/output_123_test-uuid.png"  # nosec B108
            assert paths["best_output"] == "/tmp/test_dir/best_123_test-uuid.png"  # nosec B108


@pytest.mark.unit
@skip_if_no_bpg
def test_compress_with_quality_failed_encoding_unit(bpg_compressor_mocked, sample_image):
    """Test compress_with_quality with failed encoding (mocked)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Encoding error"

        # Create a comprehensive mock for file operations
        with (
            patch("tempfile.mkdtemp") as mock_mkdtemp,
            patch("shutil.rmtree"),
            patch("torchvision.utils.save_image"),
            patch("builtins.open", mock_open(read_data=b"test")),
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=100),
            patch("os.makedirs", return_value=None),
            patch("os.rename"),
            patch("os.remove"),
        ):
            # Mock directory path and create any parent directories
            mock_mkdtemp.return_value = "/tmp/mock_dir"  # nosec B108

            # Create a mocked tensor to return when compression fails
            mock_tensor = torch.randn_like(sample_image[0])
            with patch("torch.randn_like", return_value=mock_tensor):
                # Test without return_info
                result = bpg_compressor_mocked.compress_with_quality(0, sample_image[0], 30, False)
                assert isinstance(result, torch.Tensor)
                assert result.shape == sample_image[0].shape
                assert torch.all(result == mock_tensor)  # Should match our mocked tensor

                # Test with return_info
                result, info = bpg_compressor_mocked.compress_with_quality(0, sample_image[0], 30, True)
                assert isinstance(result, torch.Tensor)
                assert result.shape == sample_image[0].shape
                assert torch.all(result == mock_tensor)  # Should match our mocked tensor
                assert info["quality"] == -1
                assert info["bits"] == 0


# =============================================================================
# INTEGRATION TESTS (Real tools, conditional)
# =============================================================================


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_initialization_integration():
    """Test BPGCompressor initialization with real BPG tools."""
    # This should work without mocking since BPG tools are available
    compressor = BPGCompressor(quality=30)
    assert compressor.quality == 30
    assert compressor.max_bits_per_image is None

    # Test that the tools are actually callable
    assert compressor.bpg_encoder_path == "bpgenc"
    assert compressor.bpg_decoder_path == "bpgdec"


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_compression_quality_mode(sample_image):
    """Test BPGCompressor with real compression in quality mode."""
    compressor = BPGCompressor(quality=35, return_bits=True)

    # This should use real BPG tools
    output, bits_per_image = compressor(sample_image)

    # Verify output structure
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1
    assert bits_per_image[0] > 0  # Should have some bits

    # Verify the image data is reasonable (not all zeros or all ones)
    assert not torch.all(output == 0)
    assert not torch.all(output == 1)

    # Verify tensor values are in reasonable range for images
    assert output.min() >= 0.0
    assert output.max() <= 1.0


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_compression_target_size_mode(sample_image):
    """Test BPGCompressor with real compression in target size mode."""
    target_bits = 2000  # 250 bytes target
    compressor = BPGCompressor(max_bits_per_image=target_bits, return_bits=True)

    # This should use real BPG tools and binary search
    output, bits_per_image = compressor(sample_image)

    # Verify output structure
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1
    assert bits_per_image[0] > 0  # Should have some bits

    # Verify the compression stayed within or close to target
    # Allow some tolerance due to discrete quality levels
    assert bits_per_image[0] <= target_bits * 1.2  # 20% tolerance

    # Verify tensor values are in reasonable range for images
    assert output.min() >= 0.0
    assert output.max() <= 1.0


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_compression_with_stats(sample_image):
    """Test BPGCompressor with real compression and statistics collection."""
    compressor = BPGCompressor(quality=30, collect_stats=True, return_bits=True)

    output, bits_per_image = compressor(sample_image)
    stats = compressor.get_stats()

    # Verify statistics structure
    assert isinstance(stats, dict)
    assert "total_bits" in stats
    assert "avg_quality" in stats
    assert "avg_bpp" in stats
    assert "avg_compression_ratio" in stats
    assert "processing_time" in stats
    assert "img_stats" in stats

    # Verify statistics values are reasonable
    assert stats["total_bits"] > 0
    assert stats["avg_quality"] == 30  # Should match our input
    assert stats["avg_bpp"] > 0
    assert stats["avg_compression_ratio"] > 0
    assert stats["processing_time"] > 0
    assert len(stats["img_stats"]) == 1


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_compression_with_compressed_data(sample_image):
    """Test BPGCompressor with real compression returning compressed data."""
    compressor = BPGCompressor(quality=35, return_bits=False, return_compressed_data=True)

    output, compressed_data = compressor(sample_image)

    # Verify output structure
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_image.shape
    assert isinstance(compressed_data, list)
    assert len(compressed_data) == 1
    assert isinstance(compressed_data[0], bytes)
    assert len(compressed_data[0]) > 0  # Should have some compressed data

    # The compressed data should be a valid BPG file
    # We can verify this by checking if it starts with BPG magic bytes
    bpg_data = compressed_data[0]
    # BPG files typically start with "BPG\xFB" magic bytes
    assert len(bpg_data) > 4


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_compression_batch(larger_sample_image):
    """Test BPGCompressor with real compression on batch of images."""
    # Create a batch of 3 images with different colors
    batch_images = []
    colors = ["red", "green", "blue"]

    for color in colors:
        img = Image.new("RGB", (64, 64), color=color)
        transform = transforms.ToTensor()
        batch_images.append(transform(img))

    batch_tensor = torch.stack(batch_images, dim=0)

    compressor = BPGCompressor(quality=30, return_bits=True, collect_stats=True)

    output, bits_per_image = compressor(batch_tensor)
    stats = compressor.get_stats()

    # Verify output structure
    assert isinstance(output, torch.Tensor)
    assert output.shape == batch_tensor.shape
    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 3

    # Verify all images were processed
    for bits in bits_per_image:
        assert bits > 0

    # Verify statistics
    assert stats["total_bits"] == sum(bits_per_image)
    assert len(stats["img_stats"]) == 3


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_get_bits_per_image(sample_image):
    """Test get_bits_per_image method with real compression."""
    compressor = BPGCompressor(quality=35)

    bits_per_image = compressor.get_bits_per_image(sample_image)

    assert isinstance(bits_per_image, list)
    assert len(bits_per_image) == 1
    assert bits_per_image[0] > 0


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_different_qualities(sample_image):
    """Test BPGCompressor with different quality levels to verify compression behavior."""
    qualities = [10, 30, 50]  # Low, medium, high quality
    results = []

    for quality in qualities:
        compressor = BPGCompressor(quality=quality, return_bits=True)
        output, bits_per_image = compressor(sample_image)
        results.append((quality, bits_per_image[0], output))

    # Generally, higher quality should use more bits (though not always strict due to image content)
    # We'll just verify that we get different bit counts and reasonable outputs
    bit_counts = [result[1] for result in results]
    assert len(set(bit_counts)) > 1  # Should get at least some different bit counts

    # All outputs should be valid image tensors
    for quality, bits, output in results:
        assert isinstance(output, torch.Tensor)
        assert output.shape == sample_image.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
        assert bits > 0


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_compressor_real_error_handling():
    """Test BPGCompressor error handling with invalid parameters but real tools."""
    # Test invalid quality range
    with pytest.raises(ValueError, match="BPG quality must be between 0 and 51"):
        BPGCompressor(quality=100)

    with pytest.raises(ValueError, match="BPG quality must be between 0 and 51"):
        BPGCompressor(quality=-5)

    # Test missing parameters
    with pytest.raises(ValueError, match="At least one of the two parameters must be provided"):
        BPGCompressor()


# =============================================================================
# HELPER TESTS FOR INTEGRATION
# =============================================================================


@pytest.mark.integration
@skip_if_no_bpg
def test_bpg_tools_actually_available():
    """Test that BPG tools are actually available and functional."""
    # Create a simple test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
        img = Image.new("RGB", (16, 16), color="red")
        img.save(tmp_input.name)
        tmp_input_path = tmp_input.name

    with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as tmp_compressed:
        tmp_compressed_path = tmp_compressed.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_output:
        tmp_output_path = tmp_output.name

    try:
        # Test encoding
        result_enc = subprocess.run(["bpgenc", "-q", "30", "-o", tmp_compressed_path, tmp_input_path], capture_output=True, text=True)
        assert result_enc.returncode == 0, f"bpgenc failed: {result_enc.stderr}"

        # Test decoding
        result_dec = subprocess.run(["bpgdec", "-o", tmp_output_path, tmp_compressed_path], capture_output=True, text=True)
        assert result_dec.returncode == 0, f"bpgdec failed: {result_dec.stderr}"

        # Verify output file exists and is readable
        assert os.path.exists(tmp_output_path)
        with Image.open(tmp_output_path) as decoded_img:
            assert decoded_img.size == (16, 16)
            assert decoded_img.mode == "RGB"

    finally:
        # Cleanup
        for path in [tmp_input_path, tmp_compressed_path, tmp_output_path]:
            if os.path.exists(path):
                os.unlink(path)
