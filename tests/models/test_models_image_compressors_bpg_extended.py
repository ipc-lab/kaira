import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch
from PIL import Image
from torchvision import transforms

from kaira.models.image.compressors.bpg import BPGCompressor


@pytest.fixture
def sample_image():
    """Fixture that provides a sample image tensor for testing."""
    img = Image.new("RGB", (32, 32), color="red")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dimension


@pytest.fixture
def bpg_compressor():
    """Fixture for a BPGCompressor with mocked subprocess calls."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        compressor = BPGCompressor(quality=30)
        yield compressor


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


def test_safe_subprocess_run(bpg_compressor):
    """Test safe subprocess execution."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # Test with default arguments
        bpg_compressor._safe_subprocess_run(["echo", "test"])
        mock_run.assert_called_with(["echo", "test"], shell=False, capture_output=True)

        # Test with custom kwargs
        bpg_compressor._safe_subprocess_run(["echo", "test"], text=True, stdout=subprocess.PIPE)
        mock_run.assert_called_with(["echo", "test"], shell=False, text=True, stdout=subprocess.PIPE)

        # Ensure shell=True is overridden
        bpg_compressor._safe_subprocess_run(["echo", "test"], shell=True)
        mock_run.assert_called_with(["echo", "test"], shell=False, capture_output=True)


def test_bpg_compressor_forward_exception_handling(sample_image):
    """Test BPGCompressor forward pass with subprocess exceptions."""
    with patch("subprocess.run") as mock_run:
        # Set up mock to raise an exception
        mock_run.side_effect = subprocess.SubprocessError("Command failed")

        with pytest.raises(RuntimeError):
            BPGCompressor(quality=30)


def test_parallel_forward_bpg(bpg_compressor):
    """Test parallel_forward_bpg method with quality mode."""
    with patch.object(bpg_compressor, "compress_with_quality") as mock_compress:
        mock_compress.return_value = torch.ones(3, 32, 32)

        img = torch.ones(3, 32, 32)
        result = bpg_compressor.parallel_forward_bpg(0, img, False)

        mock_compress.assert_called_once_with(0, img, 30, False)
        assert torch.all(result == torch.ones(3, 32, 32))


def test_parallel_forward_bpg_target_size(sample_image):
    """Test parallel_forward_bpg method with target size mode."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        compressor = BPGCompressor(max_bits_per_image=1000)

        with patch.object(compressor, "compress_with_target_size") as mock_compress:
            mock_compress.return_value = torch.ones(3, 32, 32)

            img = torch.ones(3, 32, 32)
            result = compressor.parallel_forward_bpg(0, img, False)

            mock_compress.assert_called_once_with(0, img, 1000, False)
            assert torch.all(result == torch.ones(3, 32, 32))


def test_setup_temp_paths(bpg_compressor):
    """Test _setup_temp_paths method."""
    with patch("tempfile.mkdtemp") as mock_mkdtemp:
        mock_mkdtemp.return_value = "/tmp/test_dir"
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value = "test-uuid"

            paths = bpg_compressor._setup_temp_paths(123)

            assert paths["dir"] == "/tmp/test_dir"
            assert paths["input"] == "/tmp/test_dir/input_123_test-uuid.png"
            assert paths["compressed"] == "/tmp/test_dir/compressed_123_test-uuid.bpg"
            assert paths["output"] == "/tmp/test_dir/output_123_test-uuid.png"
            assert paths["best_output"] == "/tmp/test_dir/best_123_test-uuid.png"


def test_compress_with_quality_failed_encoding(bpg_compressor, sample_image):
    """Test compress_with_quality with failed encoding."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Encoding error"

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"):
            # Test without return_info
            result = bpg_compressor.compress_with_quality(0, sample_image[0], 30, False)
            assert isinstance(result, torch.Tensor)
            assert result.shape == sample_image[0].shape

            # Test with return_info
            result, info = bpg_compressor.compress_with_quality(0, sample_image[0], 30, True)
            assert isinstance(result, torch.Tensor)
            assert result.shape == sample_image[0].shape
            assert info["quality"] == -1
            assert info["bits"] == 0


def test_compress_with_quality_failed_decoding(bpg_compressor, sample_image):
    """Test compress_with_quality with failed decoding."""
    with patch("subprocess.run") as mock_run, patch("os.path.getsize") as mock_getsize:
        # First call (encoding) succeeds
        mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=1, stderr="Decoding error")]  # Encoding successful  # Decoding fails
        mock_getsize.return_value = 100  # 100 bytes

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"):
            with patch("builtins.open", mock_open(read_data=b"test")):
                # Test with return_info and compressed data
                bpg_compressor.return_compressed_data = True
                result, info = bpg_compressor.compress_with_quality(0, sample_image[0], 30, True)

                assert isinstance(result, torch.Tensor)
                assert info["quality"] == -1
                assert info["bits"] == 0


def test_compress_with_quality_success(bpg_compressor, sample_image):
    """Test compress_with_quality with successful compression."""
    with patch("subprocess.run") as mock_run, patch("os.path.getsize") as mock_getsize:
        # Both encoding and decoding succeed
        mock_run.return_value.returncode = 0
        mock_getsize.return_value = 100  # 100 bytes

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"):
            with patch("PIL.Image.open") as mock_open_image, patch("builtins.open", mock_open(read_data=b"test")):
                mock_img = MagicMock()
                mock_img.convert.return_value = mock_img
                mock_open_image.return_value = mock_img

                with patch("torchvision.transforms.ToTensor") as mock_to_tensor:
                    mock_to_tensor.return_value.return_value = torch.ones(3, 32, 32)

                    # Test with return_info and compressed data
                    bpg_compressor.return_compressed_data = True
                    result, info = bpg_compressor.compress_with_quality(0, sample_image[0], 30, True)

                    assert torch.all(result == torch.ones(3, 32, 32))
                    assert info["quality"] == 30
                    assert info["bits"] == 800  # 100 bytes * 8 bits
                    assert "compressed_data" in info
                    assert info["compressed_data"] == b"test"

                    # Test without return_info
                    result = bpg_compressor.compress_with_quality(0, sample_image[0], 30, False)
                    assert torch.all(result == torch.ones(3, 32, 32))


def test_compress_with_target_size_initial_estimate(bpg_compressor, sample_image):
    """Test compress_with_target_size initial quality estimate."""
    with patch("subprocess.run") as mock_run, patch("os.path.getsize") as mock_getsize:
        # Initial estimate succeeds
        mock_run.return_value.returncode = 0

        # Set up size return values for different qualities
        def getsize_side_effect(path):
            if "compressed" in path:
                # Return 50 bytes (400 bits) for initial quality
                return 50
            return 1000  # Original size

        mock_getsize.side_effect = getsize_side_effect

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"), patch("os.remove"), patch("os.rename"), patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False  # No best output exists yet

            with patch("PIL.Image.open") as mock_open_image:
                mock_img = MagicMock()
                mock_img.convert.return_value = mock_img
                mock_open_image.return_value = mock_img

                with patch("torchvision.transforms.ToTensor") as mock_to_tensor:
                    mock_to_tensor.return_value.return_value = torch.ones(3, 32, 32)

                    # Test with target bits above initial estimate (400 bits) - should search higher qualities
                    result = bpg_compressor.compress_with_target_size(0, sample_image[0], 500, False)

                    # Verify that binary search starts from 30 to 51 (higher quality)
                    assert mock_run.call_count > 1
                    assert torch.all(result == torch.ones(3, 32, 32))


def test_compress_with_target_size_binary_search(bpg_compressor, sample_image):
    """Test compress_with_target_size binary search algorithm."""
    with patch("subprocess.run") as mock_run, patch("os.path.getsize") as mock_getsize:
        # All commands succeed
        mock_run.return_value.returncode = 0

        # Set up size return values for different qualities to simulate binary search
        quality_sizes = {
            # Format: quality: size in bytes
            30: 75,  # 600 bits - above target
            15: 50,  # 400 bits - below target
            22: 60,  # 480 bits - below target
            26: 70,  # 560 bits - above target
            24: 62,  # 496 bits - below target
            25: 65,  # 520 bits - above target
        }

        call_count = 0

        def getsize_side_effect(path):
            nonlocal call_count
            if "compressed" in path:
                # Return size based on the current quality being tested
                # Extract quality from the most recent command
                cmd_args = mock_run.call_args_list[-1][0][0]
                quality_idx = cmd_args.index("-q") + 1
                quality = int(cmd_args[quality_idx])
                return quality_sizes.get(quality, 100)
            return 1000  # Original size

        mock_getsize.side_effect = getsize_side_effect

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"), patch("os.remove"), patch("os.rename"), patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True  # Best output exists after first successful attempt

            with patch("PIL.Image.open") as mock_open_image:
                mock_img = MagicMock()
                mock_img.convert.return_value = mock_img
                mock_open_image.return_value = mock_img

                with patch("torchvision.transforms.ToTensor") as mock_to_tensor:
                    mock_to_tensor.return_value.return_value = torch.ones(3, 32, 32)

                    # Test with target of 500 bits
                    with patch("builtins.open", mock_open(read_data=b"test")):
                        bpg_compressor.return_compressed_data = True
                        result, info = bpg_compressor.compress_with_target_size(0, sample_image[0], 500, True)

                        # Best quality should be 24 (496 bits, just under target)
                        assert info["quality"] == 24
                        assert info["bits"] == 496
                        assert "compressed_data" in info
                        assert torch.all(result == torch.ones(3, 32, 32))


def test_compress_with_target_size_failed_search(bpg_compressor, sample_image):
    """Test compress_with_target_size when no quality meets the target."""
    with patch("subprocess.run") as mock_run, patch("os.path.getsize"):
        # Initial estimate fails
        mock_run.return_value.returncode = 1

        with patch("tempfile.mkdtemp"), patch("shutil.rmtree"), patch("torchvision.utils.save_image"):
            # Test with return_info
            bpg_compressor.return_compressed_data = True
            result, info = bpg_compressor.compress_with_target_size(0, sample_image[0], 100, True)

            assert isinstance(result, torch.Tensor)
            assert info["quality"] == -1
            assert info["bits"] == 0
            assert info["target_bits"] == 100
            assert "compressed_data" in info
            assert info["compressed_data"] == b""

            # Test without return_info
            result = bpg_compressor.compress_with_target_size(0, sample_image[0], 100, False)
            assert isinstance(result, torch.Tensor)


def test_bpg_compressor_get_stats_when_not_collected():
    """Test get_stats when stats were not collected."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        compressor = BPGCompressor(quality=30, collect_stats=False)

        with patch("logging.Logger.warning") as mock_warning:
            stats = compressor.get_stats()

            mock_warning.assert_called_once()
            assert stats == {}
