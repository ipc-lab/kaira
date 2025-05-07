"""BPG (Better Portable Graphics) image compressor wrapper."""

import logging
import multiprocessing
import os
import re
import shutil
import subprocess  # nosec
import tempfile
import time
import uuid

# Change CompletedProcess import location if needed, or just use subprocess.CompletedProcess
from subprocess import CompletedProcess  # nosec B404
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from joblib import Parallel, delayed
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from kaira.models.base import BaseModel

logger = logging.getLogger(__name__)


class BPGCompressor(BaseModel):
    """BPG (Better Portable Graphics) image compression based on bpgenc and bpgdec.

    This class provides BPG-based compression. It can operate in two modes:
    1. Fixed quality mode: directly uses the specified quality level
    2. Bit-constrained mode: finds the highest quality that stays under a bit budget
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        bpg_encoder_path: str = "bpgenc",
        bpg_decoder_path: str = "bpgdec",
        n_jobs: Optional[int] = None,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the BPG Compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: Fixed quality level for BPG compression (0-51, lower is better).
                    If provided, this exact quality will be used regardless of resulting file size.
            bpg_encoder_path: Path to the BPG encoder executable
            bpg_decoder_path: Path to the BPG decoder executable
            n_jobs: Number of parallel jobs to use (default: CPU count // 2)
            collect_stats: Whether to collect and return compression statistics
            return_bits: Whether to return bits per image in forward pass
            return_compressed_data: Whether to return the compressed binary data
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)  # Pass args and kwargs to base

        # At least one of the two parameters must be provided
        if max_bits_per_image is None and quality is None:
            raise ValueError("At least one of the two parameters must be provided")

        if quality is not None and (quality < 0 or quality > 51):
            raise ValueError("BPG quality must be between 0 and 51")

        self.max_bits_per_image = max_bits_per_image
        self.quality = quality

        # Validate executable paths to prevent command injection
        self._validate_executable_path(bpg_encoder_path)
        self._validate_executable_path(bpg_decoder_path)

        self.bpg_encoder_path = bpg_encoder_path
        self.bpg_decoder_path = bpg_decoder_path
        self.n_jobs = n_jobs if n_jobs is not None else max(1, multiprocessing.cpu_count() // 2)
        self.collect_stats = collect_stats
        self.return_bits = return_bits
        self.return_compressed_data = return_compressed_data
        self.stats: Dict[str, Any] = {}

        # Check if BPG tools are available using secure subprocess execution
        try:
            self._safe_subprocess_run([self.bpg_encoder_path, "--help"])
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error(f"BPG encoder not found at '{self.bpg_encoder_path}'. Please install BPG tools.")
            raise RuntimeError(f"BPG encoder not found at '{self.bpg_encoder_path}'")

    def _validate_executable_path(self, path: str) -> None:
        """Validate that an executable path doesn't contain shell metacharacters.

        Args:
            path: The executable path to validate

        Raises:
            ValueError: If the path contains potentially dangerous characters
        """
        # Simple validation to prevent basic command injection
        if ";" in path or "&" in path or "|" in path or ">" in path or "<" in path:
            raise ValueError(f"Executable path '{path}' contains invalid characters")

        # Check if path doesn't exist but contains shell metacharacters
        if not os.path.exists(path) and re.search(r"[${}()`\[\]\s]", path):
            raise ValueError(f"Executable path '{path}' contains potentially dangerous characters")

    def _safe_subprocess_run(self, cmd_args: List[str], **kwargs) -> CompletedProcess:
        """Execute subprocess safely with validated arguments.

        Args:
            cmd_args: Command arguments list
            **kwargs: Additional arguments for subprocess.run

        Returns:
            subprocess.CompletedProcess object
        """
        # Always enforce shell=False
        kwargs["shell"] = False

        # Default to capturing output
        if "capture_output" not in kwargs and "stdout" not in kwargs:
            kwargs["capture_output"] = True

        # Ensure return type matches annotation
        return subprocess.run(cmd_args, **kwargs)  # type: ignore # nosec B603

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]], Tuple[torch.Tensor, List[bytes]], Tuple[torch.Tensor, List[int], List[bytes]]]:
        """Process a batch of images through BPG compression.

        The compression method depends on initialization parameters:
        - If quality was provided, that fixed quality is used
        - If max_bits_per_image was provided, the highest quality meeting the bit constraint is found

        Args:
            x: Tensor of shape [batch_size, channels, height, width]
            *args: Additional positional arguments (passed to internal methods).
            **kwargs: Additional keyword arguments (passed to internal methods).

        Returns:
            If no additional returns: Just the reconstructed image tensor
            If return_bits=True: Tuple of (tensor, bits per image)
            If return_compressed_data=True: Tuple of (tensor, compressed binary data)
            If both are True: Tuple of (tensor, bits per image, compressed binary data)
        """
        start_time = time.time()

        if self.collect_stats:
            self.stats = {"total_bits": 0, "avg_quality": 0, "img_stats": []}

        # Always collect bits information if return_bits or collect_stats is True
        collect_info = self.return_bits or self.collect_stats or self.return_compressed_data

        # Process images in parallel
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.parallel_forward_bpg)(i, x[i], collect_info, *args, **kwargs) for i in range(x.shape[0]))

        # Unpack results
        images = []
        bits_per_image: List[int] = [] if self.return_bits or self.collect_stats else []
        compressed_data: List[bytes] = [] if self.return_compressed_data else []

        for result in results:
            if collect_info:
                img, info = result
                images.append(img)

                if (self.return_bits or self.collect_stats) and bits_per_image is not None:
                    bits_per_image.append(int(info.get("bits", 0)))

                if self.return_compressed_data and compressed_data is not None:
                    compressed_data.append(info.get("compressed_data", b""))

                # Update full stats if requested
                if self.collect_stats:
                    self.stats["total_bits"] += info.get("bits", 0)
                    self.stats["img_stats"].append(info)
            else:
                images.append(result)

        x_hat = torch.stack(images, dim=0).to(x.device)

        # Calculate aggregate statistics if requested
        if self.collect_stats and x.shape[0] > 0:
            self.stats["avg_quality"] = sum(s.get("quality", 0) for s in self.stats["img_stats"]) / x.shape[0]
            self.stats["avg_bpp"] = self.stats["total_bits"] / x.shape[0]
            self.stats["avg_compression_ratio"] = sum(s.get("compression_ratio", 0) for s in self.stats["img_stats"]) / x.shape[0]

        self.stats["processing_time"] = time.time() - start_time

        # Return appropriate output based on flags
        if self.return_bits and self.return_compressed_data:
            return x_hat, bits_per_image, compressed_data
        elif self.return_bits:
            return x_hat, bits_per_image
        elif self.return_compressed_data:
            return x_hat, compressed_data
        else:
            return x_hat

    def parallel_forward_bpg(self, idx: int, img: torch.Tensor, return_info: bool = False, *args: Any, **kwargs: Any):
        """Process a single image with BPG compression.

        Args:
            idx: Image index
            img: Image tensor of shape [channels, height, width]
            return_info: Whether to return compression information
            *args: Additional positional arguments (passed to compression methods).
            **kwargs: Additional keyword arguments (passed to compression methods).

        Returns:
            If return_info=False: Processed image tensor
            If return_info=True: Tuple of (tensor, info_dict)
        """
        if self.quality is not None:
            # Pass *args, **kwargs
            result = self.compress_with_quality(idx, img, self.quality, return_info, *args, **kwargs)
        else:
            # Ensure max_bits_per_image is not None before calling compress_with_target_size
            assert self.max_bits_per_image is not None, "max_bits_per_image must be set if quality is not provided"
            # Pass *args, **kwargs
            result = self.compress_with_target_size(idx, img, self.max_bits_per_image, return_info, *args, **kwargs)

        return result

    def _setup_temp_paths(self, idx):
        """Create temporary directory and generate file paths."""
        temp_dir = tempfile.mkdtemp(prefix="bpg_")
        uid = f"{idx}_{uuid.uuid4()}"

        paths = {"dir": temp_dir, "input": os.path.join(temp_dir, f"input_{uid}.png"), "compressed": os.path.join(temp_dir, f"compressed_{uid}.bpg"), "output": os.path.join(temp_dir, f"output_{uid}.png"), "best_output": os.path.join(temp_dir, f"best_{uid}.png")}

        return paths

    def compress_with_quality(self, idx: int, x: torch.Tensor, quality: int, return_info: bool = False, *args: Any, **kwargs: Any):
        """Compress image with a specific quality level.

        Args:
            idx: Image index for generating unique filenames
            x: Image tensor
            quality: BPG quality level (0-51)
            return_info: Whether to return compression information
            *args: Additional positional arguments (unused in this method).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            If return_info=False: Compressed-decompressed image tensor
            If return_info=True: Tuple of (tensor, info_dict)
        """
        paths = self._setup_temp_paths(idx)

        # Save input image
        save_image(x, paths["input"])

        # Measure original file size
        original_size = os.path.getsize(paths["input"])

        # Compress with specified quality using safe subprocess execution
        result_enc = self._safe_subprocess_run([self.bpg_encoder_path, "-q", str(quality), "-o", paths["compressed"], paths["input"]], text=True)

        if result_enc.returncode != 0:
            logger.error(f"BPG encoding failed: {result_enc.stderr}")
            shutil.rmtree(paths["dir"])
            # Return directly, don't reassign result_enc
            return (torch.randn_like(x), {"quality": -1, "bits": 0}) if return_info else torch.randn_like(x)

        # Get compressed size
        compressed_size = os.path.getsize(paths["compressed"])
        bits = compressed_size * 8

        # Read compressed data if needed
        compressed_data = None
        if self.return_compressed_data and return_info:
            with open(paths["compressed"], "rb") as f:
                compressed_data = f.read()

        # Decompress using safe subprocess execution
        result_dec = self._safe_subprocess_run([self.bpg_decoder_path, "-o", paths["output"], paths["compressed"]], text=True)

        if result_dec.returncode != 0:
            logger.error(f"BPG decoding failed: {result_dec.stderr}")
            shutil.rmtree(paths["dir"])
            # Return directly, don't reassign result_dec
            return (torch.randn_like(x), {"quality": -1, "bits": 0}) if return_info else torch.randn_like(x)

        # Load result
        transform = transforms.ToTensor()
        img = transform(Image.open(paths["output"]).convert("RGB"))

        # Prepare result
        if return_info:
            stats = {"quality": quality, "bits": bits, "bpp": bits / (x.shape[1] * x.shape[2]), "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0}
            if compressed_data is not None:
                stats["compressed_data"] = compressed_data

            # Assign to final_result instead of result
            final_result = (img, stats)
        else:
            # Assign to final_result instead of result
            final_result = img

        # Cleanup
        shutil.rmtree(paths["dir"])

        # Return the final_result
        return final_result

    # Change target_bits type hint from Optional[int] to int
    def compress_with_target_size(self, idx: int, x: torch.Tensor, target_bits: int, return_info: bool = False, *args: Any, **kwargs: Any):
        """Find highest quality that produces file size below target_bits using binary search.

        Args:
            idx: Image index for generating unique filenames
            x: Image tensor
            target_bits: Maximum bits for the compressed image
            return_info: Whether to return compression information
            *args: Additional positional arguments (unused in this method).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            If return_info=False: Compressed-decompressed image tensor
            If return_info=True: Tuple of (tensor, info_dict)
        """
        paths = self._setup_temp_paths(idx)

        # Save input image
        save_image(x, paths["input"])
        original_size = os.path.getsize(paths["input"])
        transform = transforms.ToTensor()

        # Perform initial quality estimates using safe subprocess execution
        initial_quality = 30
        result_init = self._safe_subprocess_run([self.bpg_encoder_path, "-q", str(initial_quality), "-o", paths["compressed"], paths["input"]], text=True)

        if result_init.returncode == 0:
            bits_at_q30 = os.path.getsize(paths["compressed"]) * 8

            # Check against target_bits (now guaranteed to be int)
            if bits_at_q30 <= target_bits:
                # Quality can be higher, start from here
                low, high = initial_quality, 51
            else:
                # Need lower quality
                low, high = 0, initial_quality - 1

            # Clean up the test file
            os.remove(paths["compressed"])
        else:
            # Fallback to full range if initial test fails
            low, high = 0, 51

        # Binary search for the highest quality that meets the target bit size
        best_quality = -1
        # Initialize best_bits as float
        best_bits: float = 0.0

        while low <= high:
            mid = (low + high) // 2

            # Try compression with the current quality using safe subprocess execution
            result_bs = self._safe_subprocess_run([self.bpg_encoder_path, "-q", str(mid), "-o", paths["compressed"], paths["input"]], text=True)
            if result_bs.returncode != 0:
                logger.error(f"BPG encoding failed at quality {mid}: {result_bs.stderr}")
                high = mid - 1
                continue

            # Check file size
            bytes_out = os.path.getsize(paths["compressed"])
            bitrate_out = float(bytes_out) * 8

            # Check against target_bits (now guaranteed to be int)
            if bitrate_out <= target_bits:
                # This quality works - save it and try higher quality
                best_quality = mid
                # Assign float directly
                best_bits = bitrate_out

                # Decode the image using safe subprocess execution
                result_dec_bs = self._safe_subprocess_run([self.bpg_decoder_path, "-o", paths["output"], paths["compressed"]], text=True)
                if result_dec_bs.returncode == 0:
                    # Save this as our best result so far
                    if os.path.exists(paths["best_output"]):
                        os.remove(paths["best_output"])
                    os.rename(paths["output"], paths["best_output"])

                # Try higher quality
                low = mid + 1
            else:
                # Quality too high, try lower
                high = mid - 1

            # Clean up compressed file
            if os.path.exists(paths["compressed"]):
                os.remove(paths["compressed"])

        # Load the best image we found
        if best_quality != -1:
            img = transform(Image.open(paths["best_output"]).convert("RGB"))

            if return_info:
                # Read compressed data if requested
                compressed_data = None
                if self.return_compressed_data:
                    # We need to re-compress at the best quality to get the data
                    temp_compressed = os.path.join(paths["dir"], f"final_{uuid.uuid4()}.bpg")
                    result_final = self._safe_subprocess_run([self.bpg_encoder_path, "-q", str(best_quality), "-o", temp_compressed, paths["input"]], text=True)
                    if result_final.returncode == 0:
                        with open(temp_compressed, "rb") as f:
                            compressed_data = f.read()
                        os.remove(temp_compressed)

                # Cast best_bits to int for stats dict if needed, or keep as float
                stats = {"quality": best_quality, "bits": int(best_bits), "bpp": best_bits / (x.shape[1] * x.shape[2]), "compression_ratio": original_size / (best_bits / 8) if best_bits > 0 else 0, "target_bits": target_bits}

                if compressed_data is not None:
                    stats["compressed_data"] = compressed_data

                # Assign to final_result instead of result
                final_result = (img, stats)
            else:
                # Assign to final_result instead of result
                final_result = img
        else:
            logger.warning(f"Could not find any quality level meeting target of {target_bits} bits")
            if return_info:
                stats = {"quality": -1, "bits": 0, "target_bits": target_bits}
                if self.return_compressed_data:
                    stats["compressed_data"] = b""
                # Assign to final_result instead of result
                final_result = (torch.randn_like(x), stats)
            else:
                # Assign to final_result instead of result
                final_result = torch.randn_like(x)

        # Cleanup
        shutil.rmtree(paths["dir"])
        # Return the final_result
        return final_result

    def get_stats(self):
        """Return compression statistics if collect_stats=True was set."""
        if not self.collect_stats:
            logger.warning("Statistics not collected. Initialize with collect_stats=True to enable.")
            return {}
        return self.stats

    # Update method signature to align with class variable
    def get_bits_per_image(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> List[int]:
        """Compress images and return only the bit counts per image.

        The compression method depends on whether quality or max_bits_per_image was provided
        during initialization.

        Args:
            x: Tensor of shape [batch_size, channels, height, width]
            *args: Additional positional arguments passed to forward.
            **kwargs: Additional keyword arguments passed to forward.

        Returns:
            List[int]: Number of bits used for each compressed image
        """
        # Temporarily override return_bits setting
        original_return_bits = self.return_bits
        self.return_bits = True

        try:
            # Pass *args, **kwargs to forward
            forward_output = self.forward(x, *args, **kwargs)
            # Ensure forward returned the expected tuple when return_bits is True
            if isinstance(forward_output, tuple) and len(forward_output) >= 2:
                bits_per_image = forward_output[1]
                if not isinstance(bits_per_image, list):
                    raise TypeError(f"Expected list of bits, but got {type(bits_per_image)}")
            else:
                raise TypeError(f"Forward method did not return expected tuple (tensor, bits), got {type(forward_output)}")
        finally:
            # Restore original setting
            self.return_bits = original_return_bits

        return bits_per_image
