"""Base class for image compressors."""

import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from kaira.models.base import BaseModel


class BaseImageCompressor(BaseModel):
    """Abstract base class for image compression methods.

    This class provides a consistent interface for all image compression implementations in Kaira,
    including traditional methods (JPEG, PNG), modern standards (BPG), and neural network-based
    approaches.

    All compressors support both quality-based and bit-constrained compression modes, batch
    processing capabilities, and optional compression statistics collection.
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[Union[int, float]] = None,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the image compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: Quality level for compression. Range and interpretation depend on the
                    specific compressor implementation.
            collect_stats: Whether to collect and return compression statistics
            return_bits: Whether to return bits per image in forward pass
            return_compressed_data: Whether to return the compressed binary data
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # At least one of the two parameters must be provided
        if max_bits_per_image is None and quality is None:
            raise ValueError("At least one of max_bits_per_image or quality must be provided")

        self.max_bits_per_image = max_bits_per_image
        self.quality = quality
        self.collect_stats = collect_stats
        self.return_bits = return_bits
        self.return_compressed_data = return_compressed_data
        self.stats: Dict[str, Any] = {}

        # Validate quality range if provided
        if quality is not None:
            self._validate_quality(quality)

    @abstractmethod
    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the quality parameter is within the acceptable range.

        Args:
            quality: Quality level to validate

        Raises:
            ValueError: If quality is outside the acceptable range
        """
        pass

    @abstractmethod
    def _get_quality_range(self) -> Tuple[Union[int, float], Union[int, float]]:
        """Get the valid quality range for this compressor.

        Returns:
            Tuple of (min_quality, max_quality)
        """
        pass

    @abstractmethod
    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image.

        Args:
            image: PIL Image to compress
            quality: Quality level for compression
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)
        """
        pass

    @abstractmethod
    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress bytes back to a PIL Image.

        Args:
            data: Compressed image data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        pass

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a single image tensor to PIL Image.

        Args:
            tensor: Image tensor of shape [C, H, W] with values in [0, 1]

        Returns:
            PIL Image in RGB mode
        """
        # Clamp values to [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)

        # Convert to [0, 255] range and uint8
        tensor = (tensor * 255).byte()

        # Convert from [C, H, W] to [H, W, C]
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        # Convert to numpy and create PIL Image
        array = tensor.cpu().numpy()

        if array.shape[2] == 1:
            # Grayscale
            array = array.squeeze(2)
            return Image.fromarray(array, mode="L")
        elif array.shape[2] == 3:
            # RGB
            return Image.fromarray(array, mode="RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {array.shape[2]}")

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor.

        Args:
            image: PIL Image

        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """
        # Convert to RGB if not already
        if image.mode != "RGB":
            if image.mode == "L":
                # Grayscale to RGB
                image = image.convert("RGB")
            else:
                image = image.convert("RGB")

        # Convert to tensor
        import torchvision.transforms.functional as F

        tensor = F.to_tensor(image)
        return tensor

    def _find_optimal_quality(self, image: Image.Image, max_bits: int, **kwargs: Any) -> Tuple[Union[int, float], bytes, int]:
        """Find the highest quality that produces a file size under the bit limit.

        Args:
            image: PIL Image to compress
            max_bits: Maximum allowed bits
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (optimal_quality, compressed_data, actual_bits)
        """
        min_quality, max_quality = self._get_quality_range()

        # Try minimum quality first as fallback
        try:
            fallback_data, fallback_bits = self._compress_single_image(image, min_quality, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to compress image even at minimum quality {min_quality}: {e}")

        # If even minimum quality exceeds the limit, use it anyway
        if fallback_bits > max_bits:
            return min_quality, fallback_data, fallback_bits

        # Binary search for optimal quality
        best_quality = min_quality
        best_data = fallback_data
        best_bits = fallback_bits

        low, high = min_quality, max_quality

        while low <= high:
            mid_quality = (low + high) // 2 if isinstance(low, int) else (low + high) / 2

            try:
                compressed_data, bits = self._compress_single_image(image, mid_quality, **kwargs)

                if bits <= max_bits:
                    # Can use higher quality
                    best_quality = mid_quality
                    best_data = compressed_data
                    best_bits = bits
                    low = mid_quality + (1 if isinstance(low, int) else 0.1)
                else:
                    # Need to use lower quality
                    high = mid_quality - (1 if isinstance(high, int) else 0.1)

            except Exception:
                # If compression fails at this quality, try lower
                high = mid_quality - (1 if isinstance(high, int) else 0.1)

        return best_quality, best_data, best_bits

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]], Tuple[torch.Tensor, List[bytes]], Tuple[torch.Tensor, List[int], List[bytes]]]:
        """Process a batch of images through compression.

        Args:
            x: Tensor of shape [batch_size, channels, height, width] with values in [0, 1]
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            If no additional returns: Just the reconstructed image tensor
            If return_bits=True: Tuple of (tensor, bits per image)
            If return_compressed_data=True: Tuple of (tensor, compressed binary data)
            If both are True: Tuple of (tensor, bits per image, compressed binary data)
        """
        start_time = time.time()

        if self.collect_stats:
            self.stats = {"total_bits": 0, "avg_quality": 0, "img_stats": []}

        batch_size = x.shape[0]
        reconstructed_images = []
        bits_per_image: List[int] = [] if self.return_bits or self.collect_stats else []
        compressed_data: List[bytes] = [] if self.return_compressed_data else []

        total_bits = 0
        total_quality: float = 0.0

        for i in range(batch_size):
            # Convert tensor to PIL Image
            pil_image = self._tensor_to_pil(x[i])

            if self.quality is not None:
                # Fixed quality mode
                comp_data, bits = self._compress_single_image(pil_image, self.quality, **kwargs)
                used_quality = self.quality
            else:
                # Bit-constrained mode
                if self.max_bits_per_image is None:
                    raise ValueError("max_bits_per_image must be set for bit-constrained mode")
                used_quality, comp_data, bits = self._find_optimal_quality(pil_image, self.max_bits_per_image, **kwargs)

            # Decompress back to PIL Image
            reconstructed_pil = self._decompress_single_image(comp_data, **kwargs)

            # Convert back to tensor
            reconstructed_tensor = self._pil_to_tensor(reconstructed_pil)
            reconstructed_images.append(reconstructed_tensor)

            # Collect statistics
            if self.return_bits or self.collect_stats:
                bits_per_image.append(bits)
                total_bits += bits

            if self.return_compressed_data:
                compressed_data.append(comp_data)

            if self.collect_stats:
                total_quality += used_quality
                self.stats["img_stats"].append({"quality": used_quality, "bits": bits, "compression_ratio": (pil_image.width * pil_image.height * 24) / bits})  # Assuming RGB

        # Update statistics
        if self.collect_stats:
            self.stats.update({"total_bits": total_bits, "avg_quality": total_quality / batch_size, "total_time": time.time() - start_time, "avg_bits_per_image": total_bits / batch_size if batch_size > 0 else 0})

        # Stack reconstructed images
        result_tensor = torch.stack(reconstructed_images)

        # Return based on configuration
        returns = []
        returns.append(result_tensor)
        if self.return_bits:
            returns.append(bits_per_image)
        if self.return_compressed_data:
            returns.append(compressed_data)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio.

        Args:
            original_size: Size of original data in bits
            compressed_size: Size of compressed data in bits

        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if compressed_size == 0:
            return float("inf")
        return original_size / compressed_size

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics from the last forward pass.

        Returns:
            Dictionary containing compression statistics
        """
        return self.stats.copy() if self.collect_stats else {}
