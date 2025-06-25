"""JPEG image compressor using PIL/Pillow."""

import io
from typing import Any, Optional, Tuple, Union

from PIL import Image

from kaira.models.image.compressors.base import BaseImageCompressor


class JPEGCompressor(BaseImageCompressor):
    """JPEG image compressor using libjpeg via PIL/Pillow.

    This class provides JPEG compression with standard quality settings and optimization options.
    JPEG is a widely-used lossy compression format that provides good compression ratios for
    photographic images.

    The quality parameter ranges from 1 (worst quality, highest compression) to 100 (best quality,
    lowest compression). Higher quality values result in larger file sizes but better image quality.

    Example:
        # Fixed quality compression
        compressor = JPEGCompressor(quality=85)
        compressed_images = compressor(image_batch)

        # Bit-constrained compression
        compressor = JPEGCompressor(max_bits_per_image=5000)
        compressed_images, bits_used = compressor(image_batch)

        # With compression statistics
        compressor = JPEGCompressor(quality=75, collect_stats=True, return_bits=True)
        compressed_images, bits_per_image = compressor(image_batch)
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        optimize: bool = True,
        progressive: bool = False,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the JPEG compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: JPEG quality level (1-100, higher = better quality, larger file size).
                    If provided, this exact quality will be used regardless of resulting file size.
            optimize: Enable JPEG optimization for better compression
            progressive: Enable progressive JPEG encoding
            collect_stats: Whether to collect and return compression statistics
            return_bits: Whether to return bits per image in forward pass
            return_compressed_data: Whether to return the compressed binary data
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(
            max_bits_per_image,
            quality,
            collect_stats,
            return_bits,
            return_compressed_data,
            *args,
            **kwargs,
        )

        self.optimize = optimize
        self.progressive = progressive

    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the quality parameter is within the acceptable range for JPEG.

        Args:
            quality: Quality level to validate

        Raises:
            ValueError: If quality is not between 1 and 100
        """
        if not isinstance(quality, int) or quality < 1 or quality > 100:
            raise ValueError("JPEG quality must be an integer between 1 and 100")

    def _get_quality_range(self) -> Tuple[int, int]:
        """Get the valid quality range for JPEG compression.

        Returns:
            Tuple of (min_quality=1, max_quality=100)
        """
        return (1, 100)

    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image using JPEG.

        Args:
            image: PIL Image to compress
            quality: JPEG quality level (1-100)
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)
        """
        # Ensure image is in RGB mode for JPEG
        if image.mode not in ["RGB", "L"]:
            image = image.convert("RGB")

        # Create bytes buffer
        buffer = io.BytesIO()

        # Save image as JPEG with explicit parameters
        image.save(
            buffer,
            format="JPEG",
            quality=int(quality),
            optimize=self.optimize,
            progressive=self.progressive,
        )

        # Get compressed data
        compressed_data = buffer.getvalue()
        size_in_bits = len(compressed_data) * 8

        return compressed_data, size_in_bits

    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress JPEG bytes back to a PIL Image.

        Args:
            data: Compressed JPEG data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        buffer = io.BytesIO(data)
        pil_image = Image.open(buffer)  # type: ignore

        # Ensure we load the image data
        pil_image.load()

        # Convert to RGB if not already (JPEG sometimes opens as different modes)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")  # type: ignore

        return pil_image

    def compress(self, image: Image.Image, quality: Optional[int] = None) -> bytes:
        """Compress a PIL Image to JPEG bytes.

        This is a convenience method for direct compression without the full forward pass.

        Args:
            image: PIL Image to compress
            quality: JPEG quality level (uses instance quality if not provided)

        Returns:
            Compressed JPEG data as bytes
        """
        actual_quality: Union[int, float]
        if quality is None:
            if self.quality is None:
                raise ValueError("Quality must be provided either during initialization or method call")
            actual_quality = self.quality
        else:
            actual_quality = quality

        compressed_data, _ = self._compress_single_image(image, actual_quality)
        return compressed_data

    def decompress(self, data: bytes) -> Image.Image:
        """Decompress JPEG bytes to PIL Image.

        This is a convenience method for direct decompression.

        Args:
            data: Compressed JPEG data as bytes

        Returns:
            Reconstructed PIL Image
        """
        return self._decompress_single_image(data)
