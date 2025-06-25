"""PNG image compressor using PIL/Pillow."""

import io
from typing import Any, Optional, Tuple, Union

from PIL import Image

from kaira.models.image.compressors.base import BaseImageCompressor


class PNGCompressor(BaseImageCompressor):
    """PNG image compressor using libpng via PIL/Pillow.

    This class provides PNG compression with configurable compression levels and optimization.
    PNG is a lossless compression format that provides good compression for images with
    limited colors, text, or sharp edges.

    The compress_level parameter ranges from 0 (no compression, fastest) to 9 (best compression,
    slowest). Higher compression levels result in smaller file sizes but take more time to process.

    Note: Since PNG is lossless, the "quality" parameter in bit-constrained mode actually
    refers to the compression level, which affects file size but not image quality.

    Example:
        # Fixed compression level
        compressor = PNGCompressor(quality=6)  # quality here means compression level
        compressed_images = compressor(image_batch)

        # Bit-constrained compression
        compressor = PNGCompressor(max_bits_per_image=50000)
        compressed_images, bits_used = compressor(image_batch)

        # With compression statistics
        compressor = PNGCompressor(quality=9, collect_stats=True, return_bits=True)
        compressed_images, bits_per_image = compressor(image_batch)
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,  # For PNG, this represents compression level
        compress_level: Optional[int] = None,  # Alternative parameter name for clarity
        optimize: bool = True,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the PNG compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality/compress_level, the compressor will find the highest
                               compression level that produces files smaller than this limit.
            quality: PNG compression level (0-9, higher = better compression, smaller file size).
                    This is an alias for compress_level to maintain API consistency.
            compress_level: PNG compression level (0-9, higher = better compression).
                           If both quality and compress_level are provided, compress_level takes precedence.
            optimize: Enable PNG optimization for better compression
            collect_stats: Whether to collect and return compression statistics
            return_bits: Whether to return bits per image in forward pass
            return_compressed_data: Whether to return the compressed binary data
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Handle quality vs compress_level parameter naming
        effective_quality: Optional[int]
        if compress_level is not None:
            effective_quality = compress_level
        else:
            effective_quality = quality

        super().__init__(max_bits_per_image, effective_quality, collect_stats, return_bits, return_compressed_data, *args, **kwargs)

        self.optimize = optimize

    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the compression level is within the acceptable range for PNG.

        Args:
            quality: Compression level to validate (0-9 for PNG)

        Raises:
            ValueError: If compression level is not between 0 and 9
        """
        if not isinstance(quality, int) or quality < 0 or quality > 9:
            raise ValueError("PNG compression level must be an integer between 0 and 9")

    def _get_quality_range(self) -> Tuple[int, int]:
        """Get the valid compression level range for PNG compression.

        Returns:
            Tuple of (min_level=0, max_level=9)
        """
        return (0, 9)

    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image using PNG.

        Args:
            image: PIL Image to compress
            quality: PNG compression level (0-9)
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)
        """
        # Ensure image is in appropriate mode for PNG
        # PNG supports RGB, RGBA, L (grayscale), and LA (grayscale + alpha)
        if image.mode not in ["RGB", "RGBA", "L", "LA"]:
            if image.mode == "CMYK":
                image = image.convert("RGB")
            else:
                image = image.convert("RGB")

        # Create bytes buffer
        buffer = io.BytesIO()

        # Save image as PNG with explicit parameters
        image.save(
            buffer,
            format="PNG",
            compress_level=int(quality),
            optimize=self.optimize,
        )

        # Get compressed data
        compressed_data = buffer.getvalue()
        size_in_bits = len(compressed_data) * 8

        return compressed_data, size_in_bits

    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress PNG bytes back to a PIL Image.

        Args:
            data: Compressed PNG data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        buffer = io.BytesIO(data)
        pil_image = Image.open(buffer)  # type: ignore

        # Ensure we load the image data
        pil_image.load()

        # Convert to RGB for consistency (unless it's grayscale)
        if pil_image.mode not in ["RGB", "L"]:
            if pil_image.mode in ["RGBA", "LA"]:
                # For images with alpha, we could either convert to RGB (losing alpha)
                # or keep the alpha channel. For consistency with JPEG, convert to RGB.
                pil_image = pil_image.convert("RGB")  # type: ignore
            else:
                pil_image = pil_image.convert("RGB")  # type: ignore

        return pil_image

    def compress(self, image: Image.Image, compress_level: Optional[int] = None) -> bytes:
        """Compress a PIL Image to PNG bytes.

        This is a convenience method for direct compression without the full forward pass.

        Args:
            image: PIL Image to compress
            compress_level: PNG compression level (uses instance quality if not provided)

        Returns:
            Compressed PNG data as bytes
        """
        actual_compress_level: Union[int, float]
        if compress_level is None:
            if self.quality is None:
                raise ValueError("Compression level must be provided either during initialization or method call")
            actual_compress_level = self.quality
        else:
            actual_compress_level = compress_level

        compressed_data, _ = self._compress_single_image(image, actual_compress_level)
        return compressed_data

    def decompress(self, data: bytes) -> Image.Image:
        """Decompress PNG bytes to PIL Image.

        This is a convenience method for direct decompression.

        Args:
            data: Compressed PNG data as bytes

        Returns:
            Reconstructed PIL Image
        """
        return self._decompress_single_image(data)
