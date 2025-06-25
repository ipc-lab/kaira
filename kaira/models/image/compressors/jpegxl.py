"""JPEG XL image compressor using pillow-jxl."""

import io
from typing import Any, Optional, Tuple, Union

from PIL import Image

from kaira.models.image.compressors.base import BaseImageCompressor


class JPEGXLCompressor(BaseImageCompressor):
    """JPEG XL image compressor using JPEG XL via PIL/Pillow.

    This class provides JPEG XL compression with configurable quality settings and advanced features.
    JPEG XL is a modern image compression format that provides superior compression efficiency
    compared to traditional JPEG while maintaining excellent visual quality. It supports both
    lossy and lossless compression modes.

    The quality parameter ranges from 1 (worst quality, highest compression) to 100 (best quality,
    lowest compression). JPEG XL also supports a special lossless mode when quality is set to 100.

    Example:
        # Fixed quality compression
        compressor = JPEGXLCompressor(quality=85)
        compressed_images = compressor(image_batch)

        # Bit-constrained compression
        compressor = JPEGXLCompressor(max_bits_per_image=3000)
        compressed_images, bits_used = compressor(image_batch)

        # Lossless compression
        compressor = JPEGXLCompressor(quality=100)
        compressed_images = compressor(image_batch)

        # With compression statistics
        compressor = JPEGXLCompressor(quality=90, collect_stats=True, return_bits=True)
        compressed_images, bits_per_image = compressor(image_batch)
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        effort: int = 7,
        lossless: bool = False,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the JPEG XL compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: JPEG XL quality level (1-100, higher = better quality, larger file size).
                    If provided, this exact quality will be used regardless of resulting file size.
                    Quality 100 enables lossless mode unless lossless=False is explicitly set.
            effort: Encoding effort (1-9, higher = slower but potentially better compression).
                   Default is 7 for good balance of speed and compression.
            lossless: Force lossless mode regardless of quality setting.
            collect_stats: Whether to collect and return compression statistics
            return_bits: Whether to return bits per image in forward pass
            return_compressed_data: Whether to return the compressed binary data
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(
            max_bits_per_image,
            quality if not lossless else 100,  # Use quality 100 for lossless mode if quality not provided
            collect_stats,
            return_bits,
            return_compressed_data,
            *args,
            **kwargs,
        )

        self.effort = effort
        self.lossless = lossless

        # Validate effort parameter
        if not isinstance(effort, int) or effort < 1 or effort > 9:
            raise ValueError("JPEG XL effort must be an integer between 1 and 9")

    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the quality is within the acceptable range for JPEG XL.

        Args:
            quality: Quality level to validate (1-100 for JPEG XL)

        Raises:
            ValueError: If quality is not between 1 and 100
        """
        if not isinstance(quality, (int, float)) or quality < 1 or quality > 100:
            raise ValueError("JPEG XL quality must be between 1 and 100")

    def _get_quality_range(self) -> Tuple[int, int]:
        """Get the valid quality range for JPEG XL compression.

        Returns:
            Tuple of (min_quality=1, max_quality=100)
        """
        return (1, 100)

    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image using JPEG XL.

        Args:
            image: PIL Image to compress
            quality: JPEG XL quality level (1-100)
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)

        Note:
            If JPEG XL is not supported by the current PIL installation,
            this will fall back to JPEG compression with a warning.
        """
        # Ensure image is in RGB mode for JPEG XL
        if image.mode not in ["RGB", "RGBA", "L"]:
            image = image.convert("RGB")

        # Create bytes buffer
        buffer = io.BytesIO()

        # Try to import JPEG XL plugin if available
        try:
            import importlib.util

            if importlib.util.find_spec("pillow_jxl"):
                import pillow_jxl  # This registers the JXL format  # noqa: F401
        except ImportError:
            pass

        # Check if JPEG XL is supported (use 'JXL' format name)
        if "JXL" not in Image.SAVE:
            # Fallback to JPEG if JPEG XL is not supported
            import warnings

            warnings.warn("JPEG XL format not supported by current PIL installation. Falling back to JPEG compression.")

            # Use JPEG as fallback
            image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
        else:
            # Determine if we should use lossless mode
            use_lossless = self.lossless or (quality >= 100)

            # Prepare save parameters
            save_params = {
                "format": "JXL",
                "effort": self.effort,
            }

            if use_lossless:
                save_params["lossless"] = True
            else:
                # For lossy mode, map quality (1-100) to distance parameter
                # JPEG XL uses distance where lower values = higher quality
                # Quality 100 -> distance ~0.1, Quality 1 -> distance ~15
                distance = 15.0 - (quality - 1) * 14.9 / 99
                save_params["distance"] = max(0.1, distance)

            # Save image as JPEG XL
            try:
                image.save(buffer, **save_params)  # type: ignore[arg-type]
            except Exception:
                # Fallback without advanced parameters if they're not supported
                image.save(buffer, format="JXL")

        # Get compressed data
        compressed_data = buffer.getvalue()
        size_in_bits = len(compressed_data) * 8

        return compressed_data, size_in_bits

    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress JPEG XL bytes back to a PIL Image.

        Args:
            data: Compressed JPEG XL data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        buffer = io.BytesIO(data)
        pil_image = Image.open(buffer)  # type: ignore

        # Ensure we load the image data
        pil_image.load()

        # Convert to RGB if not already (for consistency)
        if pil_image.mode not in ["RGB", "L"]:
            pil_image = pil_image.convert("RGB")  # type: ignore

        return pil_image

    def compress(self, image: Image.Image, quality: Optional[int] = None) -> bytes:
        """Compress a PIL Image to JPEG XL bytes.

        This is a convenience method for direct compression without the full forward pass.

        Args:
            image: PIL Image to compress
            quality: JPEG XL quality level (uses instance quality if not provided)

        Returns:
            Compressed JPEG XL data as bytes
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
        """Decompress JPEG XL bytes to PIL Image.

        This is a convenience method for direct decompression.

        Args:
            data: Compressed JPEG XL data as bytes

        Returns:
            Reconstructed PIL Image
        """
        return self._decompress_single_image(data)
