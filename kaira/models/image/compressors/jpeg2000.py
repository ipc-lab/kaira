"""JPEG 2000 image compressor using PIL/Pillow."""

import io
from typing import Any, Optional, Tuple, Union

from PIL import Image

from kaira.models.image.compressors.base import BaseImageCompressor


class JPEG2000Compressor(BaseImageCompressor):
    """JPEG 2000 image compressor using JPEG 2000 via PIL/Pillow.

    This class provides JPEG 2000 compression with configurable quality settings and advanced features.
    JPEG 2000 is a wavelet-based image compression standard that provides superior compression efficiency
    compared to traditional JPEG, especially at lower bit rates. It supports both lossy and lossless
    compression modes.

    The quality parameter ranges from 1 (worst quality, highest compression) to 100 (best quality,
    lowest compression). JPEG 2000 also supports a special lossless mode when quality is set to 100.

    Example:
        # Fixed quality compression
        compressor = JPEG2000Compressor(quality=85)
        compressed_images = compressor(image_batch)

        # Bit-constrained compression
        compressor = JPEG2000Compressor(max_bits_per_image=4000)
        compressed_images, bits_used = compressor(image_batch)

        # Lossless compression
        compressor = JPEG2000Compressor(quality=100)
        compressed_images = compressor(image_batch)

        # With compression statistics
        compressor = JPEG2000Compressor(quality=90, collect_stats=True, return_bits=True)
        compressed_images, bits_per_image = compressor(image_batch)
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        irreversible: Optional[bool] = None,
        progression_order: str = "LRCP",
        num_resolutions: int = 6,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the JPEG 2000 compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: JPEG 2000 quality level (1-100, higher = better quality, larger file size).
                    If provided, this exact quality will be used regardless of resulting file size.
                    Quality 100 enables lossless mode unless irreversible=True is explicitly set.
            irreversible: Force irreversible (lossy) compression even at high quality.
                         If None, automatically determined based on quality (>= 100 = reversible).
            progression_order: Progression order for encoding ("LRCP", "RLCP", "RPCL", "PCRL", "CPRL").
            num_resolutions: Number of resolution levels (1-33). More levels = better scalability.
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

        self.irreversible = irreversible
        self.progression_order = progression_order
        self.num_resolutions = num_resolutions

        # Validate progression order
        valid_orders = ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]
        if progression_order not in valid_orders:
            raise ValueError(f"Progression order must be one of {valid_orders}")

        # Validate number of resolutions
        if not isinstance(num_resolutions, int) or num_resolutions < 1 or num_resolutions > 33:
            raise ValueError("Number of resolutions must be an integer between 1 and 33")

    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the quality is within the acceptable range for JPEG 2000.

        Args:
            quality: Quality level to validate (1-100 for JPEG 2000)

        Raises:
            ValueError: If quality is not between 1 and 100
        """
        if not isinstance(quality, (int, float)) or quality < 1 or quality > 100:
            raise ValueError("JPEG 2000 quality must be between 1 and 100")

    def _get_quality_range(self) -> Tuple[int, int]:
        """Get the valid quality range for JPEG 2000 compression.

        Returns:
            Tuple of (min_quality=1, max_quality=100)
        """
        return (1, 100)

    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image using JPEG 2000.

        Args:
            image: PIL Image to compress
            quality: JPEG 2000 quality level (1-100)
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)
        """
        # Ensure image is in appropriate mode for JPEG 2000
        # JPEG 2000 supports RGB, RGBA, L (grayscale)
        if image.mode not in ["RGB", "RGBA", "L"]:
            if image.mode == "CMYK":
                image = image.convert("RGB")
            else:
                image = image.convert("RGB")

        # Create bytes buffer
        buffer = io.BytesIO()

        # Determine compression mode
        use_irreversible = self.irreversible
        if use_irreversible is None:
            # Auto-determine based on quality
            use_irreversible = quality < 100

        # Prepare save parameters
        save_params = {
            "format": "JPEG2000",
            "irreversible": use_irreversible,
            "progression": self.progression_order,
            "num_resolutions": self.num_resolutions,
        }

        if not use_irreversible:
            # Lossless mode - ignore quality
            pass
        else:
            # Lossy mode - map quality to compression ratio
            # Quality 1 -> high compression ratio (100:1)
            # Quality 99 -> low compression ratio (2:1)
            compression_ratio = 100 - (quality - 1) * 98 / 98
            save_params["quality_mode"] = "rates"
            save_params["quality_layers"] = [compression_ratio]

        # Save image as JPEG 2000
        try:
            image.save(buffer, **save_params)  # type: ignore[arg-type]
        except Exception:
            # Fallback with minimal parameters if advanced features aren't supported
            try:
                if use_irreversible and quality < 100:
                    image.save(buffer, format="JPEG2000", irreversible=True)
                else:
                    image.save(buffer, format="JPEG2000")
            except Exception:
                # Final fallback - basic JPEG2000 save
                image.save(buffer, format="JPEG2000")

        # Get compressed data
        compressed_data = buffer.getvalue()
        size_in_bits = len(compressed_data) * 8

        return compressed_data, size_in_bits

    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress JPEG 2000 bytes back to a PIL Image.

        Args:
            data: Compressed JPEG 2000 data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        buffer = io.BytesIO(data)
        pil_image = Image.open(buffer)  # type: ignore

        # Ensure we load the image data
        pil_image.load()

        # Convert to RGB if not already (for consistency, but preserve grayscale)
        if pil_image.mode not in ["RGB", "L", "RGBA"]:
            pil_image = pil_image.convert("RGB")  # type: ignore

        return pil_image

    def compress(self, image: Image.Image, quality: Optional[int] = None) -> bytes:
        """Compress a PIL Image to JPEG 2000 bytes.

        This is a convenience method for direct compression without the full forward pass.

        Args:
            image: PIL Image to compress
            quality: JPEG 2000 quality level (uses instance quality if not provided)

        Returns:
            Compressed JPEG 2000 data as bytes
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
        """Decompress JPEG 2000 bytes to PIL Image.

        This is a convenience method for direct decompression.

        Args:
            data: Compressed JPEG 2000 data as bytes

        Returns:
            Reconstructed PIL Image
        """
        return self._decompress_single_image(data)
