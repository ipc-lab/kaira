"""WebP image compressor using PIL/Pillow."""

import io
from typing import Any, Optional, Tuple, Union

from PIL import Image

from kaira.models.image.compressors.base import BaseImageCompressor


class WebPCompressor(BaseImageCompressor):
    """WebP image compressor using WebP via PIL/Pillow.

    This class provides WebP compression with configurable quality settings and advanced features.
    WebP is a modern image format developed by Google that provides superior compression efficiency
    compared to JPEG and PNG while maintaining excellent visual quality. It supports both lossy
    and lossless compression modes, as well as transparency and animation.

    The quality parameter ranges from 1 (worst quality, highest compression) to 100 (best quality,
    lowest compression). WebP also supports a special lossless mode when lossless=True.

    Example:
        # Fixed quality compression
        compressor = WebPCompressor(quality=85)
        compressed_images = compressor(image_batch)

        # Bit-constrained compression
        compressor = WebPCompressor(max_bits_per_image=3500)
        compressed_images, bits_used = compressor(image_batch)

        # Lossless compression
        compressor = WebPCompressor(lossless=True)
        compressed_images = compressor(image_batch)

        # High-effort compression
        compressor = WebPCompressor(quality=90, method=6, collect_stats=True, return_bits=True)
        compressed_images, bits_per_image = compressor(image_batch)
        stats = compressor.get_stats()
    """

    def __init__(
        self,
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        lossless: bool = False,
        method: int = 4,
        exact: bool = False,
        collect_stats: bool = False,
        return_bits: bool = True,
        return_compressed_data: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the WebP compressor.

        Args:
            max_bits_per_image: Maximum bits allowed per compressed image. If provided without
                               quality, the compressor will find the highest quality that
                               produces files smaller than this limit.
            quality: WebP quality level (1-100, higher = better quality, larger file size).
                    If provided, this exact quality will be used regardless of resulting file size.
                    Ignored when lossless=True.
            lossless: Enable lossless compression mode. When True, quality parameter is ignored.
            method: Compression method (0-6, higher = slower but potentially better compression).
                   0 = fastest, 6 = slowest but best compression. Default is 4 for balance.
            exact: Preserve RGB values in transparent regions (useful for lossless).
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

        self.lossless = lossless
        self.method = method
        self.exact = exact

        # Validate method parameter
        if not isinstance(method, int) or method < 0 or method > 6:
            raise ValueError("WebP method must be an integer between 0 and 6")

    def _validate_quality(self, quality: Union[int, float]) -> None:
        """Validate that the quality is within the acceptable range for WebP.

        Args:
            quality: Quality level to validate (1-100 for WebP)

        Raises:
            ValueError: If quality is not between 1 and 100
        """
        if not isinstance(quality, (int, float)) or quality < 1 or quality > 100:
            raise ValueError("WebP quality must be between 1 and 100")

    def _get_quality_range(self) -> Tuple[int, int]:
        """Get the valid quality range for WebP compression.

        Returns:
            Tuple of (min_quality=1, max_quality=100)
        """
        return (1, 100)

    def _compress_single_image(self, image: Image.Image, quality: Union[int, float], **kwargs: Any) -> Tuple[bytes, int]:
        """Compress a single PIL Image using WebP.

        Args:
            image: PIL Image to compress
            quality: WebP quality level (1-100, ignored if lossless=True)
            **kwargs: Additional compression parameters

        Returns:
            Tuple of (compressed_data_bytes, size_in_bits)
        """
        # WebP supports RGB, RGBA modes well
        if image.mode not in ["RGB", "RGBA"]:
            if image.mode == "L":
                # Convert grayscale to RGB for WebP
                image = image.convert("RGB")
            elif image.mode in ["CMYK", "YCbCr"]:
                image = image.convert("RGB")
            else:
                image = image.convert("RGB")

        # Create bytes buffer
        buffer = io.BytesIO()

        # Prepare save parameters
        save_params = {
            "format": "WebP",
            "method": self.method,
            "exact": self.exact,
        }

        if self.lossless:
            save_params["lossless"] = True
            # In lossless mode, quality parameter is ignored
        else:
            save_params["quality"] = int(quality)

        # Save image as WebP
        try:
            image.save(buffer, **save_params)  # type: ignore[arg-type]
        except Exception:
            # Fallback with basic parameters if advanced features aren't supported
            try:
                if self.lossless:
                    image.save(buffer, format="WebP", lossless=True)
                else:
                    image.save(buffer, format="WebP", quality=int(quality))
            except Exception:
                # Final fallback - basic WebP save
                image.save(buffer, format="WebP")

        # Get compressed data
        compressed_data = buffer.getvalue()
        size_in_bits = len(compressed_data) * 8

        return compressed_data, size_in_bits

    def _decompress_single_image(self, data: bytes, **kwargs: Any) -> Image.Image:
        """Decompress WebP bytes back to a PIL Image.

        Args:
            data: Compressed WebP data as bytes
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed PIL Image
        """
        buffer = io.BytesIO(data)
        pil_image = Image.open(buffer)  # type: ignore

        # Ensure we load the image data
        pil_image.load()

        # Convert to RGB if not already (unless it has transparency)
        if pil_image.mode not in ["RGB", "RGBA"]:
            if pil_image.mode == "L":
                # Keep grayscale as RGB for consistency
                pil_image = pil_image.convert("RGB")  # type: ignore
            else:
                pil_image = pil_image.convert("RGB")  # type: ignore

        return pil_image

    def compress(self, image: Image.Image, quality: Optional[int] = None) -> bytes:
        """Compress a PIL Image to WebP bytes.

        This is a convenience method for direct compression without the full forward pass.

        Args:
            image: PIL Image to compress
            quality: WebP quality level (uses instance quality if not provided, ignored if lossless=True)

        Returns:
            Compressed WebP data as bytes
        """
        if self.lossless:
            # In lossless mode, we can use any quality value since it's ignored
            actual_quality: Union[int, float] = 100
        else:
            if quality is None:
                if self.quality is None:
                    raise ValueError("Quality must be provided either during initialization or method call")
                actual_quality = self.quality
            else:
                actual_quality = quality

        compressed_data, _ = self._compress_single_image(image, actual_quality)
        return compressed_data

    def decompress(self, data: bytes) -> Image.Image:
        """Decompress WebP bytes to PIL Image.

        This is a convenience method for direct decompression.

        Args:
            data: Compressed WebP data as bytes

        Returns:
            Reconstructed PIL Image
        """
        return self._decompress_single_image(data)
