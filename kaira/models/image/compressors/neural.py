"""Wrapper for neural network-based image compressors from CompressAI."""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import compressai.zoo
import torch

from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model()
class NeuralCompressor(BaseModel):
    """Neural network-based image compression model.

    This class provides neural network-based compression using various pretrained models
    from the CompressAI library. It can operate in two modes:
    1. Fixed quality mode: directly uses the specified quality level
    2. Bit-constrained mode: finds the highest quality that stays under a bit budget

    The implementation efficiently manages model loading to minimize memory usage
    and supports a variety of modern image compression methods.
    """

    def __init__(
        self,
        method: str,
        metric: str = "mse",
        max_bits_per_image: Optional[int] = None,
        quality: Optional[int] = None,
        lazy_loading: bool = True,
        return_bits: bool = True,
        collect_stats: bool = False,
        return_compressed_data: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        early_stopping_threshold: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the neural compressor.

        Args:
            method: Compression method to use (e.g., "bmshj2018_factorized")
            metric: Metric used for training ("mse" or "ms-ssim")
            max_bits_per_image: Maximum bits allowed per image
            quality: Specific quality level to use
            lazy_loading: Whether to load models only when needed (saves memory)
            return_bits: Whether to return bits per image in forward pass
            collect_stats: Whether to collect and return compression statistics
            return_compressed_data: Whether to return compressed representation
            device: Device to load models on (e.g., "cuda", "cpu")
            early_stopping_threshold: Bit threshold below which to stop quality search
                                     (e.g., 0.95 means stop if within 5% of bit budget)
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # At least one of the two parameters must be provided
        if max_bits_per_image is None and quality is None:
            raise ValueError("At least one of max_bits_per_image or quality must be provided")

        self.possible_qualities = {
            # Standard models from CompressAI
            "cheng2020_anchor": list(range(1, 7)),
            "cheng2020_attn": list(range(1, 7)),
            "bmshj2018_factorized": list(range(1, 9)),
            "bmshj2018_factorized_relu": list(range(1, 9)),
            "mbt2018": list(range(1, 9)),
            "mbt2018_mean": list(range(1, 9)),
            "bmshj2018_hyperprior": list(range(1, 9)),
        }

        if method not in self.possible_qualities:
            available_methods = list(self.possible_qualities.keys())
            raise ValueError(f"Method '{method}' is not supported. Available methods: {available_methods}")

        if quality is not None and quality not in self.possible_qualities[method]:
            raise ValueError(f"Quality must be in {str(self.possible_qualities[method])}")

        if metric not in ["ms-ssim", "mse"]:
            raise ValueError("Metric must be 'ms-ssim' or 'mse'")

        self.method = method
        self.max_bits_per_image = max_bits_per_image
        self.quality = quality
        self.metric = metric
        self.lazy_loading = lazy_loading
        self.return_bits = return_bits
        self.collect_stats = collect_stats
        self.return_compressed_data = return_compressed_data
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.early_stopping_threshold = early_stopping_threshold
        self.stats: Dict[str, Any] = {}
        self._models_cache = {}

        # Initialize models - either load them all or prepare for lazy loading
        if not lazy_loading:
            if quality is not None:
                self._models_cache[quality] = self._load_model(quality)
            else:
                # If bit-constrained mode, we'll likely need multiple qualities
                # Load models from highest to lowest quality for better user experience
                for q in reversed(self.possible_qualities[method]):
                    self._models_cache[q] = self._load_model(q)

    def _load_model(self, quality: int) -> torch.nn.Module:
        """Load a model with the specified quality."""
        return getattr(compressai.zoo, self.method)(quality=quality, pretrained=True, metric=self.metric).to(self.device).eval()

    def get_model(self, quality: int) -> torch.nn.Module:
        """Get a model with the specified quality, using cache if available."""
        if quality not in self._models_cache:
            self._models_cache[quality] = self._load_model(quality)
        return self._models_cache[quality]

    def compute_bits_compressai(self, r: Dict) -> torch.Tensor:
        """Compute bits required for each image in the batch.

        Args:
            r: CompressAI model output dictionary

        Returns:
            Tensor containing bits per image
        """
        likelihoods = r["likelihoods"].values()

        n = r["x_hat"].shape[0]
        device = r["x_hat"].device

        # Create output tensor
        all_num_bits = torch.zeros(n, device=device)

        # Calculate bits for each image
        for i in range(n):
            for likelihood in likelihoods:
                all_num_bits[i] += -torch.log2(likelihood[i]).sum()

        return all_num_bits

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[Any]], Tuple[torch.Tensor, torch.Tensor, List[Any]]]:
        """Forward pass of the neural compressor.

        Args:
            x: Input image tensor
            *args: Additional positional arguments (unused in this method).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            If no additional returns: Just the reconstructed image
            If return_bits=True: Tuple of (reconstructed image, bits per image tensor)
            If return_compressed_data=True: Tuple of (reconstructed image, compressed data list)
            If both are True: Tuple of (reconstructed image, bits per image tensor, compressed data list)
        """
        start_time = time.time()

        # Initialize stats if collecting
        if self.collect_stats:
            self.stats = {"total_bits": 0, "avg_quality": 0, "img_stats": [], "model_name": self.method, "metric": self.metric, "processing_time": 0}

        if self.quality is not None:
            # When quality is specified, use that directly
            model = self.get_model(self.quality).to(x.device)

            # Get compressed representation if needed
            compressed_data = None
            if self.return_compressed_data:
                compressed_data = []
                for i in range(x.shape[0]):
                    # Use the compress() method from CompressAI models
                    comp_data = model.compress(x[i : i + 1])
                    compressed_data.append(comp_data)

            # Regular forward pass for reconstruction
            res = model(x)
            bits = self.compute_bits_compressai(res)

            reconstructed = res["x_hat"]

            if self.max_bits_per_image is not None and torch.any(bits > self.max_bits_per_image):
                warnings.warn(f"Some images exceed the max_bits_per_image constraint ({self.max_bits_per_image})")

            # Collect stats if requested
            if self.collect_stats:
                original_size = x.shape[1] * x.shape[2] * x.shape[3] * 8  # Original size in bits (8 bits per channel)
                self.stats["total_bits"] = bits.sum().item()
                self.stats["avg_quality"] = self.quality
                self.stats["processing_time"] = time.time() - start_time

                for i in range(x.shape[0]):
                    self.stats["img_stats"].append({"quality": self.quality, "bits": bits[i].item(), "bpp": bits[i].item() / (x.shape[2] * x.shape[3]), "compression_ratio": original_size / bits[i].item() if bits[i].item() > 0 else 0})

                self.stats["avg_bpp"] = self.stats["total_bits"] / (x.shape[0] * x.shape[2] * x.shape[3])
                self.stats["avg_compression_ratio"] = sum(s["compression_ratio"] for s in self.stats["img_stats"]) / x.shape[0]

            # Determine what to return based on flags
            if self.return_bits and self.return_compressed_data:
                return reconstructed, bits, compressed_data if compressed_data is not None else []
            elif self.return_bits:
                return reconstructed, bits
            elif self.return_compressed_data:
                return reconstructed, compressed_data if compressed_data is not None else []
            else:
                return reconstructed

        # Find optimal quality for each image based on bit constraint
        available_qualities = sorted(self.possible_qualities[self.method], reverse=True)
        best_qualities = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        x_hat = torch.empty_like(x)
        output_bits = torch.zeros(x.shape[0], device=x.device)

        # Initialize compressed data storage as a list if needed, otherwise None
        optimal_compressed_data: Optional[List[Any]] = [None] * x.shape[0] if self.return_compressed_data else None

        # For stats collection
        if self.collect_stats:
            original_size = x.shape[1] * x.shape[2] * x.shape[3] * 8  # Original size in bits (8 bits per channel)
            img_stats: List[Dict[str, Any]] = [{} for _ in range(x.shape[0])]

        # Start with best quality for all images
        current_quality_idx = 0
        remaining_images = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        # Iterate through qualities from highest to lowest
        while torch.any(remaining_images) and current_quality_idx < len(available_qualities):
            quality = available_qualities[current_quality_idx]
            model = self.get_model(quality).to(x.device)

            # Only process images that haven't found their optimal quality yet
            if torch.all(~remaining_images):
                break

            # Process current batch of remaining images
            current_batch = x[remaining_images]
            if current_batch.shape[0] > 0:
                res = model(current_batch)
                bits = self.compute_bits_compressai(res)

                # If compressed data is requested, get it for each image
                if self.return_compressed_data:
                    original_indices = torch.nonzero(remaining_images).squeeze(1)
                    for i, orig_idx in enumerate(original_indices):
                        # Store in case this is the best quality for this image
                        temp_comp_data = model.compress(current_batch[i : i + 1])
                        # We'll only keep it if the constraint is satisfied
                        if bits[i] <= self.max_bits_per_image:
                            # Now we know optimal_compressed_data is a list when this runs
                            assert optimal_compressed_data is not None
                            optimal_compressed_data[orig_idx.item()] = temp_comp_data

                # Mark images that satisfy the constraint with this quality
                satisfies_constraint = bits <= self.max_bits_per_image

                # Get indices in the original batch
                original_indices = torch.nonzero(remaining_images).squeeze(1)

                # Regular case with real tensors
                for i, orig_idx in enumerate(original_indices[satisfies_constraint]):
                    x_hat[orig_idx] = res["x_hat"][i]
                    output_bits[orig_idx] = bits[i]
                    best_qualities[orig_idx] = quality

                    # Collect stats if needed
                    if self.collect_stats:
                        img_stats[orig_idx.item()] = {"quality": quality, "bits": bits[i].item(), "bpp": bits[i].item() / (x.shape[2] * x.shape[3]), "compression_ratio": original_size / bits[i].item() if bits[i].item() > 0 else 0}

                # Update remaining_images mask
                remaining_images[original_indices[satisfies_constraint]] = False

                # Early stopping if within threshold
                if self.early_stopping_threshold is not None and self.max_bits_per_image is not None:
                    threshold_bits = self.max_bits_per_image * self.early_stopping_threshold
                    if torch.all(bits <= threshold_bits):
                        break

            current_quality_idx += 1

        # For any remaining images, use the lowest quality
        if torch.any(remaining_images):
            lowest_quality = available_qualities[-1]
            model = self.get_model(lowest_quality).to(x.device)

            current_batch = x[remaining_images]
            if current_batch.shape[0] > 0:
                res = model(current_batch)
                bits = self.compute_bits_compressai(res)

                # Get indices in the original batch
                original_indices = torch.nonzero(remaining_images).squeeze(1)

                # Update all remaining images
                for i, orig_idx in enumerate(original_indices):
                    x_hat[orig_idx] = res["x_hat"][i]
                    output_bits[orig_idx] = bits[i]
                    best_qualities[orig_idx] = lowest_quality

                    # Collect stats if needed
                    if self.collect_stats:
                        img_stats[orig_idx.item()] = {"quality": lowest_quality, "bits": bits[i].item(), "bpp": bits[i].item() / (x.shape[2] * x.shape[3]), "compression_ratio": original_size / bits[i].item() if bits[i].item() > 0 else 0}

                # Warn if some images still exceed the max_bits_per_image
                if torch.any(bits > self.max_bits_per_image):
                    warnings.warn("Some images exceed max_bits_per_image even at lowest quality")

        # Update stats if collecting
        if self.collect_stats:
            self.stats["img_stats"] = img_stats
            self.stats["total_bits"] = output_bits.sum().item()
            self.stats["avg_quality"] = best_qualities.float().mean().item()
            self.stats["avg_bpp"] = self.stats["total_bits"] / (x.shape[0] * x.shape[2] * x.shape[3])
            self.stats["avg_compression_ratio"] = sum(s["compression_ratio"] for s in img_stats) / x.shape[0]

        # Add processing time to stats
        if self.collect_stats:
            self.stats["processing_time"] = time.time() - start_time

        # Determine what to return based on flags
        if self.return_bits and self.return_compressed_data:
            # Ensure optimal_compressed_data is a list when returning it
            return x_hat, output_bits, optimal_compressed_data if optimal_compressed_data is not None else []
        elif self.return_bits:
            return x_hat, output_bits
        elif self.return_compressed_data:
            # Ensure optimal_compressed_data is a list when returning it
            return x_hat, optimal_compressed_data if optimal_compressed_data is not None else []
        else:
            return x_hat

    def get_stats(self):
        """Return compression statistics if collect_stats=True was set."""
        if not self.collect_stats:
            warnings.warn("Statistics not collected. Initialize with collect_stats=True to enable.")
            return {}
        return self.stats

    def get_bits_per_image(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compress images and return only the bit counts per image.

        Args:
            x: Tensor of shape [batch_size, channels, height, width]
            *args: Additional positional arguments passed to forward.
            **kwargs: Additional keyword arguments passed to forward.

        Returns:
            Tensor: Number of bits used for each compressed image
        """
        # Temporarily override return settings
        original_return_bits = self.return_bits
        original_return_compressed = self.return_compressed_data
        self.return_bits = True
        self.return_compressed_data = False  # Ensure only bits are requested from forward

        try:
            # Pass *args, **kwargs to forward
            forward_output = self.forward(x, *args, **kwargs)
            # Ensure forward returned the expected tuple when return_bits is True
            if isinstance(forward_output, tuple) and len(forward_output) >= 2:
                bits_per_image = forward_output[1]
                if not isinstance(bits_per_image, torch.Tensor):
                    raise TypeError(f"Expected tensor of bits, but got {type(bits_per_image)}")
            else:
                # Handle case where forward might just return the tensor if return_bits was originally False
                # This shouldn't happen with the temporary override, but good to be safe.
                if isinstance(forward_output, torch.Tensor) and not original_return_bits:
                    # Re-run forward correctly requesting bits if the first attempt failed due to original settings
                    self.return_bits = True
                    forward_output = self.forward(x, *args, **kwargs)
                    if isinstance(forward_output, tuple) and len(forward_output) >= 2:
                        bits_per_image = forward_output[1]
                        if not isinstance(bits_per_image, torch.Tensor):
                            raise TypeError(f"Expected tensor of bits on second attempt, but got {type(bits_per_image)}")
                    else:
                        raise TypeError(f"Forward method did not return expected tuple (tensor, bits) on second attempt, got {type(forward_output)}")
                else:
                    raise TypeError(f"Forward method did not return expected tuple (tensor, bits), got {type(forward_output)}")

        finally:
            # Restore original settings
            self.return_bits = original_return_bits
            self.return_compressed_data = original_return_compressed

        return bits_per_image

    def reset_stats(self):
        """Reset all collected statistics."""
        if not self.collect_stats:
            warnings.warn("Statistics not collected. Initialize with collect_stats=True to enable.")
            return

        self.stats = {"total_bits": 0, "avg_quality": 0, "img_stats": [], "model_name": self.method, "metric": self.metric, "processing_time": 0, "avg_bpp": 0, "avg_compression_ratio": 0}
