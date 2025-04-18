"""Soft bit thresholding module for binary data processing.

This module provides various thresholding techniques for converting soft bit representations
(probabilities, LLRs, etc.) to hard decisions. These thresholders can be used with soft decoders or
as standalone components in signal processing pipelines.

Soft bit processing is crucial in modern communication systems to extract maximum information from
the received signals. The techniques implemented here are based on established methods in
communication theory.
"""

from typing import Any, Optional

import torch

from kaira.models.registry import ModelRegistry

from ..base import BaseModel


class SoftBitThresholder(BaseModel):
    """Base class for soft bit thresholding techniques.

    This abstract class defines the interface for soft bit thresholders that convert soft bit
    representations (e.g., probabilities, LLRs) to hard binary decisions.

    Soft bit thresholding is a key technique in modern communication systems for extracting
    reliable information from noisy channel outputs.

    Implementers must override the forward method.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the soft bit thresholder.

        Args:
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        raise NotImplementedError("Subclasses must implement forward method")


@ModelRegistry.register_model("fixed_thresholder")
class FixedThresholder(SoftBitThresholder):
    """Simple fixed threshold for soft bit values.

    Applies a fixed threshold to convert soft bit values to hard decisions.
    For probability inputs (in range [0,1]), the default threshold is 0.5.
    For LLR inputs, the default threshold is 0.0.

    Example:
        With threshold=0.5 and input [0.2, 0.7, 0.4, 0.9]:
        Output will be [0.0, 1.0, 0.0, 1.0]
    """

    def __init__(self, threshold: float = 0.5, input_type: str = "prob", *args: Any, **kwargs: Any):
        """Initialize the fixed thresholder.

        Args:
            threshold: The threshold value to use. Default is 0.5 for probabilities.
            input_type: Type of soft input, can be 'prob' (probabilities between 0 and 1) or
                       'llr' (log-likelihood ratios). Affects the default threshold if not specified.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # Set appropriate default threshold based on input type
        if input_type == "llr" and threshold == 0.5:
            threshold = 0.0  # Default threshold for LLRs is 0

        self.threshold = threshold
        self.input_type = input_type

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply fixed thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        if self.input_type == "prob":
            # For probability values (between 0 and 1)
            return (x > self.threshold).float()
        elif self.input_type == "llr":
            # For LLRs, negative values favor bit=1, positive values favor bit=0
            return (x > self.threshold).float()
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")


@ModelRegistry.register_model("adaptive_thresholder")
class AdaptiveThresholder(SoftBitThresholder):
    """Adaptive thresholder for soft bit values.

    Adjusts the threshold based on the statistics of the input signal.
    This can be useful in varying channel conditions where a fixed
    threshold may not be optimal.

    Supports different adaptive threshold methods:
    - 'mean': Uses the mean of the input as threshold
    - 'median': Uses the median of the input as threshold
    - 'otsu': Uses Otsu's method for optimal bimodal threshold
    """

    def __init__(self, method: str = "mean", scale_factor: float = 1.0, input_type: str = "prob", *args: Any, **kwargs: Any):
        """Initialize the adaptive thresholder.

        Args:
            method: Method to use for adaptive thresholding ('mean', 'median', 'otsu').
            scale_factor: Factor to scale the computed threshold.
            input_type: Type of soft input ('prob' or 'llr').
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        valid_methods = ["mean", "median", "otsu"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {method}")

        self.method = method
        self.scale_factor = scale_factor
        self.input_type = input_type

    def _otsu_threshold(self, x: torch.Tensor) -> float:
        """Compute Otsu's threshold for bimodal distribution.

        Otsu's method finds the threshold that minimizes intra-class variance as described
        in :cite:`otsu1979threshold`. This method is particularly effective for signals
        with bimodal distributions.

        Args:
            x: Input tensor of soft bit values.

        Returns:
            Optimal threshold value.
        """
        # Flatten the tensor for histogram calculation
        x_flat = x.flatten()

        # Create histogram (256 bins)
        hist = torch.histc(x_flat, bins=256, min=0.0, max=1.0)
        bin_edges = torch.linspace(0, 1, 257, device=x.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate cumulative sums
        cum_sum = torch.cumsum(hist, dim=0)
        cum_mean = torch.cumsum(hist * bin_centers, dim=0)
        total_mean = cum_mean[-1]
        total_sum = cum_sum[-1]

        # Calculate between-class variance
        between_var = torch.zeros_like(hist)
        for i in range(len(hist)):
            if cum_sum[i] == 0 or cum_sum[i] == total_sum:
                continue

            w0 = cum_sum[i] / total_sum
            w1 = 1.0 - w0

            mu0 = cum_mean[i] / cum_sum[i]
            mu1 = (total_mean - cum_mean[i]) / (total_sum - cum_sum[i])

            between_var[i] = w0 * w1 * (mu0 - mu1) ** 2

        # Find threshold with maximum between-class variance
        max_idx = torch.argmax(between_var)
        return bin_centers[max_idx].item()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply adaptive thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Handle LLR inputs by converting to probability space for thresholding
        if self.input_type == "llr":
            # Convert LLRs to probabilities using sigmoid: P(bit=0) = 1 / (1 + exp(-LLR))
            x_prob = torch.sigmoid(x)
        else:
            x_prob = x

        # Compute the threshold based on the selected method
        if self.method == "mean":
            threshold = x_prob.mean().item() * self.scale_factor
        elif self.method == "median":
            threshold = x_prob.median().item() * self.scale_factor
        elif self.method == "otsu":
            threshold = self._otsu_threshold(x_prob) * self.scale_factor

        # Apply thresholding
        return (x_prob > threshold).float()


@ModelRegistry.register_model("llr_thresholder")
class LLRThresholder(SoftBitThresholder):
    """Specialized thresholder for Log-Likelihood Ratio (LLR) values.

    Handles LLR values properly, optionally applying scaling or other transformations before
    thresholding. For LLRs, positive values favor bit=0, negative values favor bit=1.

    Can also output soft probabilities instead of hard decisions if required.
    """

    def __init__(self, threshold: float = 0.0, confidence_scaling: float = 1.0, output_type: str = "hard", *args: Any, **kwargs: Any):
        """Initialize the LLR thresholder.

        Args:
            threshold: The threshold value to use. Default is 0.0 for LLRs.
            confidence_scaling: Scaling factor applied to LLRs to adjust confidence.
            output_type: Output type, either 'hard' for binary decisions or 'soft' for probabilities.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.threshold = threshold
        self.confidence_scaling = confidence_scaling
        self.output_type = output_type

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Process LLR values to produce bit decisions or probabilities.

        Args:
            x: Input tensor of LLR values.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of bit values, either hard (0.0 or 1.0) or soft (probabilities).
        """
        # Apply confidence scaling to LLRs
        scaled_llrs = x * self.confidence_scaling

        if self.output_type == "hard":
            # For LLRs, negative values favor bit=1, positive values favor bit=0
            # So we flip the comparison (< instead of >) compared to probability thresholding
            return (scaled_llrs < self.threshold).float()
        elif self.output_type == "soft":
            # Convert LLRs to probabilities using sigmoid function
            # P(bit=1) = 1 / (1 + exp(LLR))
            return torch.sigmoid(-scaled_llrs)  # Negative sign because sigmoid maps to P(bit=1)
        else:
            raise ValueError(f"Unsupported output_type: {self.output_type}")


@ModelRegistry.register_model("min_distance_thresholder")
class MinDistanceThresholder(SoftBitThresholder):
    """Thresholder based on minimum distance calculations.

    Uses minimum distance to constellation points to make hard decisions, similar to how
    demodulators work in communication systems.

    This is particularly useful for signals that have been transmitted through a channel and may
    have complex noise characteristics.
    """

    def __init__(self, reference_points: Optional[torch.Tensor] = None, noise_var: float = 1.0, input_type: str = "prob", *args: Any, **kwargs: Any):
        """Initialize the minimum distance thresholder.

        Args:
            reference_points: Reference points for distance calculation (constellation).
                If None, defaults to [0.0, 1.0] for probabilities or [-2.0, 2.0] for LLRs.
            noise_var: Noise variance used in soft distance calculations.
            input_type: Type of soft input ('prob' or 'llr').
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.input_type = input_type
        self.noise_var = noise_var

        # Set default reference points if not provided
        if reference_points is None:
            if input_type == "prob":
                self.reference_points = torch.tensor([0.0, 1.0])
            elif input_type == "llr":
                self.reference_points = torch.tensor([-2.0, 2.0])  # Representative LLR values
            else:
                raise ValueError(f"Unsupported input_type: {input_type}")
        else:
            self.reference_points = reference_points

        # Register reference points as buffer to ensure it moves with the model to device
        self.register_buffer("ref_points", self.reference_points)

    def forward(self, x: torch.Tensor, noise_var: Optional[float] = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply minimum distance thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            noise_var: Optional override for noise variance.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Convert 1D input to 2D for consistent processing
        original_shape = x.shape
        x_reshaped = x.reshape(-1, 1)

        # Calculate distances to reference points
        distances = torch.abs(x_reshaped - self.ref_points.reshape(1, -1)) ** 2

        # Find closest reference point for each input value
        min_indices = torch.argmin(distances, dim=1)

        # Map back to bit values (assuming ref_points[0] maps to bit 0)
        result = min_indices.float()

        # Reshape back to original dimensions
        return result.reshape(original_shape)


@ModelRegistry.register_model("repetition_soft_bit_decoder")
class RepetitionSoftBitDecoder(BaseModel):
    """Enhanced decoder for repetition coding with flexible soft bit processing.

    This decoder processes repeated soft bit values with various thresholding techniques.
    It supports multiple soft input types and different methods for combining repeated values.

    Example:
        With repetition_factor=3, soft_combine_method='mean', and thresholder=FixedThresholder:
        Input [0.2, 0.3, 0.1, 0.8, 0.7, 0.9] becomes [0.0, 1.0]
    """

    def __init__(self, repetition_factor: int = 3, soft_combine_method: str = "mean", thresholder: Optional[SoftBitThresholder] = None, input_type: str = "prob", *args: Any, **kwargs: Any):
        """Initialize the repetition soft bit decoder.

        Args:
            repetition_factor: Number of times each bit was repeated. Must be a positive integer.
            soft_combine_method: Method to combine repeated soft values ('mean', 'sum', 'median', 'max').
            thresholder: Optional custom thresholder. If None, uses FixedThresholder with appropriate defaults.
            input_type: Type of soft input ('prob', 'llr').
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")

        self.repetition_factor = repetition_factor

        valid_combine_methods = ["mean", "sum", "median", "max", "min"]
        if soft_combine_method not in valid_combine_methods:
            raise ValueError(f"Combine method must be one of {valid_combine_methods}, got {soft_combine_method}")

        self.soft_combine_method = soft_combine_method
        self.input_type = input_type

        # Create default thresholder if none is provided
        if thresholder is None:
            if input_type == "prob":
                self.thresholder = FixedThresholder(threshold=0.5, input_type=input_type)
            elif input_type == "llr":
                self.thresholder = LLRThresholder(threshold=0.0, output_type="hard")
            else:
                raise ValueError(f"Unsupported input_type: {input_type}")
        else:
            self.thresholder = thresholder

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the input tensor using soft bit processing.

        Args:
            x: Input tensor of shape (batch_size, encoded_length), where encoded_length =
               original_message_length * repetition_factor. Contains soft bit values.
            *args: Additional positional arguments (passed to thresholder).
            **kwargs: Additional keyword arguments (passed to thresholder).

        Returns:
            Decoded binary tensor of shape (batch_size, encoded_length // repetition_factor)
        """
        batch_size, encoded_length = x.shape
        message_length = encoded_length // self.repetition_factor

        # Reshape to separate the repetition dimension
        reshaped = x.reshape(batch_size, message_length, self.repetition_factor)

        # Combine the repeated values according to the specified method
        if self.soft_combine_method == "mean":
            combined = reshaped.mean(dim=2)
        elif self.soft_combine_method == "sum":
            combined = reshaped.sum(dim=2)
        elif self.soft_combine_method == "median":
            combined, _ = reshaped.median(dim=2)
        elif self.soft_combine_method == "max":
            combined, _ = reshaped.max(dim=2)
        elif self.soft_combine_method == "min":
            combined, _ = reshaped.min(dim=2)

        # Apply thresholding
        return self.thresholder(combined, *args, **kwargs)
