"""Soft bit thresholding module for binary data processing.

This module provides various thresholding techniques for converting soft bit representations
(probabilities, LLRs, etc.) to hard decisions. These thresholders can be used with soft decoders or
as standalone components in signal processing pipelines.

Soft bit processing is crucial in modern communication systems to extract maximum information from
the received signals. The techniques implemented here are based on established methods in
communication theory.
"""

from enum import Enum
from typing import Any, List, Optional, Union

import torch

from kaira.models.registry import ModelRegistry

from ..base import BaseModel


class InputType(str, Enum):
    """Enumeration of supported input types for soft bit thresholders.

    This enum defines the format of input values that soft bit thresholders can accept.
    """

    PROBABILITY = "prob"  #: Input values are probabilities in the range [0, 1]
    LLR = "llr"  #: Input values are log-likelihood ratios
    SOFT = "soft"  #: Input values are in a general soft format


class OutputType(str, Enum):
    """Enumeration of supported output types for soft bit thresholders.

    This enum defines the format of output values that soft bit thresholders can produce.
    """

    HARD = "hard"  #: Output values are binary decisions (0 or 1)
    SOFT = "soft"  #: Output values are soft probabilities in the range [0, 1]


class SoftBitThresholder(BaseModel):
    """Base class for soft bit thresholding techniques.

    This abstract class defines the interface for soft bit thresholders that convert soft bit
    representations (e.g., probabilities, LLRs) to hard binary decisions.

    Soft bit thresholding is a key technique in modern communication systems for extracting
    reliable information from noisy channel outputs.

    Implementers must override the forward method.
    """

    def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[Union[str, torch.device]] = None, *args: Any, **kwargs: Any):
        """Initialize the soft bit thresholder.

        Args:
            dtype: The data type for tensors used by this model.
            device: The device (CPU/CUDA) where tensors should be allocated.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.device = device

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "SoftBitThresholder":
        """Move the model to the specified device.

        Args:
            device: The device to move the model to.
            *args: Additional positional arguments for nn.Module.to().
            **kwargs: Additional keyword arguments for nn.Module.to().

        Returns:
            Self for method chaining.
        """
        self.device = device
        return super().to(device, *args, **kwargs)

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

    def __init__(self, threshold: float = 0.5, input_type: InputType = InputType.PROBABILITY, *args: Any, **kwargs: Any):
        """Initialize the fixed thresholder.

        Args:
            threshold: The threshold value to use. Default is 0.5 for probabilities.
            input_type: Type of soft input, can be 'prob' (probabilities between 0 and 1) or
                       'llr' (log-likelihood ratios). Affects the default threshold if not specified.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

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
        if self.input_type == InputType.PROBABILITY:
            # For probability values (between 0 and 1)
            return (x > self.threshold).float()
        elif self.input_type == InputType.LLR:
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

    def __init__(self, method: str = "mean", scale_factor: float = 1.0, input_type: InputType = InputType.PROBABILITY, *args: Any, **kwargs: Any):
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

        This implementation is optimized for performance with PyTorch operations.

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

        # Calculate class probabilities for all possible thresholds
        weight1 = torch.cumsum(hist, dim=0)
        weight2 = weight1[-1] - weight1

        # Ensure no division by zero
        zero_mask = (weight1 == 0) | (weight2 == 0)

        # Calculate class means for all possible thresholds
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / torch.clamp(weight1, min=1e-10)

        # Calculate total mean
        total_mean = (hist * bin_centers).sum() / torch.clamp(weight1[-1], min=1e-10)

        # Calculate mean2 from total_mean and mean1
        # mean2 = (total_sum - sum1) / weight2
        mean2 = (total_mean * weight1[-1] - weight1 * mean1) / torch.clamp(weight2, min=1e-10)

        # Calculate between-class variance
        variance = weight1 * weight2 * (mean1 - mean2) ** 2

        # Set variance to 0 where either class has no elements
        variance[zero_mask] = 0

        # Find threshold with maximum between-class variance
        max_idx = torch.argmax(variance)
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
        if self.input_type == InputType.LLR:
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

    def __init__(self, threshold: float = 0.0, confidence_scaling: float = 1.0, output_type: OutputType = OutputType.HARD, *args: Any, **kwargs: Any):
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

        if self.output_type == OutputType.HARD:
            # For LLRs, negative values favor bit=1, positive values favor bit=0
            # So we flip the comparison (< instead of >) compared to probability thresholding
            return (scaled_llrs < self.threshold).float()
        elif self.output_type == OutputType.SOFT:
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

    def __init__(self, reference_points: Optional[torch.Tensor] = None, noise_var: float = 1.0, input_type: InputType = InputType.PROBABILITY, *args: Any, **kwargs: Any):
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
            if input_type == InputType.PROBABILITY:
                self.reference_points = torch.tensor([0.0, 1.0])
            elif input_type == InputType.LLR:
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

    def __init__(self, repetition_factor: int = 3, soft_combine_method: str = "mean", thresholder: Optional[SoftBitThresholder] = None, input_type: InputType = InputType.PROBABILITY, *args: Any, **kwargs: Any):
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
            if input_type == InputType.PROBABILITY:
                self.thresholder = FixedThresholder(threshold=0.5, input_type=input_type)
            elif input_type == InputType.LLR:
                self.thresholder = LLRThresholder(threshold=0.0, output_type=OutputType.HARD)
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


@ModelRegistry.register_model("hysteresis_thresholder")
class HysteresisThresholder(SoftBitThresholder):
    """Thresholder with hysteresis for robust decision making in noisy environments.

    Uses two thresholds to create a hysteresis effect, providing more stable
    decisions for values near the decision boundary. Values must cross a higher
    threshold to transition from 0→1, and a lower threshold to transition from 1→0.

    This approach reduces oscillations in the output when the input signal is noisy
    or fluctuating around the threshold.

    Example:
        With high_threshold=0.6, low_threshold=0.4:
        - Values > 0.6 are classified as 1
        - Values < 0.4 are classified as 0
        - Values between 0.4 and 0.6 maintain their previous state
    """

    def __init__(self, high_threshold: float = 0.6, low_threshold: float = 0.4, input_type: InputType = InputType.PROBABILITY, initial_state: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any):
        """Initialize the hysteresis thresholder.

        Args:
            high_threshold: Threshold to cross for transitioning from 0→1.
            low_threshold: Threshold to cross for transitioning from 1→0.
            input_type: Type of soft input ('prob' or 'llr').
            initial_state: Optional tensor with initial states. If None,
                           all values start at state 0.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if high_threshold < low_threshold:
            raise ValueError(f"high_threshold ({high_threshold}) must be >= low_threshold ({low_threshold})")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.input_type = input_type
        self._state = initial_state

    def reset_state(self, initial_state: Optional[torch.Tensor] = None):
        """Reset the internal state of the hysteresis thresholder.

        Args:
            initial_state: Optional tensor with initial states. If None,
                          state is set to None and will be initialized on
                          the first forward pass.
        """
        self._state = initial_state

    def forward(self, x: torch.Tensor, reset_state: bool = False, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply hysteresis thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            reset_state: If True, internal state is reset before processing.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Reset state if requested
        if reset_state:
            self._state = None

        # Convert LLR to probability if needed
        if self.input_type == InputType.LLR:
            # Use sigmoid to convert LLR to probability
            x_prob = torch.sigmoid(-x)  # Negative sign because sigmoid maps to P(bit=1)
        else:
            x_prob = x

        # Initialize state if needed
        if self._state is None or self._state.shape != x_prob.shape:
            self._state = torch.zeros_like(x_prob)

        # Apply hysteresis thresholding
        # Where x > high_threshold, state becomes 1
        high_mask = x_prob > self.high_threshold
        # Where x < low_threshold, state becomes 0
        low_mask = x_prob < self.low_threshold

        # Update state based on thresholds
        new_state = self._state.clone()
        new_state[high_mask] = 1.0
        new_state[low_mask] = 0.0

        # Store the new state
        self._state = new_state

        return new_state


@ModelRegistry.register_model("weighted_thresholder")
class WeightedThresholder(SoftBitThresholder):
    """Thresholder that applies weights to input values before thresholding.

    This thresholder allows applying non-uniform weights to different parts of the input tensor,
    which is useful for systems where some bits are more reliable or important than others.

    Example:
        With weights=[1.0, 0.8, 0.5], threshold=0.6:
        Input [0.7, 0.7, 0.7] becomes [1.0, 0.0, 0.0] after weighting.
    """

    def __init__(self, weights: Union[torch.Tensor, List[float], float], threshold: float = 0.5, input_type: InputType = InputType.PROBABILITY, normalize_weights: bool = False, *args: Any, **kwargs: Any):
        """Initialize the weighted thresholder.

        Args:
            weights: Weights to apply to input values. Can be a tensor, list, or scalar.
            threshold: Threshold value to apply after weighting.
            input_type: Type of soft input values.
            normalize_weights: If True, weights are normalized to sum to 1.0.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # Convert weights to tensor if needed
        if not isinstance(weights, torch.Tensor):
            if isinstance(weights, (list, tuple)):
                weights = torch.tensor(weights, dtype=self.dtype)
            else:
                # Scalar weight, will be broadcast during use
                weights = torch.tensor([weights], dtype=self.dtype)

        # Normalize weights if requested
        if normalize_weights and weights.numel() > 1:
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)
        self.threshold = threshold
        self.input_type = input_type

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply weighted thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Handle LLR inputs by converting to probability space
        if self.input_type == InputType.LLR:
            # Convert LLRs to probabilities using sigmoid
            x_prob = torch.sigmoid(-x)  # Negative sign for P(bit=1)
        else:
            x_prob = x

        # Apply weights
        if self.weights.numel() == 1:
            # Scalar weight case
            weighted = x_prob * self.weights.item()
        else:
            # Ensure weights can be broadcast properly
            # Try to match the shape by expanding dims if needed
            weights = self.weights

            # Handle common case: 1D weights applied to 2D batch
            if len(weights.shape) == 1 and len(x_prob.shape) > 1:
                # Reshape for broadcasting across batches (assume batch is dim 0)
                reshape_dims = [1] * len(x_prob.shape)
                reshape_dims[1] = -1  # Feature dimension is typically dim 1
                weights = weights.view(*reshape_dims)

            weighted = x_prob * weights

        # Apply threshold
        return (weighted > self.threshold).float()


@ModelRegistry.register_model("ensemble_thresholder")
class SoftBitEnsembleThresholder(SoftBitThresholder):
    """Ensemble thresholder that combines decisions from multiple thresholders.

    This thresholder aggregates the outputs of multiple thresholding approaches
    using various voting schemes to produce a more robust decision. This can be
    particularly effective in challenging noise conditions or when the optimal
    thresholding strategy is unclear.

    Example:
        With thresholders=[FixedThresholder(), AdaptiveThresholder()] and voting='majority':
        The output will be 1.0 only if both thresholders output 1.0.
    """

    def __init__(self, thresholders: List[SoftBitThresholder], voting: str = "majority", weights: Optional[Union[torch.Tensor, List[float]]] = None, *args: Any, **kwargs: Any):
        """Initialize the ensemble thresholder.

        Args:
            thresholders: List of thresholders to combine.
            voting: Voting strategy to use: 'majority', 'weighted', 'any', or 'all'.
            weights: Optional weights for each thresholder (used with 'weighted' voting).
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if not thresholders:
            raise ValueError("At least one thresholder must be provided")

        valid_voting_methods = ["majority", "weighted", "any", "all"]
        if voting not in valid_voting_methods:
            raise ValueError(f"Voting method must be one of {valid_voting_methods}, got {voting}")

        self.thresholders = torch.nn.ModuleList(thresholders)
        self.voting = voting

        # Configure weights for weighted voting
        if weights is not None:
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)

            if len(weights) != len(thresholders):
                raise ValueError(f"Number of weights ({len(weights)}) must match " f"number of thresholders ({len(thresholders)})")

            # Normalize weights to sum to 1.0
            weights = weights / weights.sum()

        self.register_buffer("weights", weights if weights is not None else torch.ones(len(thresholders)))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply ensemble thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            *args: Additional positional arguments passed to thresholders.
            **kwargs: Additional keyword arguments passed to thresholders.

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Collect outputs from all thresholders
        outputs = []
        for thresholder in self.thresholders:
            outputs.append(thresholder(x, *args, **kwargs))

        # Stack outputs for efficient operations
        stacked_outputs = torch.stack(outputs, dim=0)

        # Apply the selected voting strategy
        if self.voting == "majority":
            # Count votes and compare with half the number of thresholders
            vote_sum = stacked_outputs.sum(dim=0)
            return (vote_sum >= len(self.thresholders) / 2.0).float()

        elif self.voting == "weighted":
            # Apply weights to each thresholder's output
            weighted_votes = (stacked_outputs * self.weights.view(-1, 1, 1)).sum(dim=0)
            return (weighted_votes >= 0.5).float()

        elif self.voting == "any":
            # Return 1.0 if any thresholder outputs 1.0
            return (stacked_outputs.sum(dim=0) > 0).float()

        elif self.voting == "all":
            # Return 1.0 only if all thresholders output 1.0
            return (stacked_outputs.sum(dim=0) == len(self.thresholders)).float()

        # Should never reach here due to validation in __init__
        raise ValueError(f"Unsupported voting method: {self.voting}")


@ModelRegistry.register_model("dynamic_thresholder")
class DynamicThresholder(SoftBitThresholder):
    """Thresholder with dynamically adjusting threshold for non-stationary signals.

    This thresholder adapts to changing signal conditions over time using
    exponential moving averages. It's particularly useful for systems with
    time-varying noise or signal characteristics.

    The dynamic threshold is computed as a weighted average of past input statistics
    and can adapt to gradual changes in the signal distribution.

    Example:
        With decay=0.9, initial_threshold=0.5:
        The threshold will gradually adapt to the mean of the input signal.
    """

    def __init__(self, decay: float = 0.9, initial_threshold: float = 0.5, input_type: InputType = InputType.PROBABILITY, adaptation_method: str = "mean", bias: float = 0.0, min_threshold: float = 0.1, max_threshold: float = 0.9, *args: Any, **kwargs: Any):
        """Initialize the dynamic thresholder.

        Args:
            decay: Exponential decay factor (0-1) controlling adaptation speed.
                Higher values make adaptation slower but more stable.
            initial_threshold: Starting threshold value.
            input_type: Type of input values ('prob' or 'llr').
            adaptation_method: Method to adapt threshold ('mean', 'median', 'percentile').
            bias: Fixed bias to add to computed threshold.
            min_threshold: Minimum allowed threshold value.
            max_threshold: Maximum allowed threshold value.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Decay must be between 0 and 1, got {decay}")

        valid_methods = ["mean", "median", "percentile"]
        if adaptation_method not in valid_methods:
            raise ValueError(f"Adaptation method must be one of {valid_methods}, got {adaptation_method}")

        self.decay = decay
        self.threshold = initial_threshold
        self.input_type = input_type
        self.adaptation_method = adaptation_method
        self.bias = bias
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Initialize running statistics
        self.running_mean = initial_threshold
        self.running_variance = 0.0

    def reset_stats(self, initial_threshold: Optional[float] = None):
        """Reset the running statistics.

        Args:
            initial_threshold: New initial threshold to use. If None,
                               keeps the current threshold.
        """
        if initial_threshold is not None:
            self.threshold = initial_threshold
            self.running_mean = initial_threshold

        self.running_variance = 0.0

    def forward(self, x: torch.Tensor, reset: bool = False, percentile: float = 50.0, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply dynamic thresholding to convert soft bit values to hard decisions.

        Args:
            x: Input tensor of soft bit values.
            reset: If True, reset running statistics.
            percentile: Percentile to use if adaptation_method is 'percentile'.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Tensor of hard bit decisions (0.0 or 1.0).
        """
        # Reset statistics if requested
        if reset:
            self.reset_stats()

        # Convert to probability space if needed
        if self.input_type == InputType.LLR:
            x_prob = torch.sigmoid(-x)  # Negative sign for P(bit=1)
        else:
            x_prob = x

        # Update running statistics
        batch_mean = x_prob.mean().item()
        batch_var = x_prob.var().item()

        # Update running mean with exponential decay
        self.running_mean = self.decay * self.running_mean + (1 - self.decay) * batch_mean
        # Update running variance
        self.running_variance = self.decay * self.running_variance + (1 - self.decay) * batch_var

        # Compute dynamic threshold based on selected method
        if self.adaptation_method == "mean":
            # Use running mean directly
            new_threshold = self.running_mean + self.bias

        elif self.adaptation_method == "median":
            # Estimate median using running statistics
            # For many distributions, median ≈ mean
            new_threshold = self.running_mean + self.bias

        elif self.adaptation_method == "percentile":
            # Estimate percentile using running statistics
            # For normal distribution, percentiles can be approximated
            z_score = torch.tensor((percentile - 50) / 100 * 3.0)  # Map percentile to z-score
            new_threshold = self.running_mean + z_score * torch.sqrt(torch.tensor(self.running_variance)) + self.bias

        # Apply threshold limits
        self.threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        # Apply threshold to input
        return (x_prob > self.threshold).float()


__all__ = [
    "InputType",
    "OutputType",
    "SoftBitThresholder",
    "FixedThresholder",
    "AdaptiveThresholder",
    "LLRThresholder",
    "MinDistanceThresholder",
    "RepetitionSoftBitDecoder",
    "HysteresisThresholder",
    "WeightedThresholder",
    "SoftBitEnsembleThresholder",
    "DynamicThresholder",
]
