"""Base model definitions for deep learning architectures.

This module provides the foundation for all model implementations in the Kaira framework. The
BaseModel class implements common functionality and enforces a consistent interface across
different model types.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional  # Added imports

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in the Kaira framework.

    This class extends PyTorch's nn.Module and adds framework-specific functionality. All models
    should inherit from this class to ensure compatibility with the framework's training,
    evaluation, and inference pipelines.

    The class provides a consistent interface for model implementation while allowing flexibility
    in architecture design. It enforces proper initialization and forward pass implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the model.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the forward pass computation.

        This method should be implemented by all subclasses to define how input data
        is processed through the model to produce output.

        Args:
            *args: Variable positional arguments for flexible input handling
            **kwargs: Variable keyword arguments for optional parameters

        Returns:
            Any: Model output, type depends on specific implementation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ChannelAwareBaseModel(BaseModel):
    """Abstract base class for models that require channel state information (CSI).

    This class extends BaseModel to standardize how CSI is handled in channel-aware models
    throughout the Kaira framework. It provides utility methods for CSI validation,
    normalization, and transformation, ensuring consistent CSI handling across different
    model implementations.

    CSI typically contains information about the communication channel conditions such as:
    - Signal-to-noise ratio (SNR) in dB
    - Channel gain coefficients
    - Fading characteristics
    - Quality indicators

    All subclasses must implement the forward method with explicit CSI parameter.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the channel-aware model.

        Args:
            *args: Variable positional arguments passed to BaseModel.
            **kwargs: Variable keyword arguments passed to BaseModel.
        """
        super().__init__(*args, **kwargs)

        # CSI configuration
        self._csi_shape_cache: Optional[torch.Size] = None
        self._expected_csi_length: Optional[int] = None

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the forward pass computation with CSI support.

        This method must be implemented by all subclasses to define how input data
        is processed through the model with channel state information.

        All implementations must accept CSI either as a positional argument or
        keyword argument named 'csi'. The exact signature can vary to accommodate
        different model architectures (e.g., single input + CSI, or multiple inputs + CSI).

        Common patterns:
        - forward(self, x, csi, *args, **kwargs) -> single input with CSI
        - forward(self, x, x_side, csi, *args, **kwargs) -> multiple inputs with CSI
        - forward(self, *args, csi=csi, **kwargs) -> CSI as keyword argument

        Args:
            *args: Variable positional arguments for flexible input handling
            **kwargs: Variable keyword arguments, must support 'csi' parameter

        Returns:
            Any: Model output, type depends on specific implementation

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement forward method with CSI support")

    def validate_csi(self, csi: torch.Tensor, expected_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """Validate and ensure CSI tensor is in the correct format.

        Args:
            csi (torch.Tensor): The CSI tensor to validate
            expected_shape (Optional[torch.Size]): Expected shape for the CSI tensor.
                                                  If None, uses cached shape or infers from tensor.

        Returns:
            torch.Tensor: Validated CSI tensor

        Raises:
            ValueError: If CSI tensor is invalid or has incorrect shape
            TypeError: If CSI is not a tensor
        """
        if not isinstance(csi, torch.Tensor):
            raise TypeError(f"CSI must be a torch.Tensor, got {type(csi)}")

        if csi.numel() == 0:
            raise ValueError("CSI tensor cannot be empty")

        if torch.isnan(csi).any():
            raise ValueError("CSI tensor contains NaN values")

        if torch.isinf(csi).any():
            raise ValueError("CSI tensor contains infinite values")

        # Validate shape if expected_shape is provided
        if expected_shape is not None:
            if csi.shape != expected_shape:
                raise ValueError(f"CSI shape mismatch. Expected {expected_shape}, got {csi.shape}")

        # Cache the shape for future validations
        if self._csi_shape_cache is None:
            self._csi_shape_cache = csi.shape

        return csi

    def normalize_csi(self, csi: torch.Tensor, method: str = "minmax", target_range: tuple = (0.0, 1.0)) -> torch.Tensor:
        """Normalize CSI values to a specified range.

        Args:
            csi (torch.Tensor): The CSI tensor to normalize
            method (str): Normalization method. Options: "minmax", "zscore", "none"
            target_range (tuple): Target range for minmax normalization (min, max)

        Returns:
            torch.Tensor: Normalized CSI tensor

        Raises:
            ValueError: If normalization method is not supported
        """
        if method == "none":
            return csi
        elif method == "minmax":
            min_val, max_val = target_range
            # Normalize per batch (along dim=1 for 2D tensors)
            if csi.dim() >= 2:
                csi_min = torch.min(csi, dim=1, keepdim=True)[0]
                csi_max = torch.max(csi, dim=1, keepdim=True)[0]
            else:
                csi_min = torch.min(csi)
                csi_max = torch.max(csi)

            # Avoid division by zero
            range_vals = csi_max - csi_min
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

            normalized = (csi - csi_min) / range_vals
            return normalized * (max_val - min_val) + min_val
        elif method == "zscore":
            # Normalize per batch (along dim=1 for 2D tensors)
            if csi.dim() >= 2:
                mean = torch.mean(csi, dim=1, keepdim=True)
                std = torch.std(csi, dim=1, keepdim=True, unbiased=False)
            else:
                mean = torch.mean(csi)
                std = torch.std(csi, unbiased=False)

            # Avoid division by zero
            std = torch.where(std == 0, torch.ones_like(std), std)

            return (csi - mean) / std
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    def transform_csi(self, csi: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Transform CSI tensor to match target shape requirements.

        Args:
            csi (torch.Tensor): The CSI tensor to transform
            target_shape (torch.Size): Target shape for the CSI tensor

        Returns:
            torch.Tensor: Transformed CSI tensor
        """
        current_shape = csi.shape

        # If shapes already match, return as-is
        if current_shape == target_shape:
            return csi

        # Handle different transformation cases
        if len(target_shape) > len(current_shape):
            # Add dimensions
            while len(csi.shape) < len(target_shape):
                csi = csi.unsqueeze(-1)
        elif len(target_shape) < len(current_shape):
            # Remove dimensions by flattening
            csi = csi.flatten(start_dim=len(target_shape) - 1)

        # Reshape to exact target shape if needed
        if csi.shape != target_shape:
            # Try to reshape, expanding/contracting as needed
            total_elements = csi.numel()
            target_elements = torch.prod(torch.tensor(target_shape)).item()

            if total_elements == target_elements:
                csi = csi.reshape(target_shape)
            else:
                # Pad or truncate to match target size
                if total_elements < target_elements:
                    # Pad with the last value
                    pad_size = int(target_elements - total_elements)
                    pad_value = float(csi.flatten()[-1].item()) if total_elements > 0 else 0.0
                    padding = torch.full((pad_size,), pad_value, device=csi.device, dtype=csi.dtype)
                    csi = torch.cat([csi.flatten(), padding])
                else:
                    # Truncate to target size
                    csi = csi.flatten()[: int(target_elements)]

                csi = csi.reshape(target_shape)

        return csi

    def extract_csi_features(self, csi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract common features from CSI tensor for analysis.

        Args:
            csi (torch.Tensor): The CSI tensor to analyze

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing extracted features
        """
        features = {}

        # Handle complex tensors
        if csi.is_complex():
            # Extract magnitude and phase
            magnitude = torch.abs(csi)
            phase = torch.angle(csi)

            # Basic statistics on magnitude
            features["mean"] = torch.mean(magnitude)
            features["std"] = torch.std(magnitude)
            features["min"] = torch.min(magnitude)
            features["max"] = torch.max(magnitude)

            # Complex-specific features
            features["magnitude"] = torch.mean(magnitude)
            features["phase"] = torch.mean(phase)

            # Phase statistics
            features["phase_std"] = torch.std(phase)

        else:
            # Basic statistics for real tensors
            features["mean"] = torch.mean(csi)
            features["std"] = torch.std(csi)
            features["min"] = torch.min(csi)
            features["max"] = torch.max(csi)

        # Shape information
        features["shape"] = torch.tensor(csi.shape)
        features["numel"] = torch.tensor(csi.numel())

        # Signal quality indicators (assuming CSI represents SNR in dB)
        if csi.dim() > 0 and not csi.is_complex():
            # Convert dB to linear scale for additional metrics
            linear_csi = 10 ** (csi / 10)
            features["linear_mean"] = torch.mean(linear_csi)
            features["linear_std"] = torch.std(linear_csi)

        return features

    def forward_csi_to_submodules(self, csi: torch.Tensor, modules: List[BaseModel], *args, **kwargs) -> List[Any]:
        """Helper method to consistently pass CSI to submodules.

        This method facilitates passing CSI to multiple submodules that require
        channel state information, ensuring consistent handling across the model.

        Args:
            csi (torch.Tensor): Channel state information tensor
            modules (List[BaseModel]): List of modules to apply
            *args: Positional arguments to pass to modules
            **kwargs: Keyword arguments to pass to modules

        Returns:
            List[Any]: List of outputs from each module
        """
        outputs = []
        current_input = args[0] if args else None

        for module in modules:
            if isinstance(module, ChannelAwareBaseModel):
                # Module requires CSI - pass it explicitly
                if current_input is not None:
                    output = module(current_input, csi=csi, **kwargs)
                else:
                    output = module(csi=csi, *args, **kwargs)
            else:
                # Standard module - pass without CSI
                if current_input is not None:
                    output = module(current_input, **kwargs)
                else:
                    output = module(*args, **kwargs)

            outputs.append(output)
            current_input = output  # Chain outputs for sequential processing

        return outputs

    def create_csi_for_submodules(self, csi: torch.Tensor, num_modules: int) -> List[torch.Tensor]:
        """Create appropriate CSI tensors for multiple submodules.

        Args:
            csi (torch.Tensor): Original CSI tensor
            num_modules (int): Number of submodules that need CSI

        Returns:
            List[torch.Tensor]: List of CSI tensors for each submodule
        """
        if num_modules <= 0:
            return []

        # For now, return the same CSI for all modules
        # This can be extended to create module-specific CSI transformations
        return [csi.clone() for _ in range(num_modules)]

    @staticmethod
    def extract_csi_from_channel_output(channel_output: Any) -> Optional[torch.Tensor]:
        """Extract CSI from channel output if available.

        Some channels return both the transmitted signal and CSI information.
        This static method provides a standardized way to extract CSI from
        various channel output formats.

        Args:
            channel_output: Output from a channel, which may contain CSI

        Returns:
            Optional[torch.Tensor]: Extracted CSI tensor if available, None otherwise
        """
        if isinstance(channel_output, tuple):
            # Assume tuple format: (signal, csi)
            if len(channel_output) >= 2:
                return channel_output[1]
        elif isinstance(channel_output, dict):
            # Dictionary format with CSI key
            return channel_output.get("csi")
        elif hasattr(channel_output, "csi"):
            # Object with CSI attribute
            return getattr(channel_output, "csi")

        return None

    @staticmethod
    def format_csi_for_channel(csi: torch.Tensor, channel_format: str = "tensor") -> Any:
        """Format CSI tensor for passing to channels that expect specific formats.

        Args:
            csi (torch.Tensor): CSI tensor to format
            channel_format (str): Expected format ("tensor", "dict", "kwargs")

        Returns:
            Any: Formatted CSI in the requested format
        """
        if channel_format == "tensor":
            return csi
        elif channel_format == "dict":
            return {"csi": csi}
        elif channel_format == "kwargs":
            return {"csi": csi}
        else:
            raise ValueError(f"Unsupported channel format: {channel_format}")


class ConfigurableModel(BaseModel):
    """Model that supports dynamically adding and removing steps.

    This class extends the basic model functionality with methods to add, remove, and manage model
    steps during runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the configurable model."""
        super().__init__(*args, **kwargs)
        self.steps: List[Callable] = []  # Added initialization here, changed type to List[Callable]

    def add_step(self, step: Callable) -> "ConfigurableModel":  # Changed step type to Callable
        """Add a processing step to the model.

        Args:
            step: A callable that will be added to the processing pipeline.
                Must accept and return tensor-like objects.

        Returns:
            Self for method chaining
        """
        if not callable(step):  # Added check
            raise TypeError("Step must be callable")
        self.steps.append(step)
        return self

    def remove_step(self, index: int) -> "ConfigurableModel":
        """Remove a processing step from the model.

        Args:
            index: The index of the step to remove

        Returns:
            Self for method chaining

        Raises:
            IndexError: If the index is out of range
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self.steps)-1})")
        self.steps.pop(index)
        return self

    def forward(self, input_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Process input through all steps sequentially.

        Args:
            input_data (Any): The input to process
            *args (Any): Positional arguments passed to each step
            **kwargs (Any): Additional keyword arguments passed to each step

        Returns:
            The result after applying all steps
        """
        result = input_data
        for step in self.steps:
            result = step(result, *args, **kwargs)
        return result
