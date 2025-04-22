"""Base model definitions for deep learning architectures.

This module provides the foundation for all model implementations in the Kaira framework. The
BaseModel class implements common functionality and enforces a consistent interface across
different model types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
        self.steps: List[Any] = []

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


class CSIFormat(Enum):
    """Enumeration of different CSI (Channel State Information) formats.

    This enum represents the different formats that CSI data can take in channel-aware models.
    """

    LINEAR = "linear"  # Linear scale (e.g., 0.001 to 10.0)
    DB = "db"  # Decibel scale (e.g., -30 to 10 dB)
    NORMALIZED = "normalized"  # Normalized to a specific range (e.g., 0.0 to 1.0)
    COMPLEX = "complex"  # Complex-valued CSI with magnitude and phase
    VECTOR = "vector"  # Multi-dimensional vector of channel coefficients
    MATRIX = "matrix"  # Matrix representation (e.g., for MIMO channels)


class ChannelAwareBaseModel(BaseModel, ABC):
    """Abstract base class for channel-aware models that require CSI.
    
    This class standardizes how Channel State Information (CSI) is handled
    in channel-aware models, ensuring that CSI is explicitly required
    rather than being an optional parameter. It provides utilities for
    validating, normalizing, and transforming CSI data to ensure consistent
    usage across different model implementations.
    
    Channel-aware models are neural networks that adapt their processing based
    on the current state of the communication channel, which may include properties
    like signal-to-noise ratio (SNR), fading coefficients, or other channel quality
    indicators.
    
    Attributes:
        expected_csi_dims (Tuple[int, ...]): Expected dimensions for CSI tensor
        expected_csi_format (CSIFormat): Expected format for CSI values
        csi_min_value (float): Minimum expected value for valid CSI
        csi_max_value (float): Maximum expected value for valid CSI
        auto_normalize_csi (bool): Whether to automatically normalize CSI in forward pass
        strict_validation (bool): Whether to raise errors for invalid CSI or silently fix
    """
    
    def __init__(
        self,
        expected_csi_dims: Optional[Tuple[int, ...]] = None,
        expected_csi_format: CSIFormat = CSIFormat.LINEAR,
        csi_min_value: float = float('-inf'),
        csi_max_value: float = float('inf'),
        auto_normalize_csi: bool = True,
        strict_validation: bool = True,
        *args: Any,
        **kwargs: Any
    ):
        """Initialize the channel-aware model.
        
        Args:
            expected_csi_dims (Optional[Tuple[int, ...]]): Expected dimensions for CSI tensor.
                If None, dimensions are not validated.
            expected_csi_format (CSIFormat): Expected format for CSI values.
                Defaults to CSIFormat.LINEAR.
            csi_min_value (float): Minimum expected value for valid CSI.
                Defaults to negative infinity.
            csi_max_value (float): Maximum expected value for valid CSI.
                Defaults to positive infinity.
            auto_normalize_csi (bool): Whether to automatically normalize CSI in forward pass.
                Defaults to True.
            strict_validation (bool): Whether to raise errors for invalid CSI or silently fix.
                Defaults to True (raise errors on invalid CSI).
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.expected_csi_dims = expected_csi_dims
        self.expected_csi_format = expected_csi_format
        self.csi_min_value = csi_min_value
        self.csi_max_value = csi_max_value
        self.auto_normalize_csi = auto_normalize_csi
        self.strict_validation = strict_validation
        self._last_csi = None  # Cache for debugging and visualization
    
    @abstractmethod
    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Transform the input using channel state information.
        
        Args:
            x (torch.Tensor): The input tensor to process
            csi (torch.Tensor): Channel state information tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Processed output tensor
        """
        # Implementations should first normalize CSI if auto_normalize_csi is False:
        # if not self.auto_normalize_csi:
        #    csi = self.get_normalized_csi(csi)
        pass
    
    def _validate_csi_impl(self, csi: torch.Tensor) -> tuple[bool, Optional[str]]:
        """Implementation of CSI validation logic.
        
        Args:
            csi (torch.Tensor): The CSI tensor to validate
            
        Returns:
            tuple[bool, Optional[str]]: Validation result and error message if invalid
        """
        # Basic type checking
        if not isinstance(csi, torch.Tensor):
            return False, f"CSI must be a torch.Tensor, got {type(csi)}"
        
        # Check for NaNs or infinities
        if torch.isnan(csi).any():
            return False, "CSI contains NaN values"
            
        if torch.isinf(csi).any():
            return False, "CSI contains infinite values"
        
        # Dimension validation
        if self.expected_csi_dims is not None:
            if len(csi.shape) != len(self.expected_csi_dims):
                return False, (f"CSI dimensions mismatch: expected {len(self.expected_csi_dims)} "
                              f"dimensions, got {len(csi.shape)}")
                
            # Check if dimensions match, ignoring batch size (first dim)
            for i in range(1, len(csi.shape)):
                if i < len(self.expected_csi_dims) and self.expected_csi_dims[i] != -1:
                    if csi.shape[i] != self.expected_csi_dims[i]:
                        return False, (f"CSI shape mismatch at dimension {i}: expected "
                                      f"{self.expected_csi_dims[i]}, got {csi.shape[i]}")
        
        # Value range validation
        if not torch.is_complex(csi):  # Only check range for real-valued CSI
            if (csi < self.csi_min_value).any():
                return False, f"CSI contains values below minimum {self.csi_min_value}"
                
            if (csi > self.csi_max_value).any():
                return False, f"CSI contains values above maximum {self.csi_max_value}"
        
        return True, None
        
    def validate_csi(self, csi: torch.Tensor) -> bool:
        """Validate that the CSI tensor meets this model's requirements.
        
        Args:
            csi (torch.Tensor): The CSI tensor to validate
            
        Returns:
            bool: True if CSI is valid, False otherwise
            
        Raises:
            ValueError: If CSI is severely invalid and strict_validation is True
        """
        valid, error_message = self._validate_csi_impl(csi)
        
        if not valid and self.strict_validation and error_message:
            raise ValueError(f"CSI validation error: {error_message}")
        
        return valid
    
    def get_normalized_csi(self, csi: torch.Tensor) -> torch.Tensor:
        """Get properly normalized CSI according to model requirements.
        
        This is a convenience wrapper that validates and normalizes CSI
        in a single call, with appropriate error handling based on
        the strict_validation setting.
        
        Args:
            csi (torch.Tensor): The input CSI tensor
            
        Returns:
            torch.Tensor: Normalized CSI tensor
            
        Raises:
            ValueError: If CSI validation fails and strict_validation is True
        """
        try:
            if not self.validate_csi(csi):
                return self.normalize_csi(csi)
            return csi
        except ValueError as e:
            if self.strict_validation:
                raise
            # If not strict, try to recover and normalize anyway
            return self.normalize_csi(csi)
    
    def normalize_csi(self, csi: torch.Tensor) -> torch.Tensor:
        """Normalize CSI to the format expected by this model.
        
        Args:
            csi (torch.Tensor): The CSI tensor to normalize
            
        Returns:
            torch.Tensor: Normalized CSI tensor
        """
        # Handle non-tensor input gracefully
        if not isinstance(csi, torch.Tensor):
            try:
                csi = torch.tensor(csi, dtype=torch.float32)
            except:
                raise ValueError(f"Cannot convert CSI of type {type(csi)} to tensor")
        
        # Handle NaNs and infs by replacing with reasonable values
        if torch.isnan(csi).any() or torch.isinf(csi).any():
            csi = torch.nan_to_num(csi, nan=0.0, posinf=self.csi_max_value, neginf=self.csi_min_value)
            
        # Convert to expected format if needed
        if self.expected_csi_format == CSIFormat.LINEAR:
            # Detect if input might be in dB based on negative values or range
            if csi.min() < 0 or (csi.max() <= 30 and csi.min() >= -30):
                csi = self._db_to_linear(csi)
                
        elif self.expected_csi_format == CSIFormat.DB:
            # Detect if input might be in linear scale (all positive, potentially large values)
            if csi.min() >= 0 and csi.max() > 30:
                csi = self._linear_to_db(csi)
                
        elif self.expected_csi_format == CSIFormat.NORMALIZED:
            # Normalize to [0, 1] range
            csi = self._normalize_to_range(csi, 0.0, 1.0)
                
        elif self.expected_csi_format == CSIFormat.COMPLEX:
            # If we have real values but expect complex
            if not torch.is_complex(csi):
                csi = torch.complex(csi, torch.zeros_like(csi))
        
        # Reshape if dimensions don't match expected
        if self.expected_csi_dims is not None and len(csi.shape) != len(self.expected_csi_dims):
            # Try to reshape to expected dimensions
            try:
                # Keep batch dimension, reshape rest to match expected
                if csi.numel() == csi.shape[0] * torch.prod(torch.tensor(self.expected_csi_dims[1:])):
                    new_shape = (csi.shape[0],) + self.expected_csi_dims[1:]
                    csi = csi.reshape(new_shape)
            except:
                # If reshape fails, leave as is and let validation handle it
                pass
                
        # Clamp values to expected range for real-valued tensors
        if not torch.is_complex(csi):
            csi = torch.clamp(csi, min=self.csi_min_value, max=self.csi_max_value)
            
        # Cache the normalized CSI for debugging
        self._last_csi = csi
            
        return csi
    
    def _db_to_linear(self, csi_db: torch.Tensor) -> torch.Tensor:
        """Convert CSI from dB to linear scale.
        
        Args:
            csi_db (torch.Tensor): CSI in decibels
            
        Returns:
            torch.Tensor: CSI in linear scale
        """
        return 10.0 ** (csi_db / 10.0)
    
    def _linear_to_db(self, csi_linear: torch.Tensor) -> torch.Tensor:
        """Convert CSI from linear to dB scale.
        
        Args:
            csi_linear (torch.Tensor): CSI in linear scale
            
        Returns:
            torch.Tensor: CSI in decibels
        """
        # Add small epsilon to prevent log of zero
        return 10.0 * torch.log10(csi_linear + 1e-10)
    
    def _normalize_to_range(
        self, 
        csi: torch.Tensor, 
        target_min: float = 0.0, 
        target_max: float = 1.0
    ) -> torch.Tensor:
        """Normalize CSI values to a target range.
        
        Args:
            csi (torch.Tensor): CSI tensor to normalize
            target_min (float): Target minimum value
            target_max (float): Target maximum value
            
        Returns:
            torch.Tensor: Normalized CSI
        """
        csi_min = csi.min()
        csi_max = csi.max()
        
        if csi_min == csi_max:
            return torch.ones_like(csi) * target_min
        
        normalized = (csi - csi_min) / (csi_max - csi_min)
        return normalized * (target_max - target_min) + target_min
    
    def extract_csi_from_channel_output(
        self, 
        channel_output: Union[Dict[str, Any], torch.Tensor]
    ) -> torch.Tensor:
        """Extract CSI from a channel's output dictionary or tensor.
        
        This utility helps standardize how CSI is extracted from channel outputs,
        which may include the transformed signal along with CSI and other metadata.
        
        Args:
            channel_output (Union[Dict[str, Any], torch.Tensor]): 
                Output from a channel, either as a dictionary or tensor
            
        Returns:
            torch.Tensor: Extracted CSI tensor
            
        Raises:
            ValueError: If CSI cannot be found in the channel output
        """
        # If output is already a tensor, return it as is (assuming it's the CSI)
        if isinstance(channel_output, torch.Tensor):
            return channel_output
            
        # Look for CSI in common keys for dictionary output
        if isinstance(channel_output, dict):
            # Check most common keys first for efficiency
            for key in ["csi", "h", "snr", "channel_state", "channel_coefficients", "channel_info"]:
                if key in channel_output:
                    return channel_output[key]
                    
            # Try case-insensitive search as fallback
            lowercase_keys = {k.lower(): k for k in channel_output.keys()}
            for key in ["csi", "h", "snr", "channel_state"]:
                if key in lowercase_keys:
                    return channel_output[lowercase_keys[key]]
        
        # If we didn't find any recognized CSI keys
        raise ValueError(
            "Could not extract CSI from channel output. Channel output must contain "
            "one of: 'csi', 'h', 'channel_state', 'channel_coefficients', 'snr', or be a tensor."
        )
    
    def format_csi_for_submodules(
        self, 
        csi: torch.Tensor, 
        submodule: nn.Module
    ) -> torch.Tensor:
        """Format CSI appropriately for a specific submodule.
        
        Transforms CSI to match the requirements of a particular submodule,
        based on its type and expected format.
        
        Args:
            csi (torch.Tensor): The original CSI tensor
            submodule (nn.Module): The submodule that will receive the CSI
            
        Returns:
            torch.Tensor: Formatted CSI appropriate for the submodule
        """
        # For ChannelAwareBaseModel submodules, use their expected format
        if isinstance(submodule, ChannelAwareBaseModel):
            if submodule.expected_csi_format != self.expected_csi_format:
                if submodule.expected_csi_format == CSIFormat.LINEAR:
                    return self._db_to_linear(csi)
                elif submodule.expected_csi_format == CSIFormat.DB:
                    return self._linear_to_db(csi)
                elif submodule.expected_csi_format == CSIFormat.NORMALIZED:
                    return self._normalize_to_range(csi, 0.0, 1.0)
                elif submodule.expected_csi_format == CSIFormat.COMPLEX and not torch.is_complex(csi):
                    return torch.complex(csi, torch.zeros_like(csi))
                
            # If dimensions don't match but format does, try to reshape
            if (submodule.expected_csi_dims is not None and 
                csi.dim() != len(submodule.expected_csi_dims)):
                # Try to adapt dimensions (only if batch size stays the same)
                try:
                    new_shape = [csi.shape[0]]  # Keep batch dimension
                    for dim in submodule.expected_csi_dims[1:]:
                        new_shape.append(dim if dim != -1 else 1)
                    return csi.reshape(new_shape)
                except:
                    # If reshape fails, return as is
                    pass
        
        # For other modules, return as is
        return csi
    
    def forward_csi_to_sequential(
        self, 
        x: torch.Tensor, 
        modules: List[nn.Module], 
        csi: torch.Tensor, 
        *args: Any, 
        **kwargs: Any
    ) -> torch.Tensor:
        """Forward input through a sequence of modules with CSI.
        
        This utility helps standardize how CSI is forwarded through sequential
        modules in a channel-aware model. It handles both channel-aware and
        regular modules appropriately.
        
        Args:
            x (torch.Tensor): Input tensor
            modules (List[nn.Module]): List of modules to process input sequentially
            csi (torch.Tensor): Channel state information
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Output after sequential processing
        """
        result = x
        
        # Auto-normalize CSI on first use
        if self.auto_normalize_csi:
            csi = self.get_normalized_csi(csi)
            
        # Process through modules
        for module in modules:
            if isinstance(module, ChannelAwareBaseModel):
                # Format CSI for this submodule
                formatted_csi = self.format_csi_for_submodules(csi, module)
                # Pass explicitly to channel-aware modules
                result = module(result, formatted_csi, *args, **kwargs)
            elif hasattr(module, 'forward') and 'csi' in module.forward.__code__.co_varnames:
                # Module has a forward method with a csi parameter but doesn't inherit from ChannelAwareBaseModel
                # This handles legacy or third-party modules
                result = module(result, csi, *args, **kwargs)
            else:
                # For regular modules, just pass the input
                result = module(result, *args, **kwargs)
                
        return result
        
    def get_last_csi(self) -> Optional[torch.Tensor]:
        """Get the most recently used normalized CSI tensor.
        
        This is useful for debugging and visualization purposes.
        
        Returns:
            Optional[torch.Tensor]: The last normalized CSI tensor used,
                or None if no CSI has been processed yet.
        """
        return self._last_csi
        
    @staticmethod
    def detect_csi_format(csi: torch.Tensor) -> CSIFormat:
        """Detect the most likely format of a CSI tensor.
        
        Args:
            csi (torch.Tensor): The CSI tensor to analyze
            
        Returns:
            CSIFormat: The detected format
        """
        if torch.is_complex(csi):
            return CSIFormat.COMPLEX
            
        # Check if likely in dB scale
        if csi.min() < 0 and csi.max() < 50:  # Typical dB range
            return CSIFormat.DB
            
        # Check if normalized
        if csi.min() >= 0 and csi.max() <= 1:
            return CSIFormat.NORMALIZED
            
        # Check for matrix format
        if csi.dim() >= 3:
            return CSIFormat.MATRIX
            
        # Check for vector format
        if csi.dim() == 2 and csi.shape[1] > 1:
            return CSIFormat.VECTOR
            
        # Default to linear
        return CSIFormat.LINEAR


class ConfigurableModel(BaseModel):
    """Model that supports dynamically adding and removing steps.

    This class extends the basic model functionality with methods to add, remove, and manage model
    steps during runtime.
    """

    def add_step(self, step: Any) -> "ConfigurableModel":
        """Add a processing step to the model.

        Args:
            step: A callable that will be added to the processing pipeline.
                Must accept and return tensor-like objects.

        Returns:
            Self for method chaining
        """
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
