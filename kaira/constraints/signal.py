"""Signal characteristic constraints for communication systems.

This module provides constraints related to signal characteristics such as
amplitude limitations and spectral properties.
"""

import torch
import torch.nn.functional as F
from .base import BaseConstraint


class PeakAmplitudeConstraint(BaseConstraint):
    """Peak Amplitude Constraint.
    
    Limits the maximum amplitude of the signal to prevent clipping.
    """
    
    def __init__(self, max_amplitude: float) -> None:
        """Initialize the peak amplitude constraint.
        
        Args:
            max_amplitude (float): Maximum allowed amplitude.
        """
        super().__init__()
        self.max_amplitude = max_amplitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply peak amplitude constraint.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Amplitude-constrained signal.
        """
        # Simple clipping approach
        return torch.clamp(x, -self.max_amplitude, self.max_amplitude)
    
    
class SpectralMaskConstraint(BaseConstraint):
    """Spectral Mask Constraint.
    
    Ensures the signal's spectrum complies with regulatory requirements.
    """
    
    def __init__(self, mask: torch.Tensor) -> None:
        """Initialize the spectral mask constraint.
        
        Args:
            mask (torch.Tensor): Spectral mask defining maximum power per frequency bin.
        """
        super().__init__()
        self.register_buffer('mask', mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral mask constraint.
        
        Args:
            x (torch.Tensor): Input tensor in time domain.
            
        Returns:
            torch.Tensor: Spectral mask constrained signal in time domain.
        """
        x_freq = torch.fft.fft(x, dim=-1)
        
        # Calculate power in frequency domain
        power_spectrum = torch.abs(x_freq)**2
        
        # Apply mask by scaling where needed
        excess_indices = power_spectrum > self.mask.expand_as(power_spectrum)
        
        if torch.any(excess_indices):
            # Scale frequency components to meet the mask
            scale_factor = torch.sqrt(self.mask / (power_spectrum + 1e-8))
            scale_factor = torch.where(excess_indices, scale_factor, torch.ones_like(scale_factor))
            x_freq = x_freq * scale_factor
        
        # Convert back to time domain
        return torch.fft.ifft(x_freq, dim=-1).real
