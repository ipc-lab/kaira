"""Nonlinear Channel Models."""

import torch
import numpy as np
from kaira.core import BaseChannel
from kaira.utils import to_tensor
from .utils import snr_to_noise_power

class NonlinearChannel(BaseChannel):
    """Nonlinear Channel with Memory.
    
    Models nonlinear distortion in communication systems, such as power amplifier
    nonlinearities. The nonlinearity is modeled using a memoryless polynomial function
    followed by optional AWGN.
    
    Mathematical Model:
        y = f(x) + n
        where f(x) = a₁x + a₂x² + a₃x³ + ... + aₘx^m
        and n ~ N(0, σ²)
    
    Args:
        coefficients (list): List of polynomial coefficients [a₁, a₂, ..., aₘ]
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        add_noise (bool): Whether to add noise after nonlinear distortion
        
    Example:
        >>> # Create a soft-limiter model with cubic nonlinearity
        >>> channel = NonlinearChannel(coefficients=[1.0, 0.0, -0.2])
        >>> x = torch.linspace(-2, 2, 100)
        >>> y = channel(x)  # y will show compression at high amplitudes
    """
    
    def __init__(self, coefficients, avg_noise_power=None, snr_db=None, add_noise=True):
        super().__init__()
        self.coefficients = to_tensor(coefficients)
        self.add_noise = add_noise
        
        if snr_db is not None:
            self.snr_db = snr_db
            self.avg_noise_power = None
        elif avg_noise_power is not None:
            self.avg_noise_power = to_tensor(avg_noise_power)
            self.snr_db = None
        else:
            self.avg_noise_power = 0
            self.snr_db = None
    
    def apply_nonlinearity(self, x):
        """Apply polynomial nonlinearity to input signal.
        
        Args:
            x (torch.Tensor): Input signal
            
        Returns:
            torch.Tensor: Nonlinearly distorted signal
        """
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * torch.pow(x, i+1)
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear distortion and optional AWGN to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor
            
        Returns:
            torch.Tensor: The output tensor after nonlinear distortion and noise
        """
        # Apply nonlinearity
        y = self.apply_nonlinearity(x)
        
        # Add noise if requested
        if self.add_noise:
            noise_power = self.avg_noise_power
            if self.snr_db is not None:
                signal_power = torch.mean(torch.abs(y) ** 2)
                noise_power = snr_to_noise_power(signal_power, self.snr_db)
                
            if torch.is_complex(x):
                noise_real = torch.randn_like(y.real) * torch.sqrt(noise_power / 2)
                noise_imag = torch.randn_like(y.imag) * torch.sqrt(noise_power / 2)
                noise = torch.complex(noise_real, noise_imag)
            else:
                noise = torch.randn_like(y) * torch.sqrt(noise_power)
                
            y = y + noise
            
        return y


class RappModel(NonlinearChannel):
    """Rapp Model for Power Amplifier Nonlinearity.
    
    The Rapp model is commonly used to model the AM/AM conversion of solid-state 
    power amplifiers. It provides a smooth transition from linear region to 
    saturation.
    
    Mathematical Model:
        y = x / (1 + (|x|/sat_level)^(2*smoothness))^(1/(2*smoothness)) + n
        where n ~ N(0, σ²)
        
    Args:
        saturation_level (float): Input level where amplifier begins to saturate
        smoothness (float): Smoothness parameter (higher = sharper transition)
        avg_noise_power (float, optional): The average noise power σ²
        snr_db (float, optional): SNR in dB (alternative to avg_noise_power)
        add_noise (bool): Whether to add noise after nonlinear distortion
        
    Example:
        >>> # Create a power amplifier model with saturation
        >>> pa = RappModel(saturation_level=0.8, smoothness=2.0, snr_db=30)
        >>> x = torch.linspace(-1.5, 1.5, 100)
        >>> y = pa(x)  # y will show saturation effects
    """
    
    def __init__(self, saturation_level, smoothness, avg_noise_power=None, snr_db=None, add_noise=True):
        super().__init__([], avg_noise_power, snr_db, add_noise)
        self.saturation_level = saturation_level
        self.smoothness = smoothness
    
    def apply_nonlinearity(self, x):
        """Apply Rapp model nonlinearity to input signal.
        
        Args:
            x (torch.Tensor): Input signal
            
        Returns:
            torch.Tensor: Nonlinearly distorted signal
        """
        if torch.is_complex(x):
            magnitude = torch.abs(x)
            phase = torch.angle(x)
            
            # Apply Rapp model to magnitude
            denominator = (1 + (magnitude / self.saturation_level)**(2*self.smoothness))**(1/(2*self.smoothness))
            new_magnitude = magnitude / denominator
            
            # Reconstruct complex signal with new magnitude
            return new_magnitude * torch.exp(1j * phase)
        else:
            # For real signals, preserve sign
            sign = torch.sign(x)
            magnitude = torch.abs(x)
            
            # Apply Rapp model to magnitude
            denominator = (1 + (magnitude / self.saturation_level)**(2*self.smoothness))**(1/(2*self.smoothness))
            new_magnitude = magnitude / denominator
            
            return sign * new_magnitude
