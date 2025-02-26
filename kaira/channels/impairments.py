"""Hardware Impairment Channel Models."""

import torch
import math
from kaira.core import BaseChannel
from kaira.utils import to_tensor

class PhaseNoiseChannel(BaseChannel):
    """Phase Noise Channel.
    
    Models the phase noise introduced by oscillators in communication systems.
    The phase noise is modeled as a Wiener process (random walk) with variance
    parameter controlling the severity of the noise.

    Mathematical Model:
        y = x * exp(jθ)
        where θ follows a Wiener process (accumulated Gaussian increments)

    Args:
        phase_noise_variance (float): Variance of the phase noise increments
        
    Example:
        >>> channel = PhaseNoiseChannel(phase_noise_variance=0.01)
        >>> x = torch.complex(torch.ones(1000), torch.zeros(1000))
        >>> y = channel(x)
    """

    def __init__(self, phase_noise_variance):
        super().__init__()
        self.phase_noise_variance = to_tensor(phase_noise_variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply phase noise to the input tensor.

        Args:
            x (torch.Tensor): The input complex tensor.

        Returns:
            torch.Tensor: The output tensor after applying phase noise.
        """
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
            
        batch_size = x.shape[0]
        seq_length = x.shape[-1] if len(x.shape) > 2 else 1
        
        # Generate phase noise increments (Gaussian)
        phase_incr = torch.randn(batch_size, seq_length, device=x.device) * torch.sqrt(self.phase_noise_variance)
        
        # Accumulate to get Wiener process
        phase_noise = torch.cumsum(phase_incr, dim=1)
        
        # Reshape to match input dimensions
        if len(x.shape) > 2:
            phase_noise = phase_noise.view(*x.shape)
        
        # Apply phase rotation
        phase_rotation = torch.exp(1j * phase_noise)
        return x * phase_rotation


class IQImbalanceChannel(BaseChannel):
    """I/Q Imbalance Channel.
    
    Models the amplitude and phase imbalance between in-phase and quadrature
    components in radio frequency hardware.

    Mathematical Model:
        y = (1+ε)x_I + j(1-ε)x_Q * exp(jθ)
        where ε is the amplitude imbalance and θ is the phase imbalance

    Args:
        amplitude_imbalance (float): Relative amplitude imbalance between I and Q
        phase_imbalance (float): Phase imbalance in radians
        
    Example:
        >>> channel = IQImbalanceChannel(amplitude_imbalance=0.05, phase_imbalance=0.1)
        >>> x = torch.complex(torch.ones(10), torch.ones(10))
        >>> y = channel(x)
    """

    def __init__(self, amplitude_imbalance, phase_imbalance):
        super().__init__()
        self.amplitude_imbalance = to_tensor(amplitude_imbalance)
        self.phase_imbalance = to_tensor(phase_imbalance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply I/Q imbalance to the input tensor.

        Args:
            x (torch.Tensor): The input complex tensor.

        Returns:
            torch.Tensor: The output tensor after applying I/Q imbalance.
        """
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
            
        # Extract I and Q components
        i_component = x.real
        q_component = x.imag
        
        # Apply amplitude imbalance
        i_imbalanced = i_component * (1 + self.amplitude_imbalance)
        q_imbalanced = q_component * (1 - self.amplitude_imbalance)
        
        # Apply phase imbalance
        q_phase_shifted = q_imbalanced * torch.cos(self.phase_imbalance) - i_imbalanced * torch.sin(self.phase_imbalance)
        
        # Construct imbalanced signal
        return torch.complex(i_imbalanced, q_phase_shifted)
