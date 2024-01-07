import torch
from kaira.core import BaseChannel
from kaira.utils import to_tensor

__all__ = [
    "PerfectChannel",
    "AWGNChannel",
    "ComplexAWGNChannel",
]

class AWGNChannel(BaseChannel):
    def __init__(self, avg_noise_power: float):
        """
        Initialize the AWGNChannel object.

        Args:
            avg_noise_power (float): The average noise power.

        Returns:
            None
        """
        super().__init__()
        self.avg_noise_power = to_tensor(avg_noise_power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply AWGN (Additive White Gaussian Noise) to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape BxCxWxH.

        Returns:
            torch.Tensor: The output tensor after adding AWGN.
        """
        awgn = torch.randn_like(x) * torch.sqrt(self.avg_noise_power)
        x = x + awgn
        return x

class ComplexAWGNChannel(BaseChannel):
    """
    Complex Additive White Gaussian Noise (AWGN) channel.

    This channel adds complex Gaussian noise to the input signal, simulating complex domain.

    """

    def __init__(self, avg_noise_power: float):
        super().__init__()
        self.avg_noise_power = to_tensor(avg_noise_power) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ComplexAWGNChannel.

        Args:
            x (torch.Tensor): The input signal tensor.

        Returns:
            torch.Tensor: The output signal tensor after adding complex Gaussian noise (equivalent to standard domain noise, but in complex domain).
        """
        awgn = torch.randn_like(x) * torch.sqrt(torch.tensor(self.avg_noise_power, device=x.device))
        x = x + awgn
        return x

class PerfectChannel(BaseChannel):
    """
    A perfect channel that simply returns the input without any modification.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        Forward pass of the perfect channel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor without any modification.
        """
        return x
