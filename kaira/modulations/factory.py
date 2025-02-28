"""Factory functions and public API for creating modulation scheme components."""

from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .base import BaseDemodulator, BaseModulator
from .schemes.differential import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)
from .schemes.identity import IdentityDemodulator, IdentityModulator
from .schemes.offset import (
    OQPSKDemodulator,
    OQPSKModulator,
    Pi4QPSKDemodulator,
    Pi4QPSKModulator,
)
from .schemes.pam import PAMDemodulator, PAMModulator
from .schemes.psk import (
    BPSKDemodulator,
    BPSKModulator,
    PSKDemodulator,
    PSKModulator,
    QPSKDemodulator,
    QPSKModulator,
)
from .schemes.qam import QAMDemodulator, QAMModulator

# ============================================================================
# Public API
# ============================================================================


def available_schemes() -> List[str]:
    """Get a list of all available modulation schemes.

    Returns:
        List of supported modulation scheme names
    """
    return sorted(list(_MODULATORS.keys()))


def create_modulator(scheme: str, **kwargs) -> BaseModulator:
    """Create a modulator instance for the specified scheme.

    Args:
        scheme: Modulation scheme name (case-insensitive)
        **kwargs: Additional parameters for the modulator constructor

    Returns:
        A modulator instance for the specified scheme

    Raises:
        ValueError: If the scheme is not supported
    """
    scheme = scheme.lower()
    return _create_modulator(scheme, **kwargs)


def create_demodulator(scheme: str, **kwargs) -> BaseDemodulator:
    """Create a demodulator instance for the specified scheme.

    Args:
        scheme: Modulation scheme name (case-insensitive)
        **kwargs: Additional parameters for the demodulator constructor

    Returns:
        A demodulator instance for the specified scheme

    Raises:
        ValueError: If the scheme is not supported
    """
    scheme = scheme.lower()
    return _create_demodulator(scheme, **kwargs)


def create_modem(scheme: str, **kwargs) -> Tuple[BaseModulator, BaseDemodulator]:
    """Create a matched modulator-demodulator pair for the specified scheme.

    Args:
        scheme: Modulation scheme name (case-insensitive)
        **kwargs: Additional parameters for the modulator/demodulator constructors

    Returns:
        A tuple of (modulator, demodulator) instances

    Raises:
        ValueError: If the scheme is not supported
    """
    scheme = scheme.lower()
    return _create_modulator(scheme, **kwargs), _create_demodulator(scheme, **kwargs)


def get_modulation_info(scheme: str) -> Dict[str, Union[int, float, bool]]:
    """Get information about a modulation scheme.

    Args:
        scheme: Modulation scheme name (case-insensitive)

    Returns:
        Dictionary with information about the modulation scheme

    Raises:
        ValueError: If the scheme is not supported
    """
    from .utils.metrics import calculate_spectral_efficiency

    scheme = scheme.lower()
    modulator = create_modulator(scheme)

    info = {
        "name": scheme,
        "bits_per_symbol": modulator.bits_per_symbol,
        "spectral_efficiency": calculate_spectral_efficiency(scheme),
    }

    # Add scheme-specific information
    for attr in ["order", "gray_coding", "normalize"]:
        if hasattr(modulator, attr):
            info[attr if attr != "normalize" else "normalized"] = getattr(modulator, attr)

    return info


class Modem(nn.Module):
    """Unified modulator-demodulator class.

    This class encapsulates both modulation and demodulation in a single module,
    making it easier to use in end-to-end systems.

    Attributes:
        modulator: The encapsulated modulator
        demodulator: The encapsulated demodulator
        bits_per_symbol: Number of bits per modulated symbol
    """

    def __init__(self, scheme: str, **kwargs) -> None:
        """Initialize with a modulation scheme.

        Args:
            scheme: Modulation scheme name (case-insensitive)
            **kwargs: Additional parameters for the modulator/demodulator constructors
        """
        super().__init__()
        self.modulator, self.demodulator = create_modem(scheme, **kwargs)
        self._bits_per_symbol = self.modulator.bits_per_symbol

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per modulated symbol."""
        return self._bits_per_symbol

    def modulate(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate bits to symbols.

        Args:
            x: Input bits

        Returns:
            Modulated symbols
        """
        return self.modulator(x)

    def demodulate(self, y: torch.Tensor, noise_var: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Demodulate symbols to bits or LLRs.

        Args:
            y: Received symbols
            noise_var: Noise variance for soft demodulation

        Returns:
            Demodulated bits or LLRs
        """
        return self.demodulator(y, noise_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through both modulator and demodulator (without noise).

        Args:
            x: Input bits

        Returns:
            Demodulated bits
        """
        y = self.modulator(x)
        return self.demodulator(y)

    def reset_state(self) -> None:
        """Reset internal states of both modulator and demodulator."""
        self.modulator.reset_state()
        self.demodulator.reset_state()

    def plot_constellation(self, **kwargs):
        """Plot the modulation constellation.

        Args:
            **kwargs: Additional plotting parameters

        Returns:
            Matplotlib figure object
        """
        return self.modulator.plot_constellation(**kwargs)


# ============================================================================
# Internal implementation
# ============================================================================

# Registry of available modulators and demodulators
_MODULATORS: Dict[str, Type[BaseModulator]] = {
    "bpsk": BPSKModulator,
    "qpsk": QPSKModulator,
    "psk": PSKModulator,
    "qam": QAMModulator,
    "pam": PAMModulator,
    "oqpsk": OQPSKModulator,
    "pi4qpsk": Pi4QPSKModulator,
    "dpsk": DPSKModulator,
    "dbpsk": DBPSKModulator,
    "dqpsk": DQPSKModulator,
    "identity": IdentityModulator,
}

_DEMODULATORS: Dict[str, Type[BaseDemodulator]] = {
    "bpsk": BPSKDemodulator,
    "qpsk": QPSKDemodulator,
    "psk": PSKDemodulator,
    "qam": QAMDemodulator,
    "pam": PAMDemodulator,
    "oqpsk": OQPSKDemodulator,
    "pi4qpsk": Pi4QPSKDemodulator,
    "dpsk": DPSKDemodulator,
    "dbpsk": DBPSKDemodulator,
    "dqpsk": DQPSKDemodulator,
    "identity": IdentityDemodulator,
}


def _create_modulator(scheme: str, **kwargs) -> BaseModulator:
    """Create a modulator instance for the specified scheme.

    Internal factory function used by the public API.

    Args:
        scheme: Modulation scheme name (lowercase)
        **kwargs: Additional parameters for the modulator constructor

    Returns:
        A modulator instance for the specified scheme

    Raises:
        ValueError: If the scheme is not supported
    """
    if scheme not in _MODULATORS:
        raise ValueError(f"Unsupported modulation scheme: {scheme}")

    return _MODULATORS[scheme](**kwargs)


def _create_demodulator(scheme: str, **kwargs) -> BaseDemodulator:
    """Create a demodulator instance for the specified scheme.

    Internal factory function used by the public API.

    Args:
        scheme: Modulation scheme name (lowercase)
        **kwargs: Additional parameters for the demodulator constructor

    Returns:
        A demodulator instance for the specified scheme

    Raises:
        ValueError: If the scheme is not supported
    """
    if scheme not in _DEMODULATORS:
        raise ValueError(f"Unsupported demodulation scheme: {scheme}")

    return _DEMODULATORS[scheme](**kwargs)
