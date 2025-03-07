"""Channel models for communication systems.

This package provides various channel models for simulating communication systems, including analog
and digital channels, with support for various noise models, distortions, and fading patterns.
"""

# Analog channel models
from .analog import (
    AWGNChannel,
    FlatFadingChannel,
    GaussianChannel,
    LaplacianChannel,
    NonlinearChannel,
    PhaseNoiseChannel,
    PoissonChannel,
)

# Base channel classes
from .base import BaseChannel

# Digital channel models
from .digital import BinaryErasureChannel, BinarySymmetricChannel, BinaryZChannel

# Perfect/Identity channel
from .lambda_channel import LambdaChannel
from .identity import IdealChannel, IdentityChannel, PerfectChannel

__all__ = [
    # Base classes
    "BaseChannel",
    "LambdaChannel",
    # Perfect/Identity channel
    "PerfectChannel",
    "IdentityChannel",
    "IdealChannel",
    # Digital channels
    "BinarySymmetricChannel",
    "BinaryErasureChannel",
    "BinaryZChannel",
    # Analog channels
    "AWGNChannel",
    "GaussianChannel",
    "LaplacianChannel",
    "PoissonChannel",
    "PhaseNoiseChannel",
    "FlatFadingChannel",
    "NonlinearChannel",
]
