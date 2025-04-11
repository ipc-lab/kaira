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
    LogNormalFadingChannel,
    NonlinearChannel,
    PhaseNoiseChannel,
    PoissonChannel,
    RayleighFadingChannel,
    RicianFadingChannel,
)

# Base channel classes
from .base import BaseChannel

# Digital channel models
from .digital import BinaryErasureChannel, BinarySymmetricChannel, BinaryZChannel
from .identity import IdealChannel, IdentityChannel, PerfectChannel

# Perfect/Identity channel
from .lambda_channel import LambdaChannel

# Channel registry
from .registry import ChannelRegistry

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
    "RayleighFadingChannel",
    "RicianFadingChannel",
    "LogNormalFadingChannel",
    "NonlinearChannel",
    # Channel registry
    "ChannelRegistry",
]
