"""Generic model implementations for Kaira.

This module provides generic model implementations that can be used as building blocks for more
complex models, such as sequential, parallel, and branching models.
"""

from .branching import BranchingModel
from .identity import IdentityModel
from .lambda_model import LambdaModel
from .parallel import ParallelModel
from .sequential import SequentialModel

__all__ = [
    "IdentityModel",
    "SequentialModel",
    "ParallelModel",
    "BranchingModel",
    "LambdaModel",
]
