"""Generic model implementations for Kaira.

This module provides generic model implementations that can be used as building blocks
for more complex models, such as sequential, parallel, and branching models.
"""

from .identity import IdentityModel
from .sequential import SequentialModel
from .parallel import ParallelModel
from .branching import BranchingModel
from .lambda_model import LambdaModel

__all__ = [
    "IdentityModel",
    "SequentialModel",
    "ParallelModel",
    "BranchingModel",
    "LambdaModel",
]