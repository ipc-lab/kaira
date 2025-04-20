"""Models module for Kaira."""

from . import binary, components, fec, generic, image
from .base import BaseModel, ConfigurableModel
from .channel_code import ChannelCodeModel
from .deepjscc import DeepJSCCModel
from .feedback_channel import FeedbackChannelModel
from .multiple_access_channel import MultipleAccessChannelModel
from .registry import ModelRegistry
from .wyner_ziv import WynerZivModel

__all__ = [
    # Modules
    "generic",
    "components",
    "binary",
    "image",
    "fec",
    # Base classes
    "BaseModel",
    "ConfigurableModel",
    # Specialized models
    "ChannelCodeModel",
    "DeepJSCCModel",
    "FeedbackChannelModel",
    "WynerZivModel",
    "MultipleAccessChannelModel",
    # Registry
    "ModelRegistry",
]
