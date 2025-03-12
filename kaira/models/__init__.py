"""Models module for Kaira."""
from . import components, image, generic, binary
from .base import BaseModel, ConfigurableModel
from .deepjscc import DeepJSCCModel
from .feedback_channel import FeedbackChannelModel
from .registry import ModelRegistry
from .wyner_ziv import WynerZivModel

__all__ = [
    # Modules
    "generic",
    "components", 
    "binary",
    "image",
    
    # Base classes
    "BaseModel", 
    "ConfigurableModel",
    
    # Specialized models
    "DeepJSCCModel",
    "FeedbackChannelModel",
    "WynerZivModel",
    
    # Registry
    "ModelRegistry",
]
