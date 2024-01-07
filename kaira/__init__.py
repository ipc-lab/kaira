"""
An open-source python framework for anomaly detection on streaming multivariate data.
"""
from .version import __version__
from . import core, channels, constraints, metrics, models, pipelines

__all__ = ['__version__', 'core', "channels", "constraints", "metrics", "models", "pipelines"]
