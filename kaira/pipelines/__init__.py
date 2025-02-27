"""Kaira Pipelines Module.

This module provides various pipeline implementations for building communication systems.
Each pipeline handles specific communication scenarios with appropriate components.

Pipeline Categories:
* Generic Pipelines - General purpose data processing pipelines
  - SequentialPipeline - Processes steps in sequence, each receiving output from previous
  - ParallelPipeline - Processes steps in parallel, all receiving same input
  - BranchingPipeline - Routes input to different pipelines based on conditions

* Communication System Pipelines - Specialized for communication tasks
  - DeepJSCCPipeline - Deep Joint Source-Channel Coding for image transmission
  - WynerZivPipeline - Distributed source coding with side information
  - OFDMPipeline - Orthogonal Frequency Division Multiplexing
  - MIMOPipeline - Multiple-Input Multiple-Output communication system
  - FadingChannelPipeline - Communication over fading channels
  - FeedbackChannelPipeline - Communication with feedback path
"""

# Base pipeline class only
from .base import BasePipeline

# Generic pipelines
from .sequential import SequentialPipeline
from .parallel import ParallelPipeline
from .branching import BranchingPipeline

# Communication system pipelines
from .deepjscc import DeepJSCCPipeline
from .wyner_ziv import WynerZivPipeline, WynerZivCorrelationModel
from .ofdm import OFDMPipeline
from .mimo import MIMOPipeline
from .fading_channel import FadingChannelPipeline, FadingType
from .feedback import FeedbackChannelPipeline

__all__ = [
    # Base pipeline class
    "BasePipeline",
    
    # Generic pipelines
    "SequentialPipeline",
    "ParallelPipeline",
    "BranchingPipeline",
    
    # Communication system pipelines
    "DeepJSCCPipeline",
    "WynerZivPipeline",
    "WynerZivCorrelationModel",
    "OFDMPipeline",
    "MIMOPipeline",
    "FadingChannelPipeline",
    "FadingType",
    "FeedbackChannelPipeline",
]
