"""DeepJSCC module for Kaira.

This module contains the DeepJSCCPipeline, which is a pipeline for image compression and
transmission using Deep Joint Source-Channel Coding (DeepJSCC).
"""

import torch.nn as nn

from .sequential import SequentialPipeline


class DeepJSCCPipeline(SequentialPipeline):
    """DeepJSCC Pipeline Module.

    This module implements a pipeline for image compression and transmission using Deep Joint
    Source-Channel Coding (DeepJSCC). It consists of an encoder, a channel, a constraint, and a
    decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        channel: nn.Module,
        constraint: nn.Module,
        decoder: nn.Module,
    ):
        """Initialize the DeepJSCCPipeline.

        Args:
            encoder (nn.Module): The encoder module.
            channel (nn.Module): The channel module.
            constraint (nn.Module): The constraint module.
            decoder (nn.Module): The decoder module.
        """
        super().__init__(steps=[encoder, channel, constraint, decoder])
        self.encoder = encoder
        self.channel = channel
        self.constraint = constraint
        self.decoder = decoder
