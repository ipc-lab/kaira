"""Feedback Channel module for Kaira.

This module contains the FeedbackChannelModel, which models a communication system with a feedback
path from the receiver to the transmitter.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from kaira.channels.base import BaseChannel
from kaira.models.base import BaseModel

# Import registry directly from registry module to avoid circular imports
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model("feedback_channel")
class FeedbackChannelModel(BaseModel):
    """A model that models communication with a feedback channel.

    In a feedback channel, the receiver can send information back to the transmitter,
    allowing the transmitter to adapt its strategy based on feedback. This model
    models the iterative process of transmission, reception, feedback, and adaptation.

    Attributes:
        encoder (BaseModel): The encoder at the transmitter
        forward_channel (BaseChannel): The channel from transmitter to receiver
        decoder (BaseModel): The decoder at the receiver
        feedback_generator (nn.Module): Module that generates feedback at the receiver
        feedback_channel (BaseChannel): The channel for feedback from receiver to transmitter
        feedback_processor (nn.Module): Module that processes feedback at the transmitter
        max_iterations (int): Maximum number of transmission iterations
    """

    def __init__(
        self,
        encoder: BaseModel,
        forward_channel: BaseChannel,
        decoder: BaseModel,
        feedback_generator: nn.Module,
        feedback_channel: BaseChannel,
        feedback_processor: nn.Module,
        max_iterations: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the feedback channel model.

        Args:
            encoder (BaseModel): The encoder that processes input data
            forward_channel (BaseChannel): The channel from transmitter to receiver
            decoder (BaseModel): The decoder at the receiver
            feedback_generator (nn.Module): Module that generates feedback signals
            feedback_channel (BaseChannel): The channel for feedback
            feedback_processor (nn.Module): Module that processes feedback at the transmitter
            max_iterations (int): Maximum number of transmission iterations (default: 1)
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.forward_channel = forward_channel
        self.decoder = decoder
        self.feedback_generator = feedback_generator
        self.feedback_channel = feedback_channel
        self.feedback_processor = feedback_processor
        self.max_iterations = max_iterations

    def forward(self, input_data: torch.Tensor, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process input through the feedback channel system.

        Performs an iterative transmission process where:
        1. Transmitter encodes and sends data
        2. Receiver decodes and generates feedback
        3. Feedback is sent back to the transmitter
        4. Transmitter adapts based on feedback
        5. Process repeats for the specified number of iterations

        Args:
            input_data (torch.Tensor): The input data to transmit
            *args: Additional positional arguments passed to internal components.
            **kwargs: Additional keyword arguments passed to internal components.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - final_output: The final decoded output (only if at least one iteration)
                - iterations: List of per-iteration results
                - feedback_history: History of feedback signals
        """

        # Storage for results
        iterations = []
        feedback_history = []
        final_output = None

        # Initial state - no feedback yet
        feedback = None

        # Iterative transmission process
        for i in range(self.max_iterations):
            # Process any feedback from previous iteration (skipped in first iteration)
            encoder_state = self.feedback_processor(feedback, *args, **kwargs) if i > 0 else None

            # Encode the input (with adaptation if not first iteration)
            if encoder_state is not None:
                # Pass state and other args/kwargs to encoder
                encoded = self.encoder(input_data, state=encoder_state, *args, **kwargs)
            else:
                # Pass args/kwargs to encoder
                encoded = self.encoder(input_data, *args, **kwargs)

            # Transmit through forward channel (Channels typically don't take arbitrary *args, **kwargs)
            received = self.forward_channel(encoded, *args, **kwargs)  # Pass args/kwargs

            # Decode the received signal - pass args/kwargs to decoder
            decoded = self.decoder(received, *args, **kwargs)

            # Generate feedback - pass args/kwargs to feedback generator
            # Pass input_data as the 'original' argument
            feedback = self.feedback_generator(decoded, input_data, *args, **kwargs)

            # Transmit feedback through feedback channel (Channels typically don't take arbitrary *args, **kwargs)
            feedback = self.feedback_channel(feedback, *args, **kwargs)  # Pass args/kwargs

            # Store results for this iteration
            iterations.append(
                {
                    "encoded": encoded,
                    "received": received,
                    "decoded": decoded,
                    "feedback": feedback,
                }
            )

            feedback_history.append(feedback)
            final_output = decoded

        result = {
            "iterations": iterations,
            "feedback_history": feedback_history,
        }

        # Only include final_output if we have run at least one iteration
        if final_output is not None:
            result["final_output"] = final_output

        return result
