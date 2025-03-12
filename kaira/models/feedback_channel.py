"""Feedback Channel module for Kaira.

This module contains the FeedbackChannelModel, which models a communication system with a
feedback path from the receiver to the transmitter.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from kaira.channels import BaseChannel
from .base import BaseModel
from .registry import ModelRegistry


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
        """
        super().__init__()
        self.encoder = encoder
        self.forward_channel = forward_channel
        self.decoder = decoder
        self.feedback_generator = feedback_generator
        self.feedback_channel = feedback_channel
        self.feedback_processor = feedback_processor
        self.max_iterations = max_iterations

    def forward(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Process input through the feedback channel system.

        Performs an iterative transmission process where:
        1. Transmitter encodes and sends data
        2. Receiver decodes and generates feedback
        3. Feedback is sent back to the transmitter
        4. Transmitter adapts based on feedback
        5. Process repeats for the specified number of iterations

        Args:
            input_data (torch.Tensor): The input data to transmit

        Returns:
            Dict[str, Any]: A dictionary containing:
                - final_output: The final decoded output
                - iterations: List of per-iteration results
                - feedback_history: History of feedback signals
        """
        batch_size = input_data.shape[0]
        device = input_data.device

        # Storage for results
        iterations = []
        feedback_history = []

        # Initial state - no feedback yet
        feedback = torch.zeros(batch_size, self.feedback_processor.input_size, device=device)

        # Iterative transmission process
        for i in range(self.max_iterations):
            # Process any feedback from previous iteration (skipped in first iteration)
            encoder_state = self.feedback_processor(feedback) if i > 0 else None

            # Encode the input (with adaptation if not first iteration)
            if encoder_state is not None:
                encoded = self.encoder(input_data, state=encoder_state)
            else:
                encoded = self.encoder(input_data)

            # Transmit through forward channel
            received = self.forward_channel(encoded)

            # Decode the received signal
            decoded = self.decoder(received)

            # Generate feedback
            feedback = self.feedback_generator(decoded, input_data)

            # Transmit feedback through feedback channel
            feedback = self.feedback_channel(feedback)

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

        return {
            "final_output": decoded,
            "iterations": iterations,
            "feedback_history": feedback_history,
        }
