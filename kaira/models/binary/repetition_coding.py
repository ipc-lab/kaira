"""Repetition coding module for binary data.

This module implements repetition coding for binary data transmission, a simple error correction
technique where each bit is repeated multiple times. For decoding, a majority vote scheme is used
to determine the most likely transmitted bit value.

Repetition coding provides a straightforward way to improve reliability at the expense of rate,
making it suitable for educational purposes and systems with strict reliability requirements but
modest rate requirements.
"""

from typing import Any

import torch

from kaira.models.registry import ModelRegistry

from ..base import BaseModel


@ModelRegistry.register_model("repetition_encoder")
class RepetitionEncoder(BaseModel):
    """Encoder for repetition coding.

    This encoder repeats each bit in the input sequence a specified number of times,
    creating a redundant signal that can withstand some level of bit errors.

    Example:
        With repetition_factor = 3:
        [0, 1] becomes [0, 0, 0, 1, 1, 1]

    Args:
        repetition_factor: Number of times to repeat each bit
    """

    def __init__(self, repetition_factor: int = 3, *args: Any, **kwargs: Any):
        """Initialize the repetition encoder.

        Args:
            repetition_factor: Number of times to repeat each bit. Must be a positive integer.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")
        self.repetition_factor = repetition_factor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode the input tensor using repetition coding.

        Args:
            x: Input binary tensor of shape (batch_size, message_length)
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Encoded tensor of shape (batch_size, message_length * repetition_factor)
        """
        batch_size, message_length = x.shape

        # Repeat each bit along a new dimension
        repeated = x.unsqueeze(-1).repeat(1, 1, self.repetition_factor)

        # Reshape to flatten the repeated bits into a single dimension
        return repeated.reshape(batch_size, message_length * self.repetition_factor)


@ModelRegistry.register_model("majority_vote_decoder")
class MajorityVoteDecoder(BaseModel):
    """Decoder for repetition coding using majority vote.

    This decoder reconstructs the original binary sequence from a repeated sequence by
    taking the majority vote of each group of repeated bits.

    Example:
        With repetition_factor = 3:
        [0, 0, 1, 1, 1, 0] becomes [0, 1]

    Args:
        repetition_factor: Number of times each bit was repeated in the encoding process
        input_type: Type of soft input, can be 'prob' (probabilities between 0 and 1) or
                   'llr' (log-likelihood ratios). Only used when input is floating point.
                   Default is 'prob'.
    """

    def __init__(self, repetition_factor: int = 3, input_type: str = "prob", *args: Any, **kwargs: Any):
        """Initialize the majority vote decoder.

        Args:
            repetition_factor: Number of times each bit was repeated. Must be a positive integer.
            input_type: Type of soft input, can be 'prob' (probabilities between 0 and 1) or
                       'llr' (log-likelihood ratios). Only used when input is floating point.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)
        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")
        if repetition_factor % 2 == 0:
            raise ValueError("Repetition factor should be odd for unambiguous majority voting")

        self.repetition_factor = repetition_factor

        valid_input_types = ["prob", "llr"]
        if input_type not in valid_input_types:
            raise ValueError(f"input_type must be one of {valid_input_types}, got {input_type}")
        self.input_type = input_type

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the input tensor using majority voting.

        Args:
            x: Input tensor of shape (batch_size, encoded_length), where encoded_length =
               original_message_length * repetition_factor. Values can be binary (0, 1) for
               hard decoding or probabilities/LLRs for soft decoding.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Decoded binary tensor of shape (batch_size, encoded_length // repetition_factor)
        """
        batch_size, encoded_length = x.shape
        message_length = encoded_length // self.repetition_factor

        # Reshape to separate the repetition dimension
        reshaped = x.reshape(batch_size, message_length, self.repetition_factor)

        if torch.is_floating_point(x):
            # For soft decoding (input contains probabilities or LLRs)
            if self.input_type == "prob":
                # For probability values (between 0 and 1)
                # Take average across repetitions and threshold at 0.5
                decoded = reshaped.mean(dim=2) > 0.5
            elif self.input_type == "llr":
                # For log-likelihood ratios
                # Sum LLRs and check if result > 0
                # Positive LLRs favor bit=0, negative LLRs favor bit=1
                decoded = reshaped.sum(dim=2) < 0  # Sum LLRs and threshold at 0
        else:
            # For hard decoding (input contains binary values)
            # Take majority vote across repetitions
            votes = reshaped.sum(dim=2)
            threshold = self.repetition_factor / 2
            decoded = votes > threshold

        return decoded.float()
