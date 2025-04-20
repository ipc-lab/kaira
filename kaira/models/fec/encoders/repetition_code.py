"""Repetition coding module for forward error correction.

This module implements repetition coding for binary data transmission, a simple error correction
technique where each bit is repeated multiple times. For decoding, a majority vote scheme is used
to determine the most likely transmitted bit value :cite:`lin2004error,moon2005error`.

Repetition coding provides a straightforward way to improve reliability at the expense of rate,
making it suitable for educational purposes and systems with strict reliability requirements but
modest rate requirements.

As a special case of linear block codes, repetition codes have a generator matrix consisting
of a single row of all ones, and a check matrix with (n-1) rows forming a basis for the
orthogonal complement of the all-ones vector :cite:`richardson2008modern`.
"""

from typing import Any

import torch
from scipy.special import comb

from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry

from .linear_block_code import LinearBlockCodeEncoder


@ModelRegistry.register_model("repetition_encoder")
class RepetitionCodeEncoder(LinearBlockCodeEncoder):
    """Encoder for repetition coding that extends LinearBlockCodeEncoder.

    This encoder implements a repetition code, which is a special case of linear
    block codes where each bit is repeated n times. A repetition code has the
    following properties:

    - Length: n (the repetition factor)
    - Dimension: k = 1 (one information bit produces n coded bits)
    - Redundancy: r = n - 1 (number of redundant bits)
    - Minimum distance: d = n (can correct up to ⌊(n-1)/2⌋ errors)

    Its dual is the single parity-check code. The generator matrix is a
    single row of all ones [1, 1, ..., 1].

    Attributes:
        repetition_factor (int): The length n of the code. Must be a positive integer.

    Args:
        repetition_factor (int): Number of times to repeat each bit

    Examples:
        >>> import torch
        >>> encoder = RepetitionCodeEncoder(repetition_factor=5)
        >>> encoder.code_length, encoder.code_dimension, encoder.redundancy
        (5, 1, 4)
        >>> encoder.generator_matrix
        tensor([[1., 1., 1., 1., 1.]])
        >>> encoder(torch.tensor([[1.]]))
        tensor([[1., 1., 1., 1., 1.]])
    """

    def __init__(self, repetition_factor: int = 3, **kwargs: Any):
        """Initialize the repetition encoder.

        Args:
            repetition_factor: Number of times to repeat each bit. Must be a positive integer.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If repetition_factor is less than 1.
        """
        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")

        # Create the generator matrix for a repetition code: [1, 1, ..., 1]
        generator_matrix = torch.ones((1, repetition_factor), dtype=torch.float32)

        # Remove generator_matrix from kwargs if it exists to avoid duplicate
        kwargs_copy = kwargs.copy()
        if "generator_matrix" in kwargs_copy:
            del kwargs_copy["generator_matrix"]

        # Initialize the LinearBlockCodeEncoder parent with this generator matrix
        super().__init__(generator_matrix=generator_matrix, **kwargs_copy)

        # Store repetition factor as an attribute
        self.repetition_factor = repetition_factor

    def coset_leader_weight_distribution(self) -> torch.Tensor:
        """Calculate the coset leader weight distribution of the repetition code.

        For a repetition code of length n, the coset leader weight distribution
        is given by the binomial coefficients C(n,w) for w from 0 to ⌊n/2⌋,
        with a special case for n/2 when n is even.

        Returns:
            Tensor containing the coset leader weight distribution
        """
        n = self.repetition_factor
        distribution = torch.zeros(n + 1, dtype=torch.int64)

        # Fill in the distribution values
        for w in range((n + 1) // 2):
            distribution[w] = int(comb(n, w))

        # Special case when n is even
        if n % 2 == 0:
            distribution[n // 2] = int(comb(n, n // 2) // 2)

        return distribution

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            String representation with key parameters
        """
        return f"{self.__class__.__name__}(repetition_factor={self.repetition_factor})"


@ModelRegistry.register_model("majority_vote_decoder")
class MajorityVoteDecoder(BaseModel):
    """Decoder for repetition coding using majority vote.

    This decoder reconstructs the original binary sequence from a repeated sequence by
    taking the majority vote of each group of repeated bits. It is specialized for
    repetition codes and provides more efficient implementation than the general
    linear block code decoder.

    For repetition codes, majority voting is the optimal decoding method for BSC
    (Binary Symmetric Channel) :cite:`lin2004error,moon2005error`.

    Example:
        >>> import torch
        >>> encoder = RepetitionCodeEncoder(repetition_factor=3)
        >>> decoder = MajorityVoteDecoder(repetition_factor=3)
        >>> x = torch.tensor([[1.0]])  # Original message
        >>> y = encoder(x)  # Encoded as [1, 1, 1]
        >>> y_noisy = torch.tensor([[1.0, 0.0, 1.0]])  # Channel noise flipped middle bit
        >>> decoder(y_noisy)
        tensor([[1.]])  # Correctly decoded despite error

    Args:
        repetition_factor: Number of times each bit was repeated in the encoding process
    """

    def __init__(self, repetition_factor: int = 3, *args: Any, **kwargs: Any):
        """Initialize the majority vote decoder.

        Args:
            repetition_factor: Number of times each bit was repeated. Must be a positive integer.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.

        Raises:
            ValueError: If repetition_factor is less than 1.
            Warning: If repetition_factor is even, as it could lead to ties in majority voting.
        """
        super().__init__(*args, **kwargs)
        if repetition_factor < 1:
            raise ValueError("Repetition factor must be a positive integer")
        if repetition_factor % 2 == 0:
            import warnings

            warnings.warn("Repetition factor is even, which may result in ties during majority voting. " "Consider using an odd repetition factor for unambiguous decoding.", UserWarning)
        self.repetition_factor = repetition_factor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the input tensor using majority voting.

        This method reshapes the input to group the repeated bits, then performs
        a majority vote (for hard decision) or averaging (for soft decision) to
        determine the most likely original bit values.

        Args:
            x: Input tensor of shape (..., encoded_length), where encoded_length =
               original_message_length * repetition_factor. Values can be binary (0, 1) for
               hard decoding or probabilities/LLRs for soft decoding.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Decoded binary tensor of shape (..., encoded_length // repetition_factor)
        """
        # Get the last dimension size
        last_dim_size = x.shape[-1]

        # Check if the last dimension is a multiple of the repetition factor
        if last_dim_size % self.repetition_factor != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of " f"repetition factor {self.repetition_factor}")

        # Define decoding function to apply to blocks
        def decode_block(blocks):
            # Shape: (..., num_blocks, repetition_factor)
            if torch.is_floating_point(blocks):
                # For soft decoding (input contains probabilities or LLRs)
                # Take average across repetitions and threshold at 0.5
                return (blocks.mean(dim=-1) > 0.5).float()
            else:
                # For hard decoding (input contains binary values)
                # Take majority vote across repetitions
                votes = blocks.sum(dim=-1)
                threshold = self.repetition_factor / 2
                return (votes > threshold).float()

        # Reshape input to group repeated bits and apply majority voting
        message_length = last_dim_size // self.repetition_factor

        # Apply blockwise reshape and decoding
        original_shape = x.shape

        # Reshape to expose blocks: (..., b*n) -> (..., b, n)
        reshaped = x.view(*original_shape[:-1], message_length, self.repetition_factor)

        # Apply decoding function
        decoded = decode_block(reshaped)

        return decoded

    def __repr__(self) -> str:
        """Return a string representation of the object.

        Returns:
            String representation with key parameters
        """
        return f"{self.__class__.__name__}(repetition_factor={self.repetition_factor})"
