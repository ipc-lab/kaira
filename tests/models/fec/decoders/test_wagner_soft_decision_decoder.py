"""Tests for the WagnerSoftDecisionDecoder in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.models.fec.decoders.wagner_soft_decision_decoder import WagnerSoftDecisionDecoder
from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder


class TestWagnerSoftDecisionDecoder:
    """Test suite for WagnerSoftDecisionDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)

        # Initialize the decoder with this encoder
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify that code parameters are correct for a single parity-check code
        assert decoder.code_length == 4
        assert decoder.code_dimension == 3
        assert decoder.redundancy == 1

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Create an encoder that is not a single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)

        # Initialization should raise ValueError
        with pytest.raises(ValueError, match="Wagner decoder is only applicable to single parity-check codes"):
            WagnerSoftDecisionDecoder(encoder=encoder)

    def test_decoding_no_errors(self):
        """Test decoding soft values with no errors."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        encoder(message)  # Should be [1, 0, 1, 0]

        # Create soft values corresponding to the codeword
        # Negative values represent 1s, positive values represent 0s
        # For [1, 0, 1, 0], we use soft values like [-2.0, 2.0, -2.0, 2.0]
        soft_received = torch.tensor([-2.0, 2.0, -2.0, 2.0])

        # Decode the soft values
        decoded = decoder(soft_received)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

    def test_decoding_with_noise(self):
        """Test decoding soft values with noise but no bit errors."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        encoder(message)  # Should be [1, 0, 1, 0]

        # Create noisy soft values but with correct sign
        # The sign still indicates the correct bit, but with varying reliability
        soft_received = torch.tensor([-1.2, 0.8, -1.5, 0.4])

        # Decode the soft values
        decoded = decoder(soft_received)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

    def test_decoding_with_parity_error(self):
        """Test decoding when the parity constraint is violated."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0])
        encoder(message)  # Should be [1, 0, 1, 0]

        # Create soft values with one flipped sign (least reliable bit)
        # For [1, 0, 1, 0], we use [-2.0, 2.0, -2.0, -0.1] where the last bit is wrong
        # but with low reliability (small absolute value)
        soft_received = torch.tensor([-2.0, 2.0, -2.0, -0.1])

        # Decode the soft values
        decoded = decoder(soft_received)

        # Verify that the decoded message matches the original
        # The Wagner algorithm should flip the least reliable bit (the last one)
        assert torch.all(decoded == message)

        # Test with a different bit being the least reliable
        soft_received = torch.tensor([-2.0, 0.1, -2.0, 2.0])  # Second bit is least reliable
        decoded = decoder(soft_received)

        # This should still decode correctly - just a different bit gets flipped
        assert decoded.shape == message.shape

    def test_decoding_with_return_errors(self):
        """Test decoding with return_errors=True."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create a message and encode it
        torch.tensor([1.0, 0.0, 1.0])

        # Create soft values with a parity error
        soft_received = torch.tensor([-2.0, 2.0, -2.0, -0.1])  # Last bit violates parity

        # Decode with return_errors=True
        decoded, errors = decoder(soft_received, return_errors=True)

        # Verify the error pattern
        assert errors.shape == soft_received.shape
        assert errors[3] == 1  # The last bit should be flipped
        assert torch.sum(errors) == 1  # Only one bit should be flipped

    def test_decoding_with_batch_dimension(self):
        """Test decoding with batch dimension."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create two sets of soft values (batch dimension)
        soft_received = torch.tensor([[-2.0, 2.0, -2.0, 2.0], [-2.0, 2.0, -2.0, -0.1]])  # No errors  # Parity error in least reliable bit

        # Expected messages
        expected = torch.tensor([[1, 0, 1], [1, 0, 1]])

        # Decode the soft values
        decoded = decoder(soft_received)

        # Verify the decoded messages
        assert torch.all(decoded == expected)

        # Test with return_errors=True
        decoded, errors = decoder(soft_received, return_errors=True)

        # Verify error patterns
        assert errors.shape == soft_received.shape
        assert torch.all(errors[0] == 0)  # No errors in first codeword
        assert errors[1, 3] == 1  # Error in least reliable bit of second codeword

    def test_invalid_input_dimensions(self):
        """Test decoding with invalid input dimensions."""
        # Create a (4,3) single parity-check code
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = WagnerSoftDecisionDecoder(encoder=encoder)

        # Create a soft received word with invalid length
        soft_received = torch.tensor([-2.0, 2.0, -2.0])  # Not a multiple of code length

        # Decoding should raise ValueError
        with pytest.raises(ValueError, match="Last dimension .* must be divisible by code length"):
            decoder(soft_received)
