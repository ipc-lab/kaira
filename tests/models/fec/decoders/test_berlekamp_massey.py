"""Tests for the Berlekamp-Massey decoder in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.models.fec.decoders.berlekamp_massey import BerlekampMasseyDecoder
from kaira.models.fec.encoders.bch_code import BCHCodeEncoder


class TestBerlekampMasseyDecoder:
    """Test suite for BerlekampMasseyDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a BCH encoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2

        # Initialize the decoder with this encoder
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify properties are correctly set
        assert decoder.field is encoder._field
        assert decoder.t == encoder.error_correction_capability

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with an incompatible encoder type
        from kaira.models.fec.encoders.repetition_code import RepetitionCodeEncoder

        encoder = RepetitionCodeEncoder(repetition_factor=3)

        with pytest.raises(TypeError, match="Encoder must be a BCHCodeEncoder or ReedSolomonCodeEncoder"):
            BerlekampMasseyDecoder(encoder=encoder)

    def test_berlekamp_massey_algorithm(self):
        """Test the Berlekamp-Massey algorithm implementation."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a known syndrome sequence
        # This would typically come from a received word with errors
        field = encoder._field
        syndrome = [field(0), field(1), field(3), field(7)]

        # Run the Berlekamp-Massey algorithm
        error_locator = decoder.berlekamp_massey_algorithm(syndrome)

        # Verify the result is a list of field elements
        assert isinstance(error_locator, list)
        assert all(isinstance(coef, type(field(0))) for coef in error_locator)

        # The degree of the error locator polynomial should not exceed the error-correction capability
        assert len(error_locator) <= decoder.t + 1

    def test_find_error_locations(self):
        """Test the _find_error_locations method."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a known error locator polynomial
        # For example, sigma(x) = 1 + x + x^2 in GF(2^4)
        field = encoder._field
        error_locator_poly = [field(1), field(1), field(1)]

        # Find error locations
        error_positions = decoder._find_error_locations(error_locator_poly)

        # Verify the result is a list of integers
        assert isinstance(error_positions, list)
        assert all(isinstance(pos, int) for pos in error_positions)

        # The number of error positions should match the degree of the error locator polynomial
        assert len(error_positions) <= len(error_locator_poly) - 1

    def test_decoding_no_errors(self):
        """Test decoding a codeword with no errors."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Decode the codeword (no errors)
        decoded = decoder(codeword)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

    def test_decoding_with_errors(self):
        """Test decoding a codeword with correctable errors."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Introduce errors (up to the error correction capability)
        received = codeword.clone()
        received[2] = 1.0 - received[2]  # Flip a bit
        received[8] = 1.0 - received[8]  # Flip another bit

        # Decode the received word
        decoded = decoder(received)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

        # Test with return_errors=True
        decoded, errors = decoder(received, return_errors=True)

        # Verify the error pattern
        assert errors.shape == received.shape
        assert errors[2] == 1.0  # Error at position 2
        assert errors[8] == 1.0  # Error at position 8
        assert torch.sum(errors) == 2.0  # Two errors total

    def test_decoding_with_batch_dimension(self):
        """Test decoding with batch dimension."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create messages and encode them
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.1, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
        codewords = encoder(messages)

        # Introduce errors in both codewords
        received = codewords.clone()
        received[0, 2] = 1.0 - received[0, 2]  # Flip a bit in first codeword
        received[1, 5] = 1.0 - received[1, 5]  # Flip a bit in second codeword

        # Decode the received words
        decoded = decoder(received)

        # Verify that the decoded messages match the originals
        assert torch.all(decoded == messages)

    def test_decoding_with_too_many_errors(self):
        """Test decoding with more errors than the correction capability."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.1, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Introduce more errors than the correction capability
        received = codeword.clone()
        received[1] = 1.0 - received[1]
        received[4] = 1.0 - received[4]
        received[9] = 1.0 - received[9]  # t=2, so 3 errors should be too many

        # Decode the received word
        decoded = decoder(received)

        # The decoder should fail to correct all errors
        # The result might not match the original message
        # We're just testing that it runs without errors
        assert decoded.shape == message.shape

    def test_invalid_input_dimensions(self):
        """Test decoding with invalid input dimensions."""
        # Create a BCH encoder and decoder
        encoder = BCHCodeEncoder(mu=4, delta=5)  # BCH(15,7) code with t=2
        decoder = BerlekampMasseyDecoder(encoder=encoder)

        # Create a received word with invalid length
        received = torch.tensor([1.0, 0.0, 1.0])  # Not a multiple of code length

        # Decoding should raise ValueError
        with pytest.raises(ValueError, match="Last dimension .* must be divisible by code length"):
            decoder(received)
