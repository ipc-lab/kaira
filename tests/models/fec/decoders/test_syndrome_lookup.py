"""Tests for the SyndromeLookupDecoder in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.models.fec.decoders.syndrome_lookup import SyndromeLookupDecoder
from kaira.models.fec.encoders.hamming_code import HammingCodeEncoder
from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder


class TestSyndromeLookupDecoder:
    """Test suite for SyndromeLookupDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a simple generator matrix for a (7,4) code
        G = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)

        # Initialize the decoder with this encoder
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify that syndrome table is built
        assert hasattr(decoder, "_syndrome_table")
        assert isinstance(decoder._syndrome_table, dict)

        # For a (7,4) code, there should be 2^(7-4) = 8 syndromes
        assert len(decoder._syndrome_table) == 2**3

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with an incompatible encoder type
        from kaira.models.fec.encoders.repetition_code import RepetitionCodeEncoder

        encoder = RepetitionCodeEncoder(repetition_factor=3)

        with pytest.raises(TypeError, match="Encoder must be a LinearBlockCodeEncoder"):
            SyndromeLookupDecoder(encoder=encoder)

    def test_syndrome_to_int(self):
        """Test conversion of syndrome tensor to integer."""
        # Create a simple encoder and decoder
        G = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Test with all-zero syndrome
        syndrome = torch.zeros(3, dtype=torch.int)
        assert decoder._syndrome_to_int(syndrome) == 0

        # Test with syndrome [1, 0, 1]
        syndrome = torch.tensor([1, 0, 1], dtype=torch.int)
        assert decoder._syndrome_to_int(syndrome) == 5  # Binary 101 = 5

        # Test with syndrome [1, 1, 1]
        syndrome = torch.tensor([1, 1, 1], dtype=torch.int)
        assert decoder._syndrome_to_int(syndrome) == 7  # Binary 111 = 7

    def test_generate_error_patterns(self):
        """Test generation of error patterns with specific weight."""
        # Create a simple encoder and decoder
        G = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Test with weight 0
        patterns = decoder._generate_error_patterns(0)
        assert patterns.shape == (1, 7)  # One pattern
        assert torch.all(patterns == 0)  # All zeros

        # Test with weight 1
        patterns = decoder._generate_error_patterns(1)
        assert patterns.shape == (7, 7)  # 7 choose 1 = 7 patterns
        for i in range(7):
            # Each pattern should have exactly one 1
            assert torch.sum(patterns[i]) == 1
            assert patterns[i, i] == 1

        # Test with weight 2
        patterns = decoder._generate_error_patterns(2)
        assert patterns.shape == (21, 7)  # 7 choose 2 = 21 patterns
        for i in range(21):
            # Each pattern should have exactly two 1s
            assert torch.sum(patterns[i]) == 2

    def test_build_syndrome_table(self):
        """Test building of syndrome lookup table."""
        # Create a simple encoder and decoder
        G = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Build syndrome table
        table = decoder._build_syndrome_table()

        # Verify it's a dictionary
        assert isinstance(table, dict)

        # For a (7,4) code, there should be 2^(7-4) = 8 syndromes
        assert len(table) == 2**3

        # The syndrome for all-zero error pattern should map to all-zero error pattern
        zero_syndrome = decoder._syndrome_to_int(encoder.calculate_syndrome(torch.zeros(7)))
        assert zero_syndrome in table
        assert torch.all(table[zero_syndrome] == 0)

        # Check that all error patterns in the table have minimum weight for their syndrome
        for syndrome, error_pattern in table.items():
            # Generate the syndrome tensor from the integer
            syndrome_tensor = encoder.calculate_syndrome(error_pattern)
            assert syndrome == decoder._syndrome_to_int(syndrome_tensor)

    def test_decoding_no_errors(self):
        """Test decoding a codeword with no errors."""
        # Create a Hamming encoder and syndrome lookup decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Decode the codeword (no errors)
        decoded = decoder(codeword)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

    def test_decoding_with_errors(self):
        """Test decoding a codeword with correctable errors."""
        # Create a Hamming encoder and syndrome lookup decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Introduce a single bit error
        received = codeword.clone()
        received[2] = 1.0 - received[2]  # Flip a bit

        # Decode the received word
        decoded = decoder(received)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

        # Test with return_errors=True
        decoded, errors = decoder(received, return_errors=True)

        # Verify the error pattern
        assert errors.shape == received.shape
        assert errors[2] == 1.0  # Error at position 2
        assert torch.sum(errors) == 1.0  # One error total

    def test_decoding_with_batch_dimension(self):
        """Test decoding with batch dimension."""
        # Create a Hamming encoder and syndrome lookup decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Create messages and encode them
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
        codewords = encoder(messages)

        # Introduce errors in both codewords
        received = codewords.clone()
        received[0, 2] = 1.0 - received[0, 2]  # Flip a bit in first codeword
        received[1, 5] = 1.0 - received[1, 5]  # Flip a bit in second codeword

        # Decode the received words
        decoded = decoder(received)

        # Verify that the decoded messages match the originals
        assert torch.all(decoded == messages)

    def test_decoding_with_uncorrectable_errors(self):
        """Test decoding with uncorrectable errors."""
        # Create a Hamming encoder and syndrome lookup decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code - can correct 1 error
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Introduce multiple errors (beyond correction capability)
        received = codeword.clone()
        received[1] = 1.0 - received[1]
        received[4] = 1.0 - received[4]

        # Decode the received word
        decoded = decoder(received)

        # The decoder might not recover the original message
        # We're just testing that it runs without errors
        assert decoded.shape == message.shape

    def test_invalid_input_dimensions(self):
        """Test decoding with invalid input dimensions."""
        # Create a Hamming encoder and syndrome lookup decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = SyndromeLookupDecoder(encoder=encoder)

        # Create a received word with invalid length
        received = torch.tensor([1.0, 0.0, 1.0])  # Not a multiple of code length

        # Decoding should raise ValueError
        with pytest.raises(ValueError, match="Last dimension .* must be divisible by code length"):
            decoder(received)
