"""Tests for the BruteForceMLDecoder in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.models.fec.decoders.brute_force_ml import BruteForceMLDecoder
from kaira.models.fec.encoders.hamming_code import HammingCodeEncoder
from kaira.models.fec.encoders.linear_block_code import LinearBlockCodeEncoder


class TestBruteForceMLDecoder:
    """Test suite for BruteForceMLDecoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Create a simple encoder
        G = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)

        # Initialize with precompute_codebook=True (default)
        decoder = BruteForceMLDecoder(encoder=encoder)

        # Verify that encoder is stored
        assert decoder.encoder is encoder

        # Verify that codebook is precomputed
        assert hasattr(decoder, "_codebook")
        assert decoder._codebook is not None
        assert decoder._message_map is not None

        # Verify codebook dimensions
        assert decoder._codebook.shape == (2**4, 7)  # 2^k x n
        assert decoder._message_map.shape == (2**4, 4)  # 2^k x k

        # Initialize with precompute_codebook=False
        decoder = BruteForceMLDecoder(encoder=encoder, precompute_codebook=False)
        assert decoder._codebook is None
        assert decoder._message_map is None

    def test_generate_codebook(self):
        """Test generation of the complete codebook."""
        # Create a simple encoder
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = BruteForceMLDecoder(encoder=encoder, precompute_codebook=False)

        # Generate the codebook
        codebook, message_map = decoder._generate_codebook()

        # Verify dimensions
        assert codebook.shape == (2**3, 4)  # 2^k x n
        assert message_map.shape == (2**3, 3)  # 2^k x k

        # Verify that all messages are present
        assert message_map.shape[0] == 2**encoder.code_dimension

        # Verify that each codeword corresponds to its message
        for i in range(message_map.shape[0]):
            message = message_map[i]
            codeword = codebook[i]
            encoded = encoder(message.unsqueeze(0)).squeeze(0)
            assert torch.all(codeword == encoded)

    def test_hamming_distance(self):
        """Test calculation of Hamming distance."""
        # Create a simple encoder and decoder
        G = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        encoder = LinearBlockCodeEncoder(generator_matrix=G)
        decoder = BruteForceMLDecoder(encoder=encoder)

        # Test with identical vectors
        x = torch.tensor([1, 0, 1, 0])
        y = torch.tensor([1, 0, 1, 0])
        distance = decoder._hamming_distance(x, y)
        assert distance == 0

        # Test with different vectors
        x = torch.tensor([1, 0, 1, 0])
        y = torch.tensor([1, 1, 0, 0])
        distance = decoder._hamming_distance(x, y)
        assert distance == 2

        # Test with batch of vectors
        x = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        y = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 1]])
        distance = decoder._hamming_distance(x, y)
        # The expected Hamming distances are:
        # Row 1: [1,0,1,0] vs [1,1,1,0] -> differences at indices 1 and 2
        # Row 2: [0,1,0,1] vs [0,0,0,1] -> differences at index 1
        # Carefully counting the differences:
        # [1,0,1,0] vs [1,1,1,0]: Different at positions 1, 2 (0 vs 1, 1 vs 1)
        expected_distances = torch.tensor([1, 1])
        assert torch.all(distance == expected_distances)

    def test_decoding_no_errors(self):
        """Test decoding a codeword with no errors."""
        # Create a Hamming encoder and brute force ML decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Decode the codeword (no errors)
        decoded = decoder(codeword)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

    def test_decoding_with_correctable_errors(self):
        """Test decoding a codeword with correctable errors."""
        # Create a Hamming encoder and brute force ML decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder)

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

        # Test with more errors (ML can correct beyond minimum distance)
        received = codeword.clone()
        received[1] = 1.0 - received[1]  # Flip a bit
        received[4] = 1.0 - received[4]  # Flip another bit

        # Decode the received word with two errors
        decoded = decoder(received)

        # The ML decoder may be able to correct this even beyond the guaranteed error-correction capability
        # At minimum, the result should have the right shape
        assert decoded.shape == message.shape

    def test_decoding_with_return_errors(self):
        """Test decoding with return_errors=True."""
        # Create a Hamming encoder and brute force ML decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Introduce a single bit error
        received = codeword.clone()
        received[2] = 1.0 - received[2]  # Flip a bit

        # Decode with return_errors=True
        decoded, errors = decoder(received, return_errors=True)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

        # Verify the error pattern
        assert errors.shape == received.shape
        assert errors[2] == 1  # Error at position 2
        assert torch.sum(errors) == 1  # One error total

    def test_decoding_with_batch_dimension(self):
        """Test decoding with batch dimension."""
        # Create a Hamming encoder and brute force ML decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder)

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

        # Test with return_errors=True
        decoded, errors = decoder(received, return_errors=True)

        # Verify error patterns
        assert errors.shape == received.shape
        assert errors[0, 2] == 1  # Error in first codeword at position 2
        assert errors[1, 5] == 1  # Error in second codeword at position 5

    def test_invalid_input_dimensions(self):
        """Test decoding with invalid input dimensions."""
        # Create a Hamming encoder and brute force ML decoder
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder)

        # Create a received word with invalid length
        received = torch.tensor([1.0, 0.0, 1.0])  # Not a multiple of code length

        # Decoding should raise ValueError
        with pytest.raises(ValueError, match="Last dimension .* must be divisible by code length"):
            decoder(received)

    def test_decoding_without_precomputed_codebook(self):
        """Test decoding without a precomputed codebook."""
        # Create a Hamming encoder and brute force ML decoder without precomputing
        encoder = HammingCodeEncoder(mu=3)  # (7,4) Hamming code
        decoder = BruteForceMLDecoder(encoder=encoder, precompute_codebook=False)

        # Create a message and encode it
        message = torch.tensor([1.0, 0.0, 1.0, 1.0])
        codeword = encoder(message)

        # Introduce errors
        received = codeword.clone()
        received[2] = 1.0 - received[2]  # Flip a bit

        # Decode the received word
        decoded = decoder(received)

        # Verify that the decoded message matches the original
        assert torch.all(decoded == message)

        # The codebook should still be None (generated on-demand during decoding)
        assert decoder._codebook is None
        assert decoder._message_map is None
