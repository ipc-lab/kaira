"""Tests for the RepetitionCodeEncoder class in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.repetition_code import RepetitionCodeEncoder


class TestRepetitionCodeEncoder:
    """Test suite for RepetitionCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test with default repetition factor
        encoder = RepetitionCodeEncoder()
        assert encoder.repetition_factor == 3
        assert encoder.code_length == 3
        assert encoder.code_dimension == 1
        assert encoder.redundancy == 2
        assert encoder.code_rate == 1 / 3

        # Test with custom repetition factor
        encoder = RepetitionCodeEncoder(repetition_factor=5)
        assert encoder.repetition_factor == 5
        assert encoder.code_length == 5
        assert encoder.code_dimension == 1
        assert encoder.redundancy == 4
        assert encoder.code_rate == 1 / 5

        # Verify generator matrix
        expected_generator = torch.ones((1, 5), dtype=torch.float32)
        assert torch.all(encoder.generator_matrix == expected_generator)

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with non-positive repetition factor
        with pytest.raises(ValueError, match="Repetition factor must be a positive integer"):
            RepetitionCodeEncoder(repetition_factor=0)

        # Test with negative repetition factor
        with pytest.raises(ValueError, match="Repetition factor must be a positive integer"):
            RepetitionCodeEncoder(repetition_factor=-3)

    def test_encoding(self):
        """Test encoding functionality."""
        encoder = RepetitionCodeEncoder(repetition_factor=4)

        # Test encoding a single bit
        x = torch.tensor([1.0])
        encoded = encoder(x)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.all(encoded == expected)

        # Test encoding multiple bits
        x = torch.tensor([1.0, 0.0, 1.0])
        encoded = encoder(x)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        assert torch.all(encoded == expected)

        # Test with batch dimension
        x = torch.tensor([[1.0], [0.0]])
        encoded = encoder(x)
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        assert torch.all(encoded == expected)

    def test_decoding(self):
        """Test decoding functionality."""
        encoder = RepetitionCodeEncoder(repetition_factor=5)

        # Test decoding a perfect codeword
        x = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        decoded = encoder.inverse_encode(x)
        expected = torch.tensor([1.0])
        assert torch.all(decoded == expected)

        # Test decoding with errors (still recoverable)
        x = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])  # One error
        decoded = encoder.inverse_encode(x)
        expected = torch.tensor([1.0])  # Should correct to 1
        assert torch.all(decoded == expected)

        # Test decoding with batch dimension
        x = torch.tensor([[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
        decoded = encoder.inverse_encode(x)
        expected = torch.tensor([[1.0], [0.0]])  # Should correct to [1, 0]
        assert torch.all(decoded == expected)

        # Test decoding multiple codewords
        x = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        decoded = encoder.inverse_encode(x)
        expected = torch.tensor([1.0, 0.0])
        assert torch.all(decoded == expected)

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        encoder = RepetitionCodeEncoder(repetition_factor=3)

        # Test syndrome for valid codeword
        x = torch.tensor([1.0, 1.0, 1.0])
        syndrome = encoder.calculate_syndrome(x)
        expected = torch.zeros(syndrome.shape, dtype=syndrome.dtype)
        assert torch.all(syndrome == expected)

        # Test syndrome for invalid codeword
        x = torch.tensor([1.0, 0.0, 1.0])
        syndrome = encoder.calculate_syndrome(x)
        # Syndrome should be non-zero
        assert not torch.all(syndrome == 0)

        # Test syndrome with batch dimension
        x = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        syndrome = encoder.calculate_syndrome(x)
        assert syndrome.shape[0] == 2
        assert torch.all(syndrome[0] == 0)  # First codeword is valid
        assert not torch.all(syndrome[1] == 0)  # Second codeword is invalid

    def test_coset_leader_weight_distribution(self):
        """Test calculation of coset leader weight distribution."""
        # Test with odd repetition factor
        encoder = RepetitionCodeEncoder(repetition_factor=5)
        distribution = encoder.coset_leader_weight_distribution()

        # For n=5, the distribution should be [1, 5, 10, 0, 0, 0]
        expected = torch.tensor([1, 5, 10, 0, 0, 0], dtype=torch.int64)
        assert torch.all(distribution == expected)

        # Test with even repetition factor
        encoder = RepetitionCodeEncoder(repetition_factor=6)
        distribution = encoder.coset_leader_weight_distribution()

        # For n=6, the distribution should be [1, 6, 15, 10, 0, 0, 0]
        # where C(6,3)/2 = 10 for the middle weight
        expected = torch.tensor([1, 6, 15, 10, 0, 0, 0], dtype=torch.int64)
        assert torch.all(distribution == expected)

    def test_representation(self):
        """Test string representation."""
        encoder = RepetitionCodeEncoder(repetition_factor=7)
        repr_str = repr(encoder)
        assert "RepetitionCodeEncoder" in repr_str
        assert "repetition_factor=7" in repr_str
