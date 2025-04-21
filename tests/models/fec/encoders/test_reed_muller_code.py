"""Tests for the reed_muller_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.reed_muller_code import (
    ReedMullerCodeEncoder,
    _generate_evaluation_vectors,
    _generate_reed_muller_matrix,
    calculate_reed_muller_dimension,
)


class TestReedMullerHelperFunctions:
    """Test suite for helper functions in Reed-Muller code module."""

    def test_generate_evaluation_vectors(self):
        """Test generating evaluation vectors for Reed-Muller code."""
        # Test with m=3
        m = 3
        vectors = _generate_evaluation_vectors(m)

        # Check dimensions
        assert vectors.shape == (m, 2**m)
        assert vectors.shape == (3, 8)

        # Expected evaluation vectors for m=3
        expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.int64)

        assert torch.all(vectors == expected)

        # Test with m=2
        m = 2
        vectors = _generate_evaluation_vectors(m)
        assert vectors.shape == (2, 4)

        expected = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.int64)

        assert torch.all(vectors == expected)

    def test_generate_reed_muller_matrix(self):
        """Test generating Reed-Muller generator matrix."""
        # Test RM(1,3) - first-order Reed-Muller code with m=3
        r, m = 1, 3
        matrix = _generate_reed_muller_matrix(r, m)

        # Calculate expected dimension
        expected_dimension = calculate_reed_muller_dimension(r, m)

        # Check dimensions
        assert matrix.shape == (expected_dimension, 2**m)
        assert matrix.shape == (4, 8)

        # Test with invalid parameters (r >= m)
        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            _generate_reed_muller_matrix(3, 3)

        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            _generate_reed_muller_matrix(4, 3)

        # Test special case: RM(0,3) - repetition code
        matrix = _generate_reed_muller_matrix(0, 3)
        assert matrix.shape == (1, 8)
        assert torch.all(matrix == 1)

    def test_calculate_reed_muller_dimension(self):
        """Test calculating Reed-Muller code dimension."""
        # Test some common Reed-Muller codes
        test_cases = [
            (0, 3, 1),  # RM(0,3): repetition code of length 8
            (1, 3, 4),  # RM(1,3): first-order Reed-Muller of length 8
            (2, 3, 7),  # RM(2,3): second-order Reed-Muller of length 8
            (0, 4, 1),  # RM(0,4): repetition code of length 16
            (1, 4, 5),  # RM(1,4): first-order Reed-Muller of length 16
            (2, 4, 11),  # RM(2,4): second-order Reed-Muller of length 16
            (3, 4, 15),  # RM(3,4): third-order Reed-Muller of length 16
        ]

        for r, m, expected_dimension in test_cases:
            dimension = calculate_reed_muller_dimension(r, m)
            assert dimension == expected_dimension, f"RM({r},{m}) should have dimension {expected_dimension}"

        # Test with invalid parameters
        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            calculate_reed_muller_dimension(3, 3)

        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            calculate_reed_muller_dimension(4, 3)


class TestReedMullerCodeEncoder:
    """Test suite for ReedMullerCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)
        assert encoder.order == 1
        assert encoder.length_param == 3
        assert encoder.code_length == 8  # 2^3
        assert encoder.code_dimension == 4  # (3 choose 0) + (3 choose 1)
        assert encoder.minimum_distance == 4  # 2^(3-1)

        # Test RM(0,4) - repetition code
        encoder = ReedMullerCodeEncoder(order=0, length_param=4)
        assert encoder.code_length == 16  # 2^4
        assert encoder.code_dimension == 1  # (4 choose 0)
        assert encoder.minimum_distance == 16  # 2^(4-0)

        # Test RM(2,3) - extended Hamming code
        encoder = ReedMullerCodeEncoder(order=2, length_param=3)
        assert encoder.code_length == 8  # 2^3
        assert encoder.code_dimension == 7  # (3 choose 0) + (3 choose 1) + (3 choose 2)
        assert encoder.minimum_distance == 2  # 2^(3-2)

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with invalid parameters (r >= m)
        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            ReedMullerCodeEncoder(order=3, length_param=3)

        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            ReedMullerCodeEncoder(order=4, length_param=3)

        # Test with negative order
        with pytest.raises(ValueError, match="Parameters must satisfy 0 ≤ r < m"):
            ReedMullerCodeEncoder(order=-1, length_param=3)

    def test_from_parameters(self):
        """Test creating an encoder using from_parameters class method."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder.from_parameters(order=1, length_param=3)
        assert encoder.order == 1
        assert encoder.length_param == 3
        assert encoder.code_length == 8
        assert encoder.code_dimension == 4

    def test_encoding(self):
        """Test encoding functionality."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)

        # Test encoding a message
        message = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 4-bit message
        codeword = encoder(message)

        # Check dimensions
        assert codeword.shape == torch.Size([8])  # 8-bit codeword

        # Manually calculate expected codeword
        expected = torch.matmul(message, encoder.generator_matrix) % 2
        assert torch.all(codeword == expected)

        # Test batch encoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])
        codewords = encoder(messages)

        # Check dimensions
        assert codewords.shape == torch.Size([2, 8])

        # Calculate expected codewords
        expected_batch = torch.zeros((2, 8), dtype=torch.float)
        for i, msg in enumerate(messages):
            expected_batch[i] = torch.matmul(msg, encoder.generator_matrix) % 2

        assert torch.all(codewords == expected_batch)

        # Test with invalid message length
        with pytest.raises(ValueError):
            invalid_message = torch.tensor([1.0, 0.0, 1.0])  # Wrong dimension
            encoder(invalid_message)

    def test_decoding(self):
        """Test decoding functionality."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Decode the codeword
        decoded, syndrome = encoder.inverse_encode(codeword)

        # Check decoding correctness
        assert torch.all(decoded == message)
        assert torch.all(syndrome == 0)

        # Test with bit errors (may or may not be corrected depending on error pattern)
        codeword_with_error = codeword.clone()
        codeword_with_error[0] = 1 - codeword_with_error[0]  # Flip first bit

        decoded_with_error, syndrome = encoder.inverse_encode(codeword_with_error)

        # Syndrome should be non-zero
        assert not torch.all(syndrome == 0)

        # Test batch decoding
        codewords_batch = torch.stack([codeword, codeword_with_error])
        decoded_batch, syndromes = encoder.inverse_encode(codewords_batch)

        # First syndrome should be zero, second non-zero
        assert torch.all(syndromes[0] == 0)
        assert not torch.all(syndromes[1] == 0)

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Calculate syndrome
        syndrome = encoder.calculate_syndrome(codeword)

        # Valid codeword should have zero syndrome
        assert torch.all(syndrome == 0)

        # Test with bit error
        codeword_with_error = codeword.clone()
        codeword_with_error[0] = 1 - codeword_with_error[0]  # Flip first bit

        syndrome = encoder.calculate_syndrome(codeword_with_error)

        # Invalid codeword should have non-zero syndrome
        assert not torch.all(syndrome == 0)

    def test_get_reed_partitions(self):
        """Test getting Reed partitions."""
        # Test RM(1,3)
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)

        # Get Reed partitions
        partitions = encoder.get_reed_partitions()

        # Check that we have the correct number of partitions
        # For RM(1,3), number of partitions is (3 choose 0) + (3 choose 1) = 1 + 3 = 4
        assert len(partitions) == 4

        # Test RM(2,3)
        encoder = ReedMullerCodeEncoder(order=2, length_param=3)
        partitions = encoder.get_reed_partitions()

        # For RM(2,3), number of partitions is (3 choose 0) + (3 choose 1) + (3 choose 2) = 1 + 3 + 3 = 7
        assert len(partitions) == 7

        # Check that each partition has correct shape
        for partition in partitions:
            # Each partition should be a valid reshaping of the codeword indices
            assert partition.numel() == encoder.code_length

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("reed_muller_code_encoder")

        # Verify it's the correct class
        assert model_class is ReedMullerCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("reed_muller_code_encoder", order=1, length_param=3)

        assert isinstance(model, ReedMullerCodeEncoder)
        assert model.order == 1
        assert model.length_param == 3

    def test_representation(self):
        """Test string representation."""
        encoder = ReedMullerCodeEncoder(order=1, length_param=3)
        repr_str = repr(encoder)
        assert "ReedMullerCodeEncoder" in repr_str
        assert "order=1" in repr_str
        assert "length_param=3" in repr_str
        assert "length=8" in repr_str
        assert "dimension=4" in repr_str
