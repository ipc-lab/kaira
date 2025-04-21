"""Tests for the linear_block_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.linear_block_code import (
    LinearBlockCodeEncoder,
    compute_null_space_matrix,
    compute_reduced_row_echelon_form,
    compute_right_pseudo_inverse,
)


class TestLinearBlockCodeEncoderHelperFunctions:
    """Test suite for helper functions in linear block code encoder module."""

    def test_compute_null_space_matrix(self):
        """Test computing the null space matrix of a given matrix."""
        # Define a generator matrix with known null space
        generator = torch.tensor([[1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1]], dtype=torch.float)

        # Compute the null space
        null_space = compute_null_space_matrix(generator)

        # Verify that G * H^T = 0 (null space property)
        result = torch.matmul(generator, null_space.T)
        assert torch.all(result % 2 == 0)

        # Test empty null space
        identity = torch.eye(3)
        null_space = compute_null_space_matrix(identity)
        assert null_space.shape[0] == 0

    def test_compute_reduced_row_echelon_form(self):
        """Test computing the reduced row echelon form."""
        # Define a matrix
        matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

        # Compute RREF
        rref = compute_reduced_row_echelon_form(matrix)

        # Expected result (binary form)
        expected = torch.tensor([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=torch.float)

        assert torch.allclose(rref, expected)

        # Test with identity matrix
        identity = torch.eye(3)
        rref = compute_reduced_row_echelon_form(identity)
        assert torch.allclose(rref, identity)

    def test_compute_right_pseudo_inverse(self):
        """Test computing the right pseudo-inverse."""
        # Define a generator matrix
        generator = torch.tensor([[1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1]], dtype=torch.float)

        # Compute the right pseudo-inverse
        right_inv = compute_right_pseudo_inverse(generator)

        # Verify that G * G_right_inv = I
        result = torch.matmul(generator, right_inv) % 2
        expected = torch.eye(generator.shape[0])
        assert torch.allclose(result, expected)


class TestLinearBlockCodeEncoder:
    """Test suite for LinearBlockCodeEncoder class."""

    def setup_method(self):
        """Set up the encoder for testing."""
        # Define a generator matrix for a (7,3) code
        self.generator = torch.tensor([[1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1]], dtype=torch.float)

        # Create an encoder instance
        self.encoder = LinearBlockCodeEncoder(self.generator)

    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test valid initialization
        encoder = LinearBlockCodeEncoder(self.generator)
        assert encoder.code_length == 7
        assert encoder.code_dimension == 3
        assert encoder.redundancy == 4
        assert encoder.code_rate == 3 / 7

        # Verify properties
        assert torch.allclose(encoder.parity_check_matrix, encoder.check_matrix)

    def test_encoding(self):
        """Test encoding functionality."""
        # Single message
        message = torch.tensor([1, 0, 1], dtype=torch.float)
        codeword = self.encoder(message)

        # Expected codeword calculation
        expected = torch.matmul(message, self.generator) % 2
        assert torch.allclose(codeword, expected)

        # Batch of messages
        batch = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.float)
        codewords = self.encoder(batch)

        expected_batch = torch.zeros((3, 7), dtype=torch.float)
        for i, msg in enumerate(batch):
            expected_batch[i] = torch.matmul(msg, self.generator) % 2

        assert torch.allclose(codewords, expected_batch)

        # Multiple blocks
        multiple_blocks = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.float)
        encoded = self.encoder(multiple_blocks)

        # Should encode each block of 3 separately
        expected_blocks = torch.cat([torch.matmul(multiple_blocks[:3], self.generator), torch.matmul(multiple_blocks[3:], self.generator)]) % 2

        assert torch.allclose(encoded, expected_blocks)

        # Test with invalid message length
        with pytest.raises(ValueError):
            invalid_message = torch.tensor([1, 0, 1, 0], dtype=torch.float)
            self.encoder(invalid_message)

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        # Valid codeword (should have zero syndrome)
        message = torch.tensor([1, 0, 1], dtype=torch.float)
        codeword = self.encoder(message)
        syndrome = self.encoder.calculate_syndrome(codeword)

        assert torch.all(syndrome == 0)

        # Codeword with error
        codeword_with_error = codeword.clone()
        codeword_with_error[0] = 1 - codeword_with_error[0]  # Flip first bit
        syndrome = self.encoder.calculate_syndrome(codeword_with_error)

        assert not torch.all(syndrome == 0)

        # Test with invalid codeword length
        with pytest.raises(ValueError):
            invalid_codeword = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float)
            self.encoder.calculate_syndrome(invalid_codeword)

    def test_inverse_encoding(self):
        """Test inverse encoding (decoding) functionality."""
        # Encode a message
        message = torch.tensor([1, 0, 1], dtype=torch.float)
        codeword = self.encoder(message)

        # Decode the codeword
        decoded, syndrome = self.encoder.inverse_encode(codeword)

        # Verify correct decoding
        assert torch.allclose(decoded, message)
        assert torch.all(syndrome == 0)

        # Test decoding with error
        codeword_with_error = codeword.clone()
        codeword_with_error[0] = 1 - codeword_with_error[0]  # Flip first bit
        decoded, syndrome = self.encoder.inverse_encode(codeword_with_error)

        # Note: decoded might not match original message due to error
        assert not torch.all(syndrome == 0)

        # Test batch decoding
        batch = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.float)
        codewords = self.encoder(batch)
        decoded_batch, syndromes = self.encoder.inverse_encode(codewords)

        assert torch.allclose(decoded_batch, batch)
        assert torch.all(syndromes == 0)

        # Test with invalid codeword length
        with pytest.raises(ValueError):
            invalid_codeword = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float)
            self.encoder.inverse_encode(invalid_codeword)

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("linear_block_code_encoder")

        # Verify it's the correct class
        assert model_class is LinearBlockCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("linear_block_code_encoder", generator_matrix=self.generator)

        assert isinstance(model, LinearBlockCodeEncoder)
