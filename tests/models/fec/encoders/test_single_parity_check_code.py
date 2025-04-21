"""Tests for the single_parity_check_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.single_parity_check_code import (
    SingleParityCheckCodeEncoder,
    _generate_single_parity_check_matrix,
)


class TestSingleParityCheckHelperFunctions:
    """Test suite for helper functions in single parity check code module."""

    def test_generate_single_parity_check_matrix(self):
        """Test generating a single parity-check generator matrix."""
        # Test with dimension 3
        generator = _generate_single_parity_check_matrix(3)

        # Expected generator matrix for dimension 3
        expected = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.int64)

        assert torch.all(generator == expected)

        # Test with dimension 1
        generator = _generate_single_parity_check_matrix(1)
        expected = torch.tensor([[1, 1]], dtype=torch.int64)
        assert torch.all(generator == expected)


class TestSingleParityCheckCodeEncoder:
    """Test suite for SingleParityCheckCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid and invalid parameters."""
        # Test valid initialization
        encoder = SingleParityCheckCodeEncoder(3)
        assert encoder.code_length == 4
        assert encoder.code_dimension == 3
        assert encoder.redundancy == 1
        assert encoder.minimum_distance == 2

        # Test with invalid dimension
        with pytest.raises(ValueError):
            SingleParityCheckCodeEncoder(0)

        with pytest.raises(ValueError):
            SingleParityCheckCodeEncoder(-1)

    def test_from_parameters(self):
        """Test creating an encoder using from_parameters class method."""
        # Test with dimension 5
        encoder = SingleParityCheckCodeEncoder.from_parameters(5)
        assert encoder.code_length == 6
        assert encoder.code_dimension == 5
        assert encoder.dimension == 5

    def test_from_length(self):
        """Test creating an encoder using from_length class method."""
        # Test with length 6
        encoder = SingleParityCheckCodeEncoder.from_length(6)
        assert encoder.code_length == 6
        assert encoder.code_dimension == 5
        assert encoder.dimension == 5

        # Test with invalid length
        with pytest.raises(ValueError):
            SingleParityCheckCodeEncoder.from_length(1)

    def test_encoding(self):
        """Test encoding functionality."""
        # Create encoder with dimension 3
        encoder = SingleParityCheckCodeEncoder(3)

        # Single message
        message = torch.tensor([1, 0, 1], dtype=torch.float)
        codeword = encoder(message)

        # Expected codeword: [1, 0, 1, 0] - ensures even number of 1s
        expected = torch.tensor([1, 0, 1, 0], dtype=torch.float)
        assert torch.all(codeword == expected)

        # Test with all zeros (even parity)
        message = torch.tensor([0, 0, 0], dtype=torch.float)
        codeword = encoder(message)
        expected = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        assert torch.all(codeword == expected)

        # Test with all ones (odd parity, needs 1 as parity bit)
        message = torch.tensor([1, 1, 1], dtype=torch.float)
        codeword = encoder(message)
        expected = torch.tensor([1, 1, 1, 1], dtype=torch.float)
        assert torch.all(codeword == expected)

        # Test batch encoding
        batch = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.float)
        codewords = encoder(batch)

        expected_batch = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float)

        assert torch.all(codewords == expected_batch)

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        # Create encoder with dimension 3
        encoder = SingleParityCheckCodeEncoder(3)

        # Valid codeword (even number of 1s)
        codeword = torch.tensor([1, 0, 1, 0], dtype=torch.float)
        syndrome = encoder.calculate_syndrome(codeword)

        # Syndrome should be 0 for valid codeword
        assert torch.all(syndrome == 0)

        # Invalid codeword (odd number of 1s)
        invalid_codeword = torch.tensor([1, 0, 1, 1], dtype=torch.float)
        syndrome = encoder.calculate_syndrome(invalid_codeword)

        # Syndrome should be non-zero for invalid codeword
        assert not torch.all(syndrome == 0)

        # Test batch syndrome calculation
        batch = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.float)  # valid  # invalid  # valid

        syndromes = encoder.calculate_syndrome(batch)
        expected = torch.tensor([[0], [1], [0]], dtype=torch.float)  # valid codeword syndrome  # invalid codeword syndrome  # valid codeword syndrome

        assert torch.all(syndromes == expected)

    def test_decoding(self):
        """Test decoding functionality."""
        # Create encoder with dimension 3
        encoder = SingleParityCheckCodeEncoder(3)

        # Encode a message
        message = torch.tensor([1, 0, 1], dtype=torch.float)
        codeword = encoder(message)

        # Decode the codeword
        decoded, syndrome = encoder.inverse_encode(codeword)

        # Verify correct decoding
        assert torch.all(decoded == message)
        assert torch.all(syndrome == 0)

        # Test with invalid codeword (error introduced)
        codeword_with_error = torch.tensor([1, 0, 1, 1], dtype=torch.float)
        decoded, syndrome = encoder.inverse_encode(codeword_with_error)

        # Syndrome should detect the error
        assert not torch.all(syndrome == 0)

        # Since single parity check can only detect (not correct) errors,
        # the decoded message might not match the original

        # Test batch decoding
        batch = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.float)  # valid codeword  # valid codeword  # invalid codeword

        decoded_batch, syndromes = encoder.inverse_encode(batch)

        # Expected syndromes
        expected_syndromes = torch.tensor([[0], [0], [1]], dtype=torch.float)  # valid  # valid  # invalid

        assert torch.all(syndromes == expected_syndromes)

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("single_parity_check_code_encoder")

        # Verify it's the correct class
        assert model_class is SingleParityCheckCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("single_parity_check_code_encoder", dimension=4)

        assert isinstance(model, SingleParityCheckCodeEncoder)
        assert model.code_dimension == 4
        assert model.code_length == 5
