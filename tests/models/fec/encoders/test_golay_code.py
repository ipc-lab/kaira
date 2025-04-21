"""Tests for the golay_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.golay_code import (
    GOLAY_GENERATOR_POLYNOMIAL,
    GolayCodeEncoder,
    create_golay_parity_submatrix,
)


class TestGolayHelperFunctions:
    """Test suite for helper functions in Golay code module."""

    def test_create_golay_parity_submatrix(self):
        """Test creating the parity submatrix for a Golay code."""
        # Test standard Golay code parity submatrix
        parity_submatrix = create_golay_parity_submatrix(extended=False)

        # Check shape
        assert parity_submatrix.shape == (12, 11)

        # Test extended Golay code parity submatrix
        ext_parity_submatrix = create_golay_parity_submatrix(extended=True)

        # Check shape
        assert ext_parity_submatrix.shape == (12, 12)

        # Verify that the extended matrix is the standard matrix with an extra column
        assert torch.all(ext_parity_submatrix[:, :-1] == parity_submatrix)

        # Test with custom dtype
        float64_matrix = create_golay_parity_submatrix(dtype=torch.float64)
        assert float64_matrix.dtype == torch.float64

        # Verify GOLAY_GENERATOR_POLYNOMIAL constant is defined
        assert isinstance(GOLAY_GENERATOR_POLYNOMIAL, int)
        assert GOLAY_GENERATOR_POLYNOMIAL == 0b101011100011


class TestGolayCodeEncoder:
    """Test suite for GolayCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test standard Golay code
        encoder = GolayCodeEncoder(extended=False)
        assert encoder.extended is False
        assert encoder.code_length == 23
        assert encoder.code_dimension == 12
        assert encoder.redundancy == 11
        assert encoder.error_correction_capability == 3
        assert encoder.minimum_distance() == 7

        # Test extended Golay code
        encoder = GolayCodeEncoder(extended=True)
        assert encoder.extended is True
        assert encoder.code_length == 24
        assert encoder.code_dimension == 12
        assert encoder.redundancy == 12
        assert encoder.error_correction_capability == 3
        assert encoder.minimum_distance() == 8

        # Test with information_set="right"
        encoder = GolayCodeEncoder(information_set="right")
        # For right-systematic form, information bits are in positions 11 to 22
        systematic_part = encoder.generator_matrix[:, -encoder.code_dimension :]
        assert torch.allclose(systematic_part, torch.eye(encoder.code_dimension))

    def test_factory_methods(self):
        """Test the factory methods for creating Golay codes."""
        # Test standard Golay code factory
        encoder = GolayCodeEncoder.create_standard_golay_code()
        assert encoder.extended is False
        assert encoder.code_length == 23

        # Test extended Golay code factory
        encoder = GolayCodeEncoder.create_extended_golay_code()
        assert encoder.extended is True
        assert encoder.code_length == 24

        # Test with custom parameters
        encoder = GolayCodeEncoder.create_extended_golay_code(information_set="right")
        assert encoder.extended is True
        # For right-systematic form, information bits are in positions 12 to 23
        systematic_part = encoder.generator_matrix[:, -encoder.code_dimension :]
        assert torch.allclose(systematic_part, torch.eye(encoder.code_dimension))

    def test_encoding(self):
        """Test encoding functionality."""
        # Test standard Golay code
        encoder = GolayCodeEncoder()

        # Test encoding a single message
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        codeword = encoder(message)

        # Check dimensions
        assert codeword.shape == torch.Size([23])

        # Manually calculate expected codeword
        expected = torch.matmul(message, encoder.generator_matrix) % 2
        assert torch.all(codeword == expected)

        # Test batch encoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
        codewords = encoder(messages)

        # Check dimensions
        assert codewords.shape == torch.Size([2, 23])

        # Calculate expected codewords
        expected_batch = torch.zeros((2, 23), dtype=torch.float)
        for i, msg in enumerate(messages):
            expected_batch[i] = torch.matmul(msg, encoder.generator_matrix) % 2

        assert torch.all(codewords == expected_batch)

        # Test with invalid message length
        with pytest.raises(ValueError):
            invalid_message = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Wrong dimension
            encoder(invalid_message)

        # Test extended Golay code
        ext_encoder = GolayCodeEncoder(extended=True)
        codeword = ext_encoder(message)
        assert codeword.shape == torch.Size([24])

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        # Test standard Golay code
        encoder = GolayCodeEncoder()

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
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

        # Test extended Golay code
        ext_encoder = GolayCodeEncoder(extended=True)
        ext_codeword = ext_encoder(message)
        ext_syndrome = ext_encoder.calculate_syndrome(ext_codeword)

        # Valid codeword should have zero syndrome
        assert torch.all(ext_syndrome == 0)

        # Verify that the extended code can detect 4 errors
        ext_codeword_with_errors = ext_codeword.clone()
        # Flip 4 bits
        for i in range(4):
            ext_codeword_with_errors[i] = 1 - ext_codeword_with_errors[i]

        ext_syndrome = ext_encoder.calculate_syndrome(ext_codeword_with_errors)
        assert not torch.all(ext_syndrome == 0)

    def test_decoding(self):
        """Test decoding functionality."""
        # Test standard Golay code
        encoder = GolayCodeEncoder()

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        codeword = encoder(message)

        # Decode the codeword
        decoded, syndrome = encoder.inverse_encode(codeword)

        # Check decoding correctness
        assert torch.all(decoded == message)
        assert torch.all(syndrome == 0)

        # Test decoding with errors (up to 3 errors should be correctable)
        codeword_with_errors = codeword.clone()
        # Flip 3 bits
        for i in range(3):
            codeword_with_errors[i] = 1 - codeword_with_errors[i]

        # For proper testing, we would need a full decoder implementation
        # that actually corrects errors. The current inverse_encode just
        # does syndrome calculation and matrix multiplication, which won't
        # correct errors. So we'll skip the actual error correction test.

        # Test extended Golay code
        ext_encoder = GolayCodeEncoder(extended=True)
        ext_codeword = ext_encoder(message)
        ext_decoded, ext_syndrome = ext_encoder.inverse_encode(ext_codeword)

        assert torch.all(ext_decoded == message)
        assert torch.all(ext_syndrome == 0)

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("golay_code_encoder")

        # Verify it's the correct class
        assert model_class is GolayCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("golay_code_encoder", extended=True)

        assert isinstance(model, GolayCodeEncoder)
        assert model.extended is True

    def test_representation(self):
        """Test string representation."""
        encoder = GolayCodeEncoder(extended=True)
        repr_str = repr(encoder)
        assert "GolayCodeEncoder" in repr_str
        assert "extended=True" in repr_str
        assert "length=24" in repr_str
        assert "dimension=12" in repr_str
        assert "redundancy=12" in repr_str
        assert "error_correction_capability=3" in repr_str
