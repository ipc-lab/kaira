"""Tests for the reed_solomon_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.reed_solomon_code import ReedSolomonCodeEncoder


class TestReedSolomonCodeEncoder:
    """Test suite for ReedSolomonCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test RS(15,11) code
        encoder = ReedSolomonCodeEncoder(mu=4, delta=5)
        assert encoder.mu == 4
        assert encoder.delta == 5
        assert encoder.code_length == 15  # 2^4 - 1
        assert encoder.code_dimension == 11  # n - (delta-1)
        assert encoder.redundancy == 4  # delta - 1
        assert encoder.error_correction_capability == 2  # (delta-1)/2

        # Test RS(15,7) code
        encoder = ReedSolomonCodeEncoder(mu=4, delta=9)
        assert encoder.code_length == 15
        assert encoder.code_dimension == 7
        assert encoder.redundancy == 8
        assert encoder.error_correction_capability == 4

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with invalid mu
        with pytest.raises(ValueError, match="'mu' must satisfy mu >= 2"):
            ReedSolomonCodeEncoder(mu=1, delta=5)

        # Test with invalid delta (too small)
        with pytest.raises(ValueError, match="'delta' must satisfy 2 <= delta <= 2\\^mu"):
            ReedSolomonCodeEncoder(mu=3, delta=1)

        # Test with invalid delta (too large)
        with pytest.raises(ValueError, match="'delta' must satisfy 2 <= delta <= 2\\^mu"):
            ReedSolomonCodeEncoder(mu=3, delta=9)  # max is 2^3 = 8

        # Test with redundancy >= length
        with pytest.raises(ValueError, match="The redundancy .* must be less than the code length"):
            # For mu=3, length=7, trying delta=8 would make redundancy=7
            ReedSolomonCodeEncoder(mu=3, delta=8)

    def test_encoding(self):
        """Test encoding functionality."""
        # Test with RS(15,11) code
        encoder = ReedSolomonCodeEncoder(mu=4, delta=5)

        # Test encoding a single message
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        codeword = encoder(message)

        # Check dimensions
        assert codeword.shape == torch.Size([15])

        # Manually calculate expected codeword
        expected = torch.matmul(message, encoder.generator_matrix) % 2
        assert torch.all(codeword == expected)

        # Test batch encoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
        codewords = encoder(messages)

        # Check dimensions
        assert codewords.shape == torch.Size([2, 15])

        # Calculate expected codewords
        expected_batch = torch.zeros((2, 15), dtype=torch.float)
        for i, msg in enumerate(messages):
            expected_batch[i] = torch.matmul(msg, encoder.generator_matrix) % 2

        assert torch.all(codewords == expected_batch)

        # Test with invalid message length
        with pytest.raises(ValueError):
            invalid_message = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Wrong dimension
            encoder(invalid_message)

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        # Test with RS(15,11) code
        encoder = ReedSolomonCodeEncoder(mu=4, delta=5)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
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

    def test_decoding(self):
        """Test decoding functionality."""
        # Test with RS(15,11) code
        encoder = ReedSolomonCodeEncoder(mu=4, delta=5)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        codeword = encoder(message)

        # Decode the codeword
        decoded, syndrome = encoder.inverse_encode(codeword)

        # Check decoding correctness
        assert torch.all(decoded == message)
        assert torch.all(syndrome == 0)

        # Test batch decoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
        codewords = encoder(messages)
        decoded_batch, syndromes = encoder.inverse_encode(codewords)

        assert torch.all(decoded_batch == messages)
        assert torch.all(syndromes == 0)

    def test_from_design_rate(self):
        """Test creating code from design rate."""
        # Test with mu=4 and target rate=0.5
        encoder = ReedSolomonCodeEncoder.from_design_rate(mu=4, target_rate=0.5)

        # Check that the rate is close to 0.5
        actual_rate = encoder.code_dimension / encoder.code_length
        assert abs(actual_rate - 0.5) < 0.1

        # Test with target rate=0.8
        encoder = ReedSolomonCodeEncoder.from_design_rate(mu=4, target_rate=0.8)
        actual_rate = encoder.code_dimension / encoder.code_length
        assert abs(actual_rate - 0.8) < 0.1

        # Test with invalid parameters
        with pytest.raises(ValueError, match="Invalid parameters"):
            ReedSolomonCodeEncoder.from_design_rate(mu=1, target_rate=0.5)

        with pytest.raises(ValueError, match="Invalid parameters"):
            ReedSolomonCodeEncoder.from_design_rate(mu=4, target_rate=0)

        with pytest.raises(ValueError, match="Invalid parameters"):
            ReedSolomonCodeEncoder.from_design_rate(mu=4, target_rate=1.5)

    def test_standard_codes(self):
        """Test standard code creation and properties."""
        # Get standard codes dictionary
        std_codes = ReedSolomonCodeEncoder.get_standard_codes()

        # Check that standard codes dictionary contains expected entries
        assert "RS(15,11)" in std_codes
        assert "RS(15,7)" in std_codes
        assert "RS(255,223)" in std_codes

        # Create a standard code
        encoder = ReedSolomonCodeEncoder.create_standard_code("RS(15,11)")
        assert encoder.mu == 4
        assert encoder.delta == 5
        assert encoder.code_length == 15
        assert encoder.code_dimension == 11

        # Test with invalid standard code name
        with pytest.raises(ValueError, match="Unknown standard code"):
            ReedSolomonCodeEncoder.create_standard_code("RS(100,50)")

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("reed_solomon_encoder")

        # Verify it's the correct class
        assert model_class is ReedSolomonCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("reed_solomon_encoder", mu=4, delta=5)

        assert isinstance(model, ReedSolomonCodeEncoder)
        assert model.mu == 4
        assert model.delta == 5

    def test_representation(self):
        """Test string representation."""
        encoder = ReedSolomonCodeEncoder(mu=4, delta=5)
        repr_str = repr(encoder)
        assert "ReedSolomonCodeEncoder" in repr_str
        assert "mu=4" in repr_str
        assert "delta=5" in repr_str
        assert "length=15" in repr_str
        assert "dimension=11" in repr_str
