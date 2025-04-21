"""Tests for the bch_code module in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.algebra import BinaryPolynomial
from kaira.models.fec.encoders.bch_code import (
    BCHCodeEncoder,
    compute_bch_generator_polynomial,
    create_bch_generator_matrix,
    get_valid_bose_distances,
    is_bose_distance,
)


class TestBCHHelperFunctions:
    """Test suite for helper functions in BCH code module."""

    def test_compute_bch_generator_polynomial(self):
        """Test computing the generator polynomial for a BCH code."""
        # Test for BCH(15,7) code with mu=4, delta=5
        generator_poly = compute_bch_generator_polynomial(4, 5)

        # Verify it's a BinaryPolynomial
        assert isinstance(generator_poly, BinaryPolynomial)

        # Verify the degree is correct (4-bit messages, 15-bit codewords, so 8-bit redundancy)
        assert generator_poly.degree == 8

        # Test for BCH(7,4) code with mu=3, delta=3 (equivalent to Hamming code)
        generator_poly = compute_bch_generator_polynomial(3, 3)
        assert generator_poly.degree == 3

        # Test caching - second call should be faster
        generator_poly2 = compute_bch_generator_polynomial(4, 5)
        assert generator_poly2 is generator_poly  # Should be the same object due to caching

    def test_is_bose_distance(self):
        """Test checking if delta is a valid Bose distance."""
        # Test known valid Bose distances
        assert is_bose_distance(3, 2) is True  # δ=2 is always valid
        assert is_bose_distance(3, 3) is True  # δ=3 is valid for mu=3
        assert is_bose_distance(3, 7) is True  # Maximum δ=2^μ-1 is always valid

        # Test with invalid parameters
        assert is_bose_distance(3, 8) is False  # δ > 2^μ-1 is invalid
        assert is_bose_distance(3, 1) is False  # δ < 2 is invalid

        # Check caching works
        # The function should return quickly on second call
        assert is_bose_distance(4, 5) is True

    def test_get_valid_bose_distances(self):
        """Test getting all valid Bose distances for a given mu."""
        # Test for mu=3
        valid_distances = get_valid_bose_distances(3)
        # Expected valid distances for mu=3 are 2, 3, 5, 7
        assert set(valid_distances) == {2, 3, 5, 7}

        # Test for mu=4
        valid_distances = get_valid_bose_distances(4)
        # Should include key values like 2, 3, 5, and 15
        assert 2 in valid_distances
        assert 3 in valid_distances
        assert 5 in valid_distances
        assert 15 in valid_distances  # Maximum δ=2^μ-1 is always valid

        # Test caching - second call should be faster
        valid_distances2 = get_valid_bose_distances(3)
        assert valid_distances2 == valid_distances[: len(valid_distances2)]

    def test_create_bch_generator_matrix(self):
        """Test creating the generator matrix for a BCH code."""
        # Test for BCH(15,7) code
        mu = 4
        delta = 5
        generator_poly = compute_bch_generator_polynomial(mu, delta)

        # Create generator matrix
        gen_matrix = create_bch_generator_matrix(2**mu - 1, generator_poly)

        # Check dimensions
        expected_dim = 2**mu - 1 - generator_poly.degree
        assert gen_matrix.shape == (expected_dim, 2**mu - 1)
        assert gen_matrix.shape == (7, 15)

        # Check systematic form (should have identity in the first k columns)
        identity_part = gen_matrix[:, :expected_dim]
        expected_identity = torch.eye(expected_dim)
        assert torch.allclose(identity_part, expected_identity)


class TestBCHCodeEncoder:
    """Test suite for BCHCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test BCH(15,7) code
        encoder = BCHCodeEncoder(mu=4, delta=5)
        assert encoder.mu == 4
        assert encoder.delta == 5
        assert encoder.code_length == 15  # 2^4 - 1
        assert encoder.code_dimension == 7  # Standard dimension for BCH(15,7)
        assert encoder.redundancy == 8  # n - k
        assert encoder.error_correction_capability == 2  # (delta-1)/2
        assert encoder.minimum_distance() >= 5  # >= delta for BCH codes

        # Test standard Hamming code as BCH code
        encoder = BCHCodeEncoder(mu=3, delta=3)
        assert encoder.code_length == 7  # 2^3 - 1
        assert encoder.code_dimension == 4  # Standard dimension for BCH(7,4)
        assert encoder.redundancy == 3  # n - k
        assert encoder.error_correction_capability == 1  # Can correct 1 error

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with invalid mu
        with pytest.raises(ValueError, match="'mu' must satisfy mu >= 2"):
            BCHCodeEncoder(mu=1, delta=3)

        # Test with invalid delta (too small)
        with pytest.raises(ValueError, match="'delta' must satisfy 2 <= delta <= 2\\*\\*mu - 1"):
            BCHCodeEncoder(mu=3, delta=1)

        # Test with invalid delta (too large)
        with pytest.raises(ValueError, match="'delta' must satisfy 2 <= delta <= 2\\*\\*mu - 1"):
            BCHCodeEncoder(mu=3, delta=8)  # max is 2^3-1 = 7

        # Test with non-Bose distance
        # For mu=3, valid distances are {2, 3, 5, 7}
        with pytest.raises(ValueError, match="'delta' must be a Bose distance"):
            BCHCodeEncoder(mu=3, delta=4)  # 4 is not a valid Bose distance for mu=3

    def test_encoding(self):
        """Test encoding functionality."""
        # Test with BCH(15,7) code
        encoder = BCHCodeEncoder(mu=4, delta=5)

        # Test encoding a single message
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Check dimensions
        assert codeword.shape == torch.Size([15])

        # Manually calculate expected codeword
        expected = torch.matmul(message, encoder.generator_matrix) % 2
        assert torch.all(codeword == expected)

        # Test batch encoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
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
        # Test with BCH(15,7) code
        encoder = BCHCodeEncoder(mu=4, delta=5)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Calculate standard syndrome
        syndrome = encoder.calculate_syndrome(codeword)

        # Valid codeword should have zero syndrome
        assert torch.all(syndrome == 0)

        # Test BCH-specific syndrome calculation
        bch_syndrome = encoder.bch_syndrome(codeword)

        # Each syndrome value should be zero for valid codeword
        assert len(bch_syndrome) == 1  # Single codeword
        for _, value in bch_syndrome[0]:
            assert value == 0

        # Test with bit error
        codeword_with_error = codeword.clone()
        codeword_with_error[0] = 1 - codeword_with_error[0]  # Flip first bit

        syndrome = encoder.calculate_syndrome(codeword_with_error)
        # Invalid codeword should have non-zero syndrome
        assert not torch.all(syndrome == 0)

        bch_syndrome = encoder.bch_syndrome(codeword_with_error)
        # At least one syndrome value should be non-zero
        has_nonzero = False
        for _, value in bch_syndrome[0]:
            if value != 0:
                has_nonzero = True
                break
        assert has_nonzero

    def test_decoding(self):
        """Test decoding functionality."""
        # Test with BCH(15,7) code
        encoder = BCHCodeEncoder(mu=4, delta=5)

        # Create a valid codeword
        message = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        codeword = encoder(message)

        # Decode the codeword
        decoded, syndrome = encoder.inverse_encode(codeword)

        # Check decoding correctness
        assert torch.all(decoded == message)
        assert torch.all(syndrome == 0)

        # Test batch decoding
        messages = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        codewords = encoder(messages)
        decoded_batch, syndromes = encoder.inverse_encode(codewords)

        assert torch.all(decoded_batch == messages)
        assert torch.all(syndromes == 0)

    def test_from_design_rate(self):
        """Test creating code from design rate."""
        # Test with mu=4 and target rate=0.5
        encoder = BCHCodeEncoder.from_design_rate(mu=4, target_rate=0.5)

        # Check that the rate is close to 0.5
        actual_rate = encoder.code_dimension / encoder.code_length
        assert abs(actual_rate - 0.5) < 0.1

        # Test with invalid parameters
        with pytest.raises(ValueError, match="'mu' must satisfy mu >= 2"):
            BCHCodeEncoder.from_design_rate(mu=1, target_rate=0.5)

        with pytest.raises(ValueError, match="'target_rate' must be between 0 and 1"):
            BCHCodeEncoder.from_design_rate(mu=4, target_rate=0)

        with pytest.raises(ValueError, match="'target_rate' must be between 0 and 1"):
            BCHCodeEncoder.from_design_rate(mu=4, target_rate=1.5)

    def test_standard_codes(self):
        """Test standard code creation and properties."""
        # Get standard codes dictionary
        std_codes = BCHCodeEncoder.get_standard_codes()

        # Check that standard codes dictionary contains expected entries
        assert "BCH(7,4)" in std_codes
        assert "BCH(15,7)" in std_codes
        assert "BCH(31,16)" in std_codes

        # Create a standard code
        encoder = BCHCodeEncoder.create_standard_code("BCH(15,7)")
        assert encoder.mu == 4
        assert encoder.delta == 5
        assert encoder.code_length == 15
        assert encoder.code_dimension == 7

        # Test with invalid standard code name
        with pytest.raises(ValueError, match="Unknown standard code"):
            BCHCodeEncoder.create_standard_code("BCH(100,50)")

    def test_model_registry(self):
        """Test that the encoder is properly registered with the model registry."""
        from kaira.models.registry import ModelRegistry

        # Get the registered model class
        model_class = ModelRegistry.get_model_cls("bch_code_encoder")

        # Verify it's the correct class
        assert model_class is BCHCodeEncoder

        # Create an instance through the registry
        model = ModelRegistry.create("bch_code_encoder", mu=3, delta=3)

        assert isinstance(model, BCHCodeEncoder)
        assert model.mu == 3
        assert model.delta == 3

    def test_representation(self):
        """Test string representation."""
        encoder = BCHCodeEncoder(mu=4, delta=5)
        repr_str = repr(encoder)
        assert "BCHCodeEncoder" in repr_str
        assert "mu=4" in repr_str
        assert "delta=5" in repr_str
        assert "length=15" in repr_str
        assert "dimension=7" in repr_str
        assert "t=2" in repr_str
