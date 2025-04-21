"""Tests for the HammingCodeEncoder class in kaira.models.fec.encoders package."""

import pytest
import torch

from kaira.models.fec.encoders.hamming_code import (
    HammingCodeEncoder,
    create_hamming_parity_submatrix,
)


class TestHammingCodeUtils:
    """Test suite for Hamming code utility functions."""

    def test_create_hamming_parity_submatrix(self):
        """Test the create_hamming_parity_submatrix function."""
        # Test with mu=3 (standard Hamming code)
        mu = 3
        parity_submatrix = create_hamming_parity_submatrix(mu=mu)

        # Check dimensions
        # For mu=3, k=4, m=3
        k = 2**mu - mu - 1
        m = mu
        assert parity_submatrix.shape == (k, m)
        assert parity_submatrix.shape == (4, 3)

        # Test with extended=True
        parity_submatrix = create_hamming_parity_submatrix(mu=mu, extended=True)
        assert parity_submatrix.shape == (k, m + 1)  # Extra column for overall parity

        # Test with custom dtype
        parity_submatrix = create_hamming_parity_submatrix(mu=mu, dtype=torch.int32)
        assert parity_submatrix.dtype == torch.int32

        # Test with invalid mu (should raise ValueError)
        with pytest.raises(ValueError, match="'mu' must be at least 2"):
            create_hamming_parity_submatrix(mu=1)

        # Test with larger mu value
        mu = 4
        parity_submatrix = create_hamming_parity_submatrix(mu=mu)
        k = 2**mu - mu - 1
        m = mu
        assert parity_submatrix.shape == (k, m)
        assert parity_submatrix.shape == (11, 4)


class TestHammingCodeEncoder:
    """Test suite for HammingCodeEncoder class."""

    def test_initialization(self):
        """Test initialization with valid parameters."""
        # Test with mu=3 (standard Hamming code)
        encoder = HammingCodeEncoder(mu=3)
        assert encoder.mu == 3
        assert encoder.extended is False
        assert encoder.code_length == 7  # 2^3 - 1
        assert encoder.code_dimension == 4  # 2^3 - 3 - 1
        assert encoder.redundancy == 3  # mu
        assert encoder.code_rate == 4 / 7

        # Test with extended=True
        encoder = HammingCodeEncoder(mu=3, extended=True)
        assert encoder.extended is True
        assert encoder.code_length == 8  # 2^3
        assert encoder.code_dimension == 4  # Same as standard
        assert encoder.redundancy == 4  # mu + 1
        assert encoder.code_rate == 4 / 8

        # Test with larger mu
        encoder = HammingCodeEncoder(mu=4)
        assert encoder.code_length == 15  # 2^4 - 1
        assert encoder.code_dimension == 11  # 2^4 - 4 - 1
        assert encoder.redundancy == 4  # mu

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises appropriate errors."""
        # Test with invalid mu
        with pytest.raises(ValueError, match="'mu' must be at least 2"):
            HammingCodeEncoder(mu=1)

    def test_minimum_distance(self):
        """Test minimum_distance property."""
        # Standard Hamming code has minimum distance 3
        encoder = HammingCodeEncoder(mu=3)
        assert encoder.minimum_distance() == 3

        # Extended Hamming code has minimum distance 4
        encoder = HammingCodeEncoder(mu=3, extended=True)
        assert encoder.minimum_distance() == 4

    def test_encoding(self):
        """Test encoding functionality."""
        # Test with mu=3 (standard Hamming code)
        encoder = HammingCodeEncoder(mu=3)

        # Test encoding a message
        x = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 4-bit message
        encoded = encoder(x)
        assert encoded.shape == torch.Size([7])  # 7-bit codeword

        # Verify that the first k bits are the original message (systematic code)
        assert torch.all(encoded[:4] == x)

        # Test with batch dimension
        x = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
        encoded = encoder(x)
        assert encoded.shape == torch.Size([2, 7])
        assert torch.all(encoded[:, :4] == x)

        # Test extended Hamming code
        encoder = HammingCodeEncoder(mu=3, extended=True)
        x = torch.tensor([1.0, 0.0, 1.0, 1.0])
        encoded = encoder(x)
        assert encoded.shape == torch.Size([8])  # 8-bit codeword

    def test_decoding(self):
        """Test decoding functionality."""
        # Test with mu=3 (standard Hamming code)
        encoder = HammingCodeEncoder(mu=3)

        # Test decoding a perfect codeword
        information_bits = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 4-bit message
        x = encoder(information_bits)  # Encode to get the codeword
        decoded, syndrome = encoder.inverse_encode(x)
        assert torch.equal(decoded, information_bits)  # Decoding should return the original message

        # Test decoding with a single bit error
        x_error = x.clone()
        x_error[1] = 1 - x_error[1]  # Introduce an error in bit 1
        decoded, syndrome = encoder.inverse_encode(x_error)
        assert not torch.equal(syndrome, torch.zeros_like(syndrome))  # Syndrome should be non-zero
        assert torch.equal(decoded, information_bits)  # Should correct the error

        # Test with batch dimension
        x_batch = torch.stack([x.clone(), x_error.clone()])  # Stack tensors properly
        decoded, syndrome = encoder.inverse_encode(x_batch)
        expected_batch = torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
        assert torch.equal(decoded, expected_batch)

        # Test extended Hamming code
        encoder = HammingCodeEncoder(mu=3, extended=True)
        extended_info_bits = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 4-bit message
        extended_x = encoder(extended_info_bits)  # Get proper extended Hamming codeword
        decoded, syndrome = encoder.inverse_encode(extended_x)
        assert torch.equal(decoded, extended_info_bits)  # 4-bit message

    def test_syndrome_calculation(self):
        """Test syndrome calculation."""
        encoder = HammingCodeEncoder(mu=3)

        # Test syndrome for valid codeword
        information_bits = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 4-bit message
        x = encoder(information_bits)  # Encode to get the codeword
        syndrome = encoder.calculate_syndrome(x)
        assert torch.equal(syndrome, torch.zeros_like(syndrome))  # Syndrome should be all zeros

        # Test syndrome for invalid codeword
        x_error = x.clone()
        x_error[1] = 1 - x_error[1]  # Introduce an error in bit 1
        # Calculate syndrome for the erroneous codeword
        syndrome = encoder.calculate_syndrome(x_error)
        assert not torch.equal(syndrome, torch.zeros_like(syndrome))  # Syndrome should be non-zero

    def test_representation(self):
        """Test string representation."""
        encoder = HammingCodeEncoder(mu=3, extended=True)
        repr_str = repr(encoder)

        # Check that the representation includes key parameters
        assert "HammingCodeEncoder" in repr_str
        assert "mu=3" in repr_str
        assert "extended=True" in repr_str
        assert "length=8" in repr_str
        assert "dimension=4" in repr_str
        assert "redundancy=4" in repr_str
