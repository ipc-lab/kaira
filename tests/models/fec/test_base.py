"""Tests for the base encoder and decoder modules in kaira.models.fec package."""

import pytest
import torch

from kaira.models.fec.decoders.base import BaseBlockDecoder
from kaira.models.fec.encoders.base import BaseBlockCodeEncoder


class MockBlockCodeEncoder(BaseBlockCodeEncoder):
    """Mock implementation of BaseBlockCodeEncoder for testing."""

    def __init__(self, code_length, code_dimension, **kwargs):
        """Initialize the mock encoder."""
        super().__init__(code_length, code_dimension, **kwargs)

    def forward(self, x, *args, **kwargs):
        """Mock implementation of forward method."""
        # Check if the last dimension is a multiple of k
        last_dim_size = x.shape[-1]
        if last_dim_size % self.code_dimension != 0:
            raise ValueError(f"Last dimension size {last_dim_size} must be a multiple of the code dimension {self.code_dimension}")

        # Simple mock encoding: repeat each bit to achieve the desired code length
        k = self.code_dimension
        n = self.code_length
        repeat_factor = n // k

        # Reshape to expose the dimension to repeat
        orig_shape = x.shape
        x_reshaped = x.reshape(-1, k)

        # Repeat each bit to create a simple encoding
        encoded = torch.repeat_interleave(x_reshaped, repeat_factor, dim=1)

        # Reshape back to match original dimensions, but with expanded last dim
        new_shape = list(orig_shape)
        new_shape[-1] = new_shape[-1] * repeat_factor
        return encoded.reshape(*new_shape)


class MockBlockDecoder(BaseBlockDecoder):
    """Mock implementation of BaseBlockDecoder for testing."""

    def forward(self, received, *args, **kwargs):
        """Mock implementation of forward method."""
        # Simple mock decoding: take every nth bit (similar to the old inverse_encode)
        k = self.encoder.code_dimension
        n = self.encoder.code_length
        ratio = n // k

        # Reshape to expose the dimension to downsample
        orig_shape = received.shape
        received_reshaped = received.reshape(-1, n)

        # Take every ratio-th element
        decoded = received_reshaped[:, ::ratio]

        # Reshape back to match original dimensions, but with reduced last dim
        new_shape = list(orig_shape)
        new_shape[-1] = new_shape[-1] // ratio
        return decoded.reshape(*new_shape)


class TestBaseBlockCodeEncoder:
    """Test suite for BaseBlockCodeEncoder class."""

    def test_initialization(self):
        """Test initialization of BaseBlockCodeEncoder with valid and invalid parameters."""
        # Test valid initialization
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)
        assert encoder.code_length == 6
        assert encoder.code_dimension == 3
        assert encoder.redundancy == 3
        assert encoder.parity_bits == 3
        assert encoder.code_rate == 0.5

        # Test invalid code_length
        with pytest.raises(ValueError, match="Code length must be positive"):
            MockBlockCodeEncoder(code_length=0, code_dimension=3)

        # Test invalid code_dimension
        with pytest.raises(ValueError, match="Code dimension must be positive"):
            MockBlockCodeEncoder(code_length=6, code_dimension=0)

        # Test code_dimension > code_length
        with pytest.raises(ValueError, match="Code dimension .* must not exceed code length"):
            MockBlockCodeEncoder(code_length=3, code_dimension=6)

    def test_properties(self):
        """Test properties of BaseBlockCodeEncoder."""
        encoder = MockBlockCodeEncoder(code_length=8, code_dimension=4)

        # Test basic properties
        assert encoder.code_length == 8
        assert encoder.code_dimension == 4
        assert encoder.redundancy == 4
        assert encoder.parity_bits == 4
        assert encoder.code_rate == 0.5

    def test_forward(self):
        """Test forward method of MockBlockCodeEncoder."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)

        # Test with a simple input: a single codeword
        x = torch.tensor([1, 0, 1])
        encoded = encoder.forward(x)
        assert encoded.shape == torch.Size([6])
        assert torch.all(encoded == torch.tensor([1, 1, 0, 0, 1, 1]))

        # Test with batch dimension
        x = torch.tensor([[1, 0, 1], [0, 1, 0]])
        encoded = encoder.forward(x)
        assert encoded.shape == torch.Size([2, 6])
        assert torch.all(encoded == torch.tensor([[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]]))

        # Test with multiple codewords per input (last dim multiple of code_dimension)
        x = torch.tensor([1, 0, 1, 0, 1, 1])  # 2 messages of 3 bits each
        encoded = encoder.forward(x)
        assert encoded.shape == torch.Size([12])  # 2 codewords of 6 bits each

    def test_extract_message(self):
        """Test extract_message method of MockBlockCodeEncoder."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)

        # Test with a single codeword
        x = torch.tensor([1, 1, 0, 0, 1, 1])
        # Since this mock doesn't implement inverse_encode, extract_message should fail
        with pytest.raises(AttributeError):
            encoder.extract_message(x)

    def test_forward_error_handling(self):
        """Test error handling in forward method."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)

        # Test with invalid input size (not multiple of code_dimension)
        x = torch.tensor([1, 0])  # Size 2, but need multiple of 3
        with pytest.raises(ValueError, match="must be a multiple of the code dimension"):
            encoder.forward(x)


class TestBaseBlockDecoder:
    """Test suite for BaseBlockDecoder class."""

    def test_initialization(self):
        """Test initialization of BaseBlockDecoder."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)
        decoder = MockBlockDecoder(encoder=encoder)

        # Test encoder assignment
        assert decoder.encoder is encoder

    def test_properties(self):
        """Test properties of BaseBlockDecoder."""
        encoder = MockBlockCodeEncoder(code_length=8, code_dimension=4)
        decoder = MockBlockDecoder(encoder=encoder)

        # Test properties derived from encoder
        assert decoder.code_length == 8
        assert decoder.code_dimension == 4
        assert decoder.redundancy == 4
        assert decoder.code_rate == 0.5

    def test_forward(self):
        """Test forward method of MockBlockDecoder."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)
        decoder = MockBlockDecoder(encoder=encoder)

        # Test with a single codeword
        received = torch.tensor([1, 1, 0, 0, 1, 1])
        decoded = decoder.forward(received)
        assert decoded.shape == torch.Size([3])
        assert torch.all(decoded == torch.tensor([1, 0, 1]))

        # Test with batch dimension
        received = torch.tensor([[1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0]])
        decoded = decoder.forward(received)
        assert decoded.shape == torch.Size([2, 3])
        assert torch.all(decoded == torch.tensor([[1, 0, 1], [0, 1, 0]]))

        # Test with noisy codeword
        noisy = torch.tensor([1, 0, 0, 0, 1, 1])  # First bit flipped
        decoded = decoder.forward(noisy)
        # Note: Our simple mock decoder can't correct errors, so this shows the effect of errors
        assert torch.all(decoded == torch.tensor([1, 0, 1]))

    def test_encoder_decoder_integration(self):
        """Test integration between encoder and decoder."""
        encoder = MockBlockCodeEncoder(code_length=6, code_dimension=3)
        decoder = MockBlockDecoder(encoder=encoder)

        # Test that encoder.forward + decoder.forward recovers the original message
        original = torch.tensor([1, 0, 1])
        encoded = encoder.forward(original)
        decoded = decoder.forward(encoded)
        assert torch.all(decoded == original)

        # Test with batch dimension
        original = torch.tensor([[1, 0, 1], [0, 1, 0]])
        encoded = encoder.forward(original)
        decoded = decoder.forward(encoded)
        assert torch.all(decoded == original)
