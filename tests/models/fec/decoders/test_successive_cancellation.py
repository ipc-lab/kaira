"""Tests for the successive_cancellation module in kaira.models.fec.decoders package."""

import pytest
import torch

from kaira.models.fec.decoders.successive_cancellation import SuccessiveCancellationDecoder
from kaira.models.fec.encoders.polar_code import PolarCodeEncoder


class TestSuccessiveCancellationDecoder:
    """Test suite for SuccessiveCancellationDecoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.code_dimension = 2
        self.code_length = 4
        self.encoder = PolarCodeEncoder(self.code_dimension, self.code_length, polar_i=False, load_rank=True)  # Changed to True so info_indices is properly set

    def test_initialization_basic(self):
        """Test basic initialization of SuccessiveCancellationDecoder."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        assert decoder.encoder is self.encoder
        assert decoder.code_dimension == self.code_dimension
        assert decoder.code_length == self.code_length
        assert torch.equal(decoder.info_indices, self.encoder.info_indices)
        assert decoder.device == self.encoder.device
        assert decoder.dtype == self.encoder.dtype
        assert decoder.polar_i == self.encoder.polar_i
        assert decoder.frozen_zeros == self.encoder.frozen_zeros
        assert decoder.m == self.encoder.m
        assert decoder.regime == "sum_product"  # default
        assert decoder.clip == 1000.0  # default

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        decoder = SuccessiveCancellationDecoder(self.encoder, regime="min_sum", clip=500.0)

        assert decoder.regime == "min_sum"
        assert decoder.clip == 500.0

    def test_initialization_invalid_regime(self):
        """Test initialization with invalid regime."""
        with pytest.raises(ValueError, match="Invalid regime"):
            SuccessiveCancellationDecoder(self.encoder, regime="invalid")

    def test_f2_method(self):
        """Test f2 method for combining binary vectors."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        x1 = torch.tensor([[0, 1], [1, 0]])
        x2 = torch.tensor([[1, 0], [0, 1]])

        result = decoder.f2((x1, x2))

        assert result.shape == (2, 4)  # Should concatenate the vectors
        # Check XOR operation: torch.remainder(x1 + x2, 2)
        expected_first_half = torch.remainder(x1 + x2, 2)
        assert torch.equal(result[:, :2], expected_first_half)
        assert torch.equal(result[:, 2:], x2)

    def test_checknode_sum_product(self):
        """Test checknode operation with sum_product regime."""
        decoder = SuccessiveCancellationDecoder(self.encoder, regime="sum_product")

        y1 = torch.tensor([1.0, 2.0, -1.0])
        y2 = torch.tensor([0.5, -1.0, 2.0])

        result = decoder.checknode((y1, y2))

        assert result.shape == y1.shape
        assert torch.isfinite(result).all()

    def test_checknode_min_sum(self):
        """Test checknode operation with min_sum regime."""
        decoder = SuccessiveCancellationDecoder(self.encoder, regime="min_sum")

        y1 = torch.tensor([1.0, 2.0, -1.0])
        y2 = torch.tensor([0.5, -1.0, 2.0])

        result = decoder.checknode((y1, y2))

        assert result.shape == y1.shape
        assert torch.isfinite(result).all()

    def test_bitnode_operation(self):
        """Test bitnode operation."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        y1 = torch.tensor([[1.0, 2.0]])
        y2 = torch.tensor([[0.5, -1.0]])
        x = torch.tensor([[0.0, 1.0]])

        result = decoder.bitnode((y1, y2, x))

        assert result.shape == y1.shape
        assert torch.isfinite(result).all()

    def test_decode_recursive_basic(self):
        """Test basic decode_recursive functionality."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        llr = torch.randn(1, self.code_length)

        # Use the decoder's own info_indices, not the encoder's potentially None value
        u, x, y_final = decoder.decode_recursive(llr, decoder.info_indices)

        assert u.shape == (1, self.code_length)
        assert x.shape == (1, self.code_length)
        assert y_final.shape == (1, self.code_length)
        assert torch.all((u == 0) | (u == 1))
        assert torch.all((x == 0) | (x == 1))

    def test_decode_recursive_with_polar_i(self):
        """Test decode_recursive with polar_i enabled."""
        encoder_polar_i = PolarCodeEncoder(2, 4, polar_i=True, load_rank=True)
        decoder = SuccessiveCancellationDecoder(encoder_polar_i)

        llr = torch.randn(1, self.code_length)

        u, x, y_final = decoder.decode_recursive(llr, decoder.info_indices)

        assert u.shape == (1, self.code_length)
        assert x.shape == (1, self.code_length)
        assert y_final.shape == (1, self.code_length)
        assert torch.all((u == 0) | (u == 1))
        assert torch.all((x == 0) | (x == 1))

    def test_forward_with_return_for_loss(self):
        """Test forward method with return_for_loss=True."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        received = torch.randn(1, self.code_length)

        # Test with return_for_loss=True to get LLR values
        llr_output = decoder.forward(received, return_for_loss=True)

        assert llr_output.shape == (1, self.code_length)
        assert torch.isfinite(llr_output).all()

    def test_forward_decoding(self):
        """Test forward decoding method."""
        decoder = SuccessiveCancellationDecoder(self.encoder)
        batch_size = 3
        received = torch.randn(batch_size, self.code_length)

        decoded = decoder.forward(received)

        assert decoded.shape == (batch_size, self.code_dimension)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_forward_decoding_single_sample(self):
        """Test forward decoding with single sample."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        received = torch.randn(1, self.code_length)
        decoded = decoder.forward(received)

        assert decoded.shape == (1, self.code_dimension)

    def test_forward_decoding_device_consistency(self):
        """Test that decoding maintains device consistency."""
        device = torch.device("cpu")
        encoder = PolarCodeEncoder(2, 4, device=device)
        decoder = SuccessiveCancellationDecoder(encoder)

        received = torch.randn(2, 4, device=device)
        decoded = decoder.forward(received)

        assert decoded.device == device

    def test_clipping_functionality(self):
        """Test that clipping works correctly."""
        decoder = SuccessiveCancellationDecoder(self.encoder, clip=10.0)

        # Create extreme LLR values
        llr = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]])

        # Decode and check that no extreme values cause issues
        decoded = decoder.forward(llr)

        assert decoded.shape == (1, self.code_dimension)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_frozen_bits_handling(self):
        """Test handling of frozen bits."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        # Test that frozen bits are handled correctly
        llr = torch.randn(2, self.code_length)

        u, x, y_final = decoder.decode_recursive(llr, decoder.info_indices)

        # Check that we get the right shapes and types
        assert u.shape == (2, self.code_length)
        assert x.shape == (2, self.code_length)
        assert y_final.shape == (2, self.code_length)
        assert torch.all((u == 0) | (u == 1))
        assert torch.all((x == 0) | (x == 1))

    def test_regime_differences(self):
        """Test differences between sum_product and min_sum regimes."""
        decoder_sp = SuccessiveCancellationDecoder(self.encoder, regime="sum_product")
        decoder_ms = SuccessiveCancellationDecoder(self.encoder, regime="min_sum")

        received = torch.tensor([[1.0, -1.0, 0.5, -0.5]])

        decoded_sp = decoder_sp.forward(received)
        decoded_ms = decoder_ms.forward(received)

        assert decoded_sp.shape == decoded_ms.shape
        # Results might be different due to different check node operations

    def test_frozen_zeros_effect(self):
        """Test effect of frozen_zeros parameter."""
        encoder_zeros = PolarCodeEncoder(2, 4, frozen_zeros=True)
        encoder_ones = PolarCodeEncoder(2, 4, frozen_zeros=False)

        decoder_zeros = SuccessiveCancellationDecoder(encoder_zeros)
        decoder_ones = SuccessiveCancellationDecoder(encoder_ones)

        assert decoder_zeros.frozen_zeros
        assert not decoder_ones.frozen_zeros

        # Test that both decoders work
        received = torch.randn(1, 4)

        decoded_zeros = decoder_zeros.forward(received)
        decoded_ones = decoder_ones.forward(received)

        assert decoded_zeros.shape == (1, 2)
        assert decoded_ones.shape == (1, 2)

    def test_larger_code(self):
        """Test with larger polar code."""
        large_encoder = PolarCodeEncoder(4, 8, load_rank=True)
        decoder = SuccessiveCancellationDecoder(large_encoder)

        received = torch.randn(2, 8)
        decoded = decoder.forward(received)

        assert decoded.shape == (2, 4)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            received = torch.randn(batch_size, self.code_length)
            decoded = decoder.forward(received)
            assert decoded.shape == (batch_size, self.code_dimension)

    def test_deterministic_decoding(self):
        """Test that decoding is deterministic for same input."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        received = torch.tensor([[2.0, -1.5, 1.0, -0.5]])

        decoded1 = decoder.forward(received)
        decoded2 = decoder.forward(received)

        assert torch.allclose(decoded1, decoded2)

    def test_edge_cases(self):
        """Test edge cases."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        # Test with zero LLR
        received_zeros = torch.zeros(1, self.code_length)
        decoded_zeros = decoder.forward(received_zeros)
        assert decoded_zeros.shape == (1, self.code_dimension)

        # Test with very large positive LLR (strong 0 bits)
        received_large_pos = torch.full((1, self.code_length), 100.0)
        decoded_large_pos = decoder.forward(received_large_pos)
        assert decoded_large_pos.shape == (1, self.code_dimension)

        # Test with very large negative LLR (strong 1 bits)
        received_large_neg = torch.full((1, self.code_length), -100.0)
        decoded_large_neg = decoder.forward(received_large_neg)
        assert decoded_large_neg.shape == (1, self.code_dimension)

    def test_info_indices_extraction(self):
        """Test that information bits are correctly extracted."""
        decoder = SuccessiveCancellationDecoder(self.encoder)

        received = torch.randn(2, self.code_length)
        decoded = decoder.forward(received)

        # Check that we get exactly the information bits
        assert decoded.shape == (2, self.code_dimension)

        # Verify that the number of info indices matches code dimension
        assert torch.sum(decoder.info_indices) == self.code_dimension

    def test_different_dtypes(self):
        """Test with different data types."""
        encoder_float64 = PolarCodeEncoder(2, 4, dtype=torch.float64)
        decoder = SuccessiveCancellationDecoder(encoder_float64)

        received = torch.randn(1, 4, dtype=torch.float64)
        decoded = decoder.forward(received)

        assert decoded.dtype == torch.float64
        assert decoded.shape == (1, 2)

    def test_zero_dimension_code(self):
        """Test with very low rate code."""
        low_rate_encoder = PolarCodeEncoder(1, 4, load_rank=True)  # Very low rate
        decoder = SuccessiveCancellationDecoder(low_rate_encoder)

        received = torch.randn(1, 4)
        decoded = decoder.forward(received)

        assert decoded.shape == (1, 1)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_high_rate_code(self):
        """Test with high rate code."""
        high_rate_encoder = PolarCodeEncoder(6, 8, load_rank=True)  # High rate
        decoder = SuccessiveCancellationDecoder(high_rate_encoder)

        received = torch.randn(1, 8)
        decoded = decoder.forward(received)

        assert decoded.shape == (1, 6)
        assert torch.all((decoded == 0) | (decoded == 1))
