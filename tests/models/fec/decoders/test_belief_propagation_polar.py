"""Tests for the belief_propagation_polar module in kaira.models.fec.decoders package."""

import numpy as np
import pytest
import torch

from kaira.models.fec.decoders.belief_propagation_polar import BeliefPropagationPolarDecoder
from kaira.models.fec.encoders.polar_code import PolarCodeEncoder


class TestBeliefPropagationPolarDecoder:
    """Test suite for BeliefPropagationPolarDecoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.code_dimension = 2
        self.code_length = 4
        # Create info_indices: boolean array of length 4 with exactly 2 True values
        info_indices = np.array([True, True, False, False])
        self.encoder = PolarCodeEncoder(self.code_dimension, self.code_length, polar_i=False, load_rank=False, info_indices=info_indices)  # BP decoder doesn't support polar_i=True

    def test_initialization_basic(self):
        """Test basic initialization of BeliefPropagationPolarDecoder."""
        decoder = BeliefPropagationPolarDecoder(self.encoder)

        assert decoder.encoder is self.encoder
        assert decoder.code_dimension == self.code_dimension
        assert decoder.code_length == self.code_length
        assert np.array_equal(decoder.info_indices, self.encoder.info_indices)
        assert decoder.device == self.encoder.device
        assert decoder.dtype == self.encoder.dtype
        assert not decoder.polar_i
        assert decoder.frozen_zeros == self.encoder.frozen_zeros
        assert decoder.m == self.encoder.m
        assert decoder.iteration_num == 10  # default
        assert not decoder.early_stop  # default
        assert decoder.regime == "sum_product"  # default
        assert decoder.clip == 1000000.0  # default
        assert decoder.perm is None  # default

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=20, early_stop=True, regime="min_sum", clip=500.0, perm="cycle")

        assert decoder.iteration_num == 20
        assert decoder.early_stop
        assert decoder.regime == "min_sum"
        assert decoder.clip == 500.0
        assert decoder.perm == "cycle"

    def test_initialization_polar_i_error(self):
        """Test that polar_i=True raises error."""
        encoder_with_polar_i = PolarCodeEncoder(2, 4, polar_i=True)

        with pytest.raises(ValueError, match="Belief Propagation decoder does not support polar_i=True"):
            BeliefPropagationPolarDecoder(encoder_with_polar_i)

    def test_initialization_invalid_regime(self):
        """Test initialization with invalid regime."""
        with pytest.raises(ValueError, match="Invalid regime"):
            BeliefPropagationPolarDecoder(self.encoder, regime="invalid")

    def test_print_decoder_type(self, capsys):
        """Test print_decoder_type method."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=15)
        decoder.print_decoder_type()

        captured = capsys.readouterr()
        assert "BeliefPropagationPolarDecoder" in captured.out
        assert "Polar Code Length: 4" in captured.out
        assert "Polar Code Dimension: 2" in captured.out
        assert "Number of iterations: 15" in captured.out
        assert "Function used during decoding: sum_product" in captured.out

    def test_get_cyclic_permutations_none(self):
        """Test cyclic permutations with perm=None."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, perm=None)

        expected = np.arange(decoder.m).reshape(1, decoder.m)
        np.testing.assert_array_equal(decoder.permutations, expected)

    def test_get_cyclic_permutations_cycle(self):
        """Test cyclic permutations with perm='cycle'."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, perm="cycle")

        assert decoder.permutations.shape[1] == decoder.m
        assert decoder.permutations.shape[0] > 1  # Should have multiple permutations

    def test_checknode_sum_product(self):
        """Test checknode operation with sum_product regime."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, regime="sum_product")

        y1 = torch.tensor([1.0, 2.0, -1.0])
        y2 = torch.tensor([0.5, -1.0, 2.0])

        result = decoder.checknode(y1, y2)

        assert result.shape == y1.shape
        assert torch.isfinite(result).all()

    def test_checknode_min_sum(self):
        """Test checknode operation with min_sum regime."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, regime="min_sum")

        y1 = torch.tensor([1.0, 2.0, -1.0])
        y2 = torch.tensor([0.5, -1.0, 2.0])

        result = decoder.checknode(y1, y2)

        assert result.shape == y1.shape
        assert torch.isfinite(result).all()

    def test_initialize_graph(self):
        """Test graph initialization."""
        decoder = BeliefPropagationPolarDecoder(self.encoder)

        batch_size = 2
        llr = torch.randn(batch_size, self.code_length)

        R, L = decoder._initialize_graph(llr)

        assert R.shape == (batch_size, decoder.m + 1, self.code_length)
        assert L.shape == (batch_size, decoder.m + 1, self.code_length)
        assert R.device == torch.device(decoder.device)
        assert L.device == torch.device(decoder.device)

        # Check that frozen bits are initialized correctly
        if decoder.frozen_zeros:
            assert torch.all(R[:, 0, decoder.frozen_ind] == decoder.clip)
        else:
            assert torch.all(R[:, 0, decoder.frozen_ind] == -decoder.clip)

        # Check that L is initialized with received LLR
        torch.testing.assert_close(L[:, -1, :], llr)

    def test_update_right(self):
        """Test update_right method."""
        decoder = BeliefPropagationPolarDecoder(self.encoder)

        batch_size = 2
        llr = torch.randn(batch_size, self.code_length)
        R, L = decoder._initialize_graph(llr)

        perm = np.arange(decoder.m)
        R_updated = decoder.update_right(R, L, perm)

        assert R_updated.shape == R.shape
        assert torch.isfinite(R_updated).all()
        assert len(decoder.R_all) > 1  # Should have stored the update

    def test_update_left(self):
        """Test update_left method."""
        decoder = BeliefPropagationPolarDecoder(self.encoder)

        batch_size = 2
        llr = torch.randn(batch_size, self.code_length)
        R, L = decoder._initialize_graph(llr)

        perm = np.arange(decoder.m)
        L_updated = decoder.update_left(R, L, perm)

        assert L_updated.shape == L.shape
        assert torch.isfinite(L_updated).all()
        assert len(decoder.L_all) > 1  # Should have stored the update

    def test_decode_iterative_basic(self):
        """Test basic iterative decoding."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=2)

        batch_size = 2
        llr = torch.randn(batch_size, self.code_length)

        u_bits, x_bits = decoder.decode_iterative(llr)

        assert u_bits.shape == (batch_size, self.code_length)
        assert x_bits.shape == (batch_size, self.code_length)
        assert torch.all((u_bits == 0) | (u_bits == 1))
        assert torch.all((x_bits == 0) | (x_bits == 1))

    def test_decode_iterative_with_early_stop(self):
        """Test iterative decoding with early stopping."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=5, early_stop=True)

        batch_size = 1
        # Create a simple case that should converge quickly
        llr = torch.tensor([[10.0, -10.0, 5.0, -5.0]])

        u_bits, x_bits = decoder.decode_iterative(llr)

        assert u_bits.shape == (batch_size, self.code_length)
        assert x_bits.shape == (batch_size, self.code_length)

    def test_forward_decoding(self):
        """Test forward decoding method."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=2)

        # Test with proper LLR values
        batch_size = 3
        received = torch.randn(batch_size, self.code_length)

        decoded = decoder.forward(received)

        assert decoded.shape == (batch_size, self.code_dimension)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_forward_decoding_single_sample(self):
        """Test forward decoding with single sample."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=2)

        received = torch.randn(1, self.code_length)
        decoded = decoder.forward(received)

        assert decoded.shape == (1, self.code_dimension)

    def test_forward_decoding_device_consistency(self):
        """Test that decoding maintains device consistency."""
        device = torch.device("cpu")
        encoder = PolarCodeEncoder(2, 4, device=device, polar_i=False)
        decoder = BeliefPropagationPolarDecoder(encoder)

        received = torch.randn(2, 4, device=device)
        decoded = decoder.forward(received)

        assert decoded.device == device

    def test_clipping_functionality(self):
        """Test that clipping works correctly."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, clip=10.0)

        # Create extreme LLR values
        llr = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]])

        R, L = decoder._initialize_graph(llr)

        # Perform one update
        perm = np.arange(decoder.m)
        R_updated = decoder.update_right(R, L, perm)

        # Check that values are clipped
        assert torch.all(R_updated >= -decoder.clip)
        assert torch.all(R_updated <= decoder.clip)

    def test_different_permutations(self):
        """Test decoding with different permutation settings."""
        # No permutation
        decoder_none = BeliefPropagationPolarDecoder(self.encoder, perm=None, bp_iters=2)

        # Cyclic permutation
        decoder_cycle = BeliefPropagationPolarDecoder(self.encoder, perm="cycle", bp_iters=2)

        received = torch.randn(2, self.code_length)

        decoded_none = decoder_none.forward(received)
        decoded_cycle = decoder_cycle.forward(received)

        assert decoded_none.shape == decoded_cycle.shape
        assert decoded_none.shape == (2, self.code_dimension)

    def test_mask_dict_usage(self):
        """Test that mask_dict is properly used."""
        decoder = BeliefPropagationPolarDecoder(self.encoder)

        # Check that mask_dict has correct shape
        assert decoder.mask_dict.shape[0] == decoder.m

        # Test with custom mask_dict
        custom_encoder = PolarCodeEncoder(2, 4, polar_i=False)
        custom_encoder.mask_dict = None  # Force regeneration

        decoder_custom = BeliefPropagationPolarDecoder(custom_encoder)
        assert decoder_custom.mask_dict is not None
        assert decoder_custom.mask_dict.shape[0] == decoder_custom.m

    def test_regime_differences(self):
        """Test differences between sum_product and min_sum regimes."""
        decoder_sp = BeliefPropagationPolarDecoder(self.encoder, regime="sum_product", bp_iters=1)
        decoder_ms = BeliefPropagationPolarDecoder(self.encoder, regime="min_sum", bp_iters=1)

        received = torch.tensor([[1.0, -1.0, 0.5, -0.5]])

        decoded_sp = decoder_sp.forward(received)
        decoded_ms = decoder_ms.forward(received)

        assert decoded_sp.shape == decoded_ms.shape
        # Results might be different due to different check node operations

    def test_frozen_zeros_effect(self):
        """Test effect of frozen_zeros parameter."""
        encoder_zeros = PolarCodeEncoder(2, 4, frozen_zeros=True, polar_i=False)
        encoder_ones = PolarCodeEncoder(2, 4, frozen_zeros=False, polar_i=False)

        decoder_zeros = BeliefPropagationPolarDecoder(encoder_zeros)
        decoder_ones = BeliefPropagationPolarDecoder(encoder_ones)

        assert decoder_zeros.frozen_zeros
        assert not decoder_ones.frozen_zeros

        # Test initialization differences
        llr = torch.randn(1, 4)
        R_zeros, _ = decoder_zeros._initialize_graph(llr)
        R_ones, _ = decoder_ones._initialize_graph(llr)

        # Frozen bits should be initialized differently
        frozen_idx = decoder_zeros.frozen_ind
        if np.any(frozen_idx):
            assert not torch.allclose(R_zeros[:, 0, frozen_idx], R_ones[:, 0, frozen_idx])

    def test_early_stop_warning(self, capsys):
        """Test warning when using cyclic permutation without early stopping."""
        BeliefPropagationPolarDecoder(self.encoder, perm="cycle", early_stop=False)

        captured = capsys.readouterr()
        assert "Warning: Cyclic permutation is used, but early stopping is disabled" in captured.out

    def test_larger_code(self):
        """Test with larger polar code."""
        # Create info_indices: boolean array of length 8 with exactly 4 True values
        large_info_indices = np.array([True, True, True, True, False, False, False, False])
        large_encoder = PolarCodeEncoder(4, 8, polar_i=False, load_rank=False, info_indices=large_info_indices)
        decoder = BeliefPropagationPolarDecoder(large_encoder, bp_iters=2)

        received = torch.randn(2, 8)
        decoded = decoder.forward(received)

        assert decoded.shape == (2, 4)
        assert torch.all((decoded == 0) | (decoded == 1))

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=2)

        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            received = torch.randn(batch_size, self.code_length)
            decoded = decoder.forward(received)
            assert decoded.shape == (batch_size, self.code_dimension)

    def test_deterministic_decoding(self):
        """Test that decoding is deterministic for same input."""
        decoder = BeliefPropagationPolarDecoder(self.encoder, bp_iters=3)

        received = torch.tensor([[2.0, -1.5, 1.0, -0.5]])

        decoded1 = decoder.forward(received)
        decoded2 = decoder.forward(received)

        assert torch.allclose(decoded1, decoded2)
