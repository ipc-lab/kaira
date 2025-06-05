"""Tests for the polar_code module in kaira.models.fec.encoders package."""

import numpy as np
import pytest
import torch

from kaira.models.fec.encoders.polar_code import (
    PolarCodeEncoder,
    _index_matrix,
    calculate_gm,
)


class TestPolarCodeHelperFunctions:
    """Test suite for helper functions in polar code module."""

    def test_index_matrix(self):
        """Test generating index matrix for polar code construction."""
        # Test with N=4
        result = _index_matrix(4)
        expected = np.array([[1, 1], [2, 3]])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, expected)

        # Test with N=8
        result = _index_matrix(8)
        assert result.shape == (4, 3)
        # Check that all values are in valid range
        assert np.all(result >= 1)
        assert np.all(result <= 8)

    def test_index_matrix_invalid_input(self):
        """Test index matrix with invalid input."""
        # Test with non-power-of-2
        with pytest.raises(AssertionError):
            _index_matrix(5)

        with pytest.raises(AssertionError):
            _index_matrix(7)

    def test_calculate_gm(self):
        """Test calculating generator matrix for polar code."""
        device = torch.device("cpu")

        # Test with code_length=4
        gm = calculate_gm(4, device)
        expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
        assert gm.shape == (4, 4)
        assert torch.allclose(gm, expected)
        assert gm.device == device

        # Test with code_length=8
        gm = calculate_gm(8, device)
        assert gm.shape == (8, 8)
        assert gm.device == device


class TestPolarCodeEncoder:
    """Test suite for PolarCodeEncoder class."""

    def test_initialization_basic(self):
        """Test basic initialization of PolarCodeEncoder."""
        code_dimension = 2
        code_length = 4
        encoder = PolarCodeEncoder(code_dimension, code_length)

        assert encoder.code_dimension == code_dimension
        assert encoder.code_length == code_length
        assert encoder.m == 2  # log2(4)
        assert encoder.device == "cpu"  # device is stored as string
        assert encoder.dtype == torch.float32
        assert not encoder.polar_i
        assert not encoder.frozen_zeros  # default is False
        assert encoder.load_rank

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        # Provide valid info_indices when load_rank=False
        info_indices = np.array([False, False, False, False, False, True, True, True])
        encoder = PolarCodeEncoder(code_dimension=3, code_length=8, device="cpu", dtype=torch.float64, polar_i=True, frozen_zeros=False, load_rank=False, info_indices=info_indices)

        assert encoder.code_dimension == 3
        assert encoder.code_length == 8
        assert encoder.m == 3  # log2(8)
        assert encoder.dtype == torch.float64
        assert encoder.polar_i
        assert not encoder.frozen_zeros
        assert not encoder.load_rank
        np.testing.assert_array_equal(encoder.info_indices, info_indices)

    def test_initialization_invalid_code_length(self):
        """Test initialization with invalid code length."""
        with pytest.raises(AssertionError):
            PolarCodeEncoder(2, 5)  # Not a power of 2

        with pytest.raises(AssertionError):
            PolarCodeEncoder(2, 7)  # Not a power of 2

    def test_initialization_missing_info_indices(self):
        """Test initialization fails when load_rank=False but info_indices not provided."""
        with pytest.raises(ValueError, match="When load_rank=False, info_indices must be provided"):
            PolarCodeEncoder(2, 4, load_rank=False)

    def test_initialization_invalid_info_indices(self):
        """Test initialization fails with invalid info_indices."""
        # Wrong length
        with pytest.raises(ValueError, match="info_indices must have length 4"):
            PolarCodeEncoder(2, 4, load_rank=False, info_indices=np.array([True, True]))

        # Wrong number of True values
        with pytest.raises(ValueError, match="info_indices must have exactly 2 True values"):
            PolarCodeEncoder(2, 4, load_rank=False, info_indices=np.array([True, True, True, False]))

    def test_info_indices_generation(self):
        """Test generation of information bit indices."""
        encoder = PolarCodeEncoder(2, 4, load_rank=False, info_indices=np.array([False, False, True, True]))

        # Check that info_indices is a boolean array
        assert isinstance(encoder.info_indices, np.ndarray)
        assert encoder.info_indices.dtype == bool
        assert len(encoder.info_indices) == 4

        # Should have exactly 2 True values (code_dimension)
        assert np.sum(encoder.info_indices) == 2

    def test_polar_transform_basic(self):
        """Test basic polar transform functionality."""
        # Provide valid info_indices when load_rank=False is implied (default is True now)
        encoder = PolarCodeEncoder(2, 4, polar_i=False)

        # Test with single input
        u = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        result = encoder.polar_transform(u)

        assert result.shape == (1, 4)
        assert result.dtype == torch.float32

    def test_polar_transform_with_polar_i(self):
        """Test polar transform with polar_i enabled."""
        encoder = PolarCodeEncoder(2, 4, polar_i=True)

        u = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        result = encoder.polar_transform(u)

        assert result.shape == (1, 4)

    def test_polar_transform_return_array(self):
        """Test polar transform with return_arr=True."""
        encoder = PolarCodeEncoder(2, 4)

        u = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        result = encoder.polar_transform(u, return_arr=True)

        assert isinstance(result, list)
        assert len(result) == encoder.m + 1  # m stages + initial

    def test_forward_encoding(self):
        """Test forward encoding process."""
        encoder = PolarCodeEncoder(2, 4)

        # Test with binary input
        input_msg = torch.tensor([[1, 0]], dtype=torch.float32)
        encoded = encoder.forward(input_msg)

        assert encoded.shape == (1, 4)
        assert encoded.dtype == torch.float32

    def test_forward_encoding_batch(self):
        """Test forward encoding with batch input."""
        encoder = PolarCodeEncoder(3, 8)

        # Test with batch of inputs
        batch_size = 5
        input_msgs = torch.randint(0, 2, (batch_size, 3), dtype=torch.float32)
        encoded = encoder.forward(input_msgs)

        assert encoded.shape == (batch_size, 8)

    def test_forward_encoding_different_shapes(self):
        """Test forward encoding with different input shapes."""
        encoder = PolarCodeEncoder(4, 8)

        # Test with 2D input reshaped to proper dimensions
        input_msgs = torch.randint(0, 2, (6, 4), dtype=torch.float32)  # 6 samples of 4 bits each
        encoded = encoder.forward(input_msgs)

        assert encoded.shape == (6, 8)

    def test_get_generator_matrix(self):
        """Test getting generator matrix."""
        encoder = PolarCodeEncoder(2, 4)

        G = encoder.get_generator_matrix()

        assert G.shape == (4, 4)  # Full generator matrix shape is (N, N)
        assert G.dtype == torch.float32

    def test_get_syndrome_matrix(self):
        """Test getting syndrome matrix placeholder."""
        encoder = PolarCodeEncoder(2, 4)

        # Polar codes don't use syndrome matrices, so this should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            encoder.calculate_syndrome(torch.tensor([[1, 0, 1, 0]], dtype=torch.float32))

    def test_inverse_encode_placeholder(self):
        """Test inverse encode placeholder method."""
        encoder = PolarCodeEncoder(2, 4)

        x = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            encoder.inverse_encode(x)

    def test_calculate_syndrome_placeholder(self):
        """Test calculate syndrome placeholder method."""
        encoder = PolarCodeEncoder(2, 4)

        x = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            encoder.calculate_syndrome(x)

    def test_device_consistency(self):
        """Test that all tensors are on the correct device."""
        device = torch.device("cpu")
        encoder = PolarCodeEncoder(2, 4, device=device)

        # Check that generator matrix is on correct device
        G = encoder.get_generator_matrix()
        assert G.device == device

        # Check encoding output device
        input_msg = torch.tensor([[1, 0]], dtype=torch.float32, device=device)
        encoded = encoder.forward(input_msg)
        assert encoded.device == device

    def test_frozen_zeros_vs_ones(self):
        """Test difference between frozen_zeros=True and False."""
        encoder_zeros = PolarCodeEncoder(2, 4, frozen_zeros=True)
        encoder_ones = PolarCodeEncoder(2, 4, frozen_zeros=False)

        # Both should have same info_indices
        np.testing.assert_array_equal(encoder_zeros.info_indices, encoder_ones.info_indices)

        # But may behave differently in encoding
        input_msg = torch.tensor([[1, 0]], dtype=torch.float32)
        encoded_zeros = encoder_zeros.forward(input_msg)
        encoded_ones = encoder_ones.forward(input_msg)

        assert encoded_zeros.shape == encoded_ones.shape

    def test_mask_dict_generation(self):
        """Test mask dictionary generation."""
        encoder = PolarCodeEncoder(2, 4)

        if encoder.mask_dict is not None:
            assert isinstance(encoder.mask_dict, np.ndarray)
            assert encoder.mask_dict.shape[0] == encoder.m

    def test_rank_loading(self):
        """Test rank-based index loading."""
        # Test with load_rank=True (default)
        encoder_rank = PolarCodeEncoder(2, 4, load_rank=True)

        # Test with load_rank=False - need to provide info_indices manually
        info_indices = np.array([False, False, True, True])  # Example indices
        encoder_no_rank = PolarCodeEncoder(2, 4, load_rank=False, info_indices=info_indices)

        # Both should have valid info_indices
        assert len(encoder_rank.info_indices) == 4
        assert len(encoder_no_rank.info_indices) == 4
        assert np.sum(encoder_rank.info_indices) == 2
        assert np.sum(encoder_no_rank.info_indices) == 2

    def test_encoding_deterministic(self):
        """Test that encoding is deterministic."""
        encoder = PolarCodeEncoder(2, 4)

        input_msg = torch.tensor([[1, 0]], dtype=torch.float32)

        encoded1 = encoder.forward(input_msg)
        encoded2 = encoder.forward(input_msg)

        assert torch.allclose(encoded1, encoded2)

    def test_different_code_rates(self):
        """Test encoders with different code rates."""
        # High rate code
        encoder_high = PolarCodeEncoder(6, 8)
        assert encoder_high.code_dimension == 6
        assert encoder_high.code_length == 8

        # Low rate code
        encoder_low = PolarCodeEncoder(2, 8)
        assert encoder_low.code_dimension == 2
        assert encoder_low.code_length == 8

        # Test encoding
        input_high = torch.randint(0, 2, (1, 6), dtype=torch.float32)
        input_low = torch.randint(0, 2, (1, 2), dtype=torch.float32)

        encoded_high = encoder_high.forward(input_high)
        encoded_low = encoder_low.forward(input_low)

        assert encoded_high.shape == (1, 8)
        assert encoded_low.shape == (1, 8)
