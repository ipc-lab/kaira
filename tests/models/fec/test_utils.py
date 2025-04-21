"""Tests for the utils module in kaira.models.fec package."""

import pytest
import torch

from kaira.models.fec.utils import (
    apply_blockwise,
    from_binary_tensor,
    hamming_distance,
    hamming_weight,
    to_binary_tensor,
)


class TestUtils:
    """Test suite for utility functions in the FEC package."""

    def test_hamming_distance(self):
        """Test hamming_distance function."""
        # Basic test case
        x = torch.tensor([1, 0, 1, 0])
        y = torch.tensor([1, 1, 0, 0])
        assert hamming_distance(x, y) == 2

        # Test with all same values
        x = torch.tensor([1, 1, 1, 1])
        y = torch.tensor([1, 1, 1, 1])
        assert hamming_distance(x, y) == 0

        # Test with all different values
        x = torch.tensor([1, 1, 1, 1])
        y = torch.tensor([0, 0, 0, 0])
        assert hamming_distance(x, y) == 4

        # Test with batch dimension
        x = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
        y = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]])
        result = hamming_distance(x, y)
        assert torch.all(result == torch.tensor([2, 2]))

        # Test with float tensors (should still work as it uses !=)
        x = torch.tensor([1.0, 0.0, 1.0, 0.0])
        y = torch.tensor([1.0, 1.0, 0.0, 0.0])
        assert hamming_distance(x, y) == 2

        # Test with multiple batch dimensions
        x = torch.tensor([[[1, 0], [1, 0]], [[0, 1], [1, 1]]])
        y = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 0]]])
        result = hamming_distance(x, y)
        assert result.shape == (2, 2)
        assert torch.all(result == torch.tensor([[1, 1], [1, 1]]))

    def test_hamming_weight(self):
        """Test hamming_weight function."""
        # Basic test case
        x = torch.tensor([1, 0, 1, 1, 0])
        assert hamming_weight(x) == 3

        # Test with all zeros
        x = torch.tensor([0, 0, 0, 0])
        assert hamming_weight(x) == 0

        # Test with all ones
        x = torch.tensor([1, 1, 1, 1])
        assert hamming_weight(x) == 4

        # Test with batch dimension
        x = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
        result = hamming_weight(x)
        assert torch.all(result == torch.tensor([2, 2]))

        # Test with float tensors (should sum them directly)
        x = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert hamming_weight(x) == 2.0

        # Test with multiple batch dimensions
        x = torch.tensor([[[1, 0], [1, 1]], [[0, 1], [1, 0]]])
        result = hamming_weight(x)
        assert result.shape == (2, 2)
        assert torch.all(result == torch.tensor([[1, 2], [1, 1]]))

    def test_to_binary_tensor(self):
        """Test to_binary_tensor function."""
        # Basic test case: decimal 10 = binary 1010
        result = to_binary_tensor(10, 4)
        expected = torch.tensor([1, 0, 1, 0])
        assert torch.all(result == expected)

        # Test with leading zeros
        result = to_binary_tensor(10, 6)  # decimal 10 = binary 001010
        expected = torch.tensor([0, 0, 1, 0, 1, 0])
        assert torch.all(result == expected)

        # Test with fewer bits than needed
        result = to_binary_tensor(10, 3)  # Only keep the 3 least significant bits
        expected = torch.tensor([0, 1, 0])  # 010 in binary
        assert torch.all(result == expected)

        # Test with 0
        result = to_binary_tensor(0, 4)
        expected = torch.tensor([0, 0, 0, 0])
        assert torch.all(result == expected)

        # Test with larger number
        result = to_binary_tensor(255, 8)  # decimal 255 = binary 11111111
        expected = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
        assert torch.all(result == expected)

        # Test with specified device (default is None/CPU)
        device = torch.device("cpu")
        result = to_binary_tensor(10, 4, device=device)
        assert result.device == device

        # Test with custom dtype
        result = to_binary_tensor(10, 4, device=None, dtype=torch.float32)
        assert result.dtype == torch.float32

        # Test with negative number (should handle the absolute value)
        result = to_binary_tensor(-10, 4)
        expected = torch.tensor([1, 0, 1, 0])  # Same as positive 10
        assert torch.all(result == expected)

    def test_from_binary_tensor(self):
        """Test from_binary_tensor function."""
        # Basic test case: binary 1010 = decimal 10
        x = torch.tensor([1, 0, 1, 0])
        assert from_binary_tensor(x) == 10

        # Test with leading zeros
        x = torch.tensor([0, 0, 1, 0, 1, 0])  # binary 001010 = decimal 10
        assert from_binary_tensor(x) == 10

        # Test with all zeros
        x = torch.tensor([0, 0, 0, 0])
        assert from_binary_tensor(x) == 0

        # Test with all ones
        x = torch.tensor([1, 1, 1, 1])  # binary 1111 = decimal 15
        assert from_binary_tensor(x) == 15

        # Test with larger number
        x = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])  # binary 11111111 = decimal 255
        assert from_binary_tensor(x) == 255

        # Test with float tensor
        x = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert from_binary_tensor(x) == 10

        # Test with boolean tensor
        x = torch.tensor([True, False, True, False])
        assert from_binary_tensor(x) == 10

        # Test with empty tensor (should return 0)
        x = torch.tensor([])
        assert from_binary_tensor(x) == 0

    def test_apply_blockwise(self):
        """Test apply_blockwise function."""
        # Basic test case: apply NOT operation to each block of size 2
        x = torch.tensor([1, 0, 1, 0, 1, 1])
        result = apply_blockwise(x, 2, lambda b: 1 - b)
        expected = torch.tensor([0, 1, 0, 1, 0, 0])
        assert torch.all(result == expected)

        # Test with more complex function (sum each block)
        x = torch.tensor([1, 2, 3, 4, 5, 6])
        result = apply_blockwise(x, 2, lambda b: torch.sum(b, dim=-1, keepdim=True).repeat(1, 2))
        expected = torch.tensor([3, 3, 7, 7, 11, 11])
        assert torch.all(result == expected)

        # Test with batch dimension
        x = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        result = apply_blockwise(x, 2, lambda b: 1 - b)
        expected = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        assert torch.all(result == expected)

        # Test with non-divisible block size (should raise AssertionError)
        x = torch.tensor([1, 0, 1, 0, 1])
        with pytest.raises(AssertionError):
            apply_blockwise(x, 2, lambda b: 1 - b)

        # Test with different block sizes
        x = torch.tensor([1, 0, 1, 0, 1, 0])
        # Block size 3
        result = apply_blockwise(x, 3, lambda b: 1 - b)
        expected = torch.tensor([0, 1, 0, 1, 0, 1])
        assert torch.all(result == expected)

        # Test with block size equal to tensor size
        x = torch.tensor([1, 0, 1, 0])
        result = apply_blockwise(x, 4, lambda b: 1 - b)
        expected = torch.tensor([0, 1, 0, 1])
        assert torch.all(result == expected)

        # Test with multiple dimensions
        x = torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
        result = apply_blockwise(x, 2, lambda b: 1 - b)
        expected = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
        assert torch.all(result == expected)
