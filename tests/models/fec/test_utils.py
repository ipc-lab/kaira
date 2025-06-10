"""Tests for the utils module in kaira.models.fec package."""

import pytest
import torch

from kaira.models.fec.utils import (
    Taylor_arctanh,
    apply_blockwise,
    cyclic_perm,
    from_binary_tensor,
    hamming_distance,
    hamming_weight,
    llr_to_bits,
    min_sum,
    reorder_from_idx,
    row_reduction,
    sign_to_bin,
    stop_criterion,
    sum_product,
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

    def test_taylor_arctanh(self):
        """Test Taylor_arctanh function."""
        # Test basic approximation
        x = torch.tensor([0.5])
        result = Taylor_arctanh(x, num_series=10)
        expected = torch.arctanh(x)
        assert torch.allclose(result, expected, atol=1e-3)

        # Test with zero
        x = torch.tensor([0.0])
        result = Taylor_arctanh(x, num_series=10)
        expected = torch.tensor([0.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with small values
        x = torch.tensor([0.1, 0.2, 0.3])
        result = Taylor_arctanh(x, num_series=20)
        expected = torch.arctanh(x)
        assert torch.allclose(result, expected, atol=1e-4)

        # Test with batch dimension
        x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        result = Taylor_arctanh(x, num_series=15)
        expected = torch.arctanh(x)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_sign_to_bin(self):
        """Test sign_to_bin function."""
        # Test basic conversion: sign_to_bin(x) = 0.5 * (1 - x)
        x = torch.tensor([1.0, -1.0, 1.0, -1.0])
        result = sign_to_bin(x)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert torch.allclose(result, expected)

        # Test with zero
        x = torch.tensor([1.0, 0.0, -1.0])
        result = sign_to_bin(x)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)

        # Test with batch dimension
        x = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        result = sign_to_bin(x)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.allclose(result, expected)

        # Test with large values: formula is 0.5 * (1 - x)
        x = torch.tensor([100.0, -50.0, 1000.0, -0.001])
        result = sign_to_bin(x)
        expected = 0.5 * (1 - x)  # Use actual formula
        assert torch.allclose(result, expected)

    def test_row_reduction(self):
        """Test row_reduction function."""
        # Test basic row reduction
        matrix = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=torch.int)
        reduced, rank = row_reduction(matrix)
        assert rank == 2  # This matrix has rank 2 in GF(2)
        assert reduced.shape == matrix.shape

        # Test with zero matrix
        matrix = torch.zeros((3, 3), dtype=torch.int)
        reduced, rank = row_reduction(matrix)
        assert rank == 0
        assert torch.equal(reduced, matrix)

        # Test with identity matrix
        matrix = torch.eye(3, dtype=torch.int)
        reduced, rank = row_reduction(matrix)
        assert rank == 3
        assert torch.equal(reduced, matrix)

        # Test with partial columns
        matrix = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=torch.int)
        reduced, rank = row_reduction(matrix, num_cols=2)
        assert rank <= 2
        assert reduced.shape == matrix.shape

    def test_reorder_from_idx(self):
        """Test reorder_from_idx function."""
        # Test basic reordering
        a = [1, 2, 3, 4, 5]
        result = reorder_from_idx(2, a)
        expected = [3, 4, 5, 1, 2]
        assert result == expected

        # Test with index 0 (no change)
        a = [1, 2, 3, 4]
        result = reorder_from_idx(0, a)
        assert result == a

        # Test with index at end
        a = [1, 2, 3, 4]
        result = reorder_from_idx(4, a)
        expected = [1, 2, 3, 4]  # a[4:] is empty, a[:4] is the whole list
        assert result == expected

        # Test with single element
        a = [42]
        result = reorder_from_idx(0, a)
        assert result == a
        result = reorder_from_idx(1, a)
        assert result == a  # a[1:] is empty, a[:1] is [42]

    def test_cyclic_perm(self):
        """Test cyclic_perm function."""
        # Test basic cyclic permutation
        a = [1, 2, 3]
        result = cyclic_perm(a)
        expected = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        assert result == expected

        # Test with single element
        a = [42]
        result = cyclic_perm(a)
        expected = [[42]]
        assert result == expected

        # Test with empty list
        a = []
        result = cyclic_perm(a)
        expected = []
        assert result == expected

        # Test with two elements
        a = [1, 2]
        result = cyclic_perm(a)
        expected = [[1, 2], [2, 1]]
        assert result == expected

    def test_stop_criterion(self):
        """Test stop_criterion function."""
        # Test when all codewords are satisfied
        x = torch.tensor([[1, 0, 1], [0, 1, 0]])
        u = torch.tensor([[1, 0], [0, 1]])
        code_gm = torch.tensor([[1, 0, 1], [0, 1, 0]])
        not_satisfied = torch.tensor([0, 1])

        result = stop_criterion(x, u, code_gm, not_satisfied)
        assert len(result) == 0

        # Test when no codewords are satisfied
        x = torch.tensor([[1, 0, 1], [0, 1, 0]])
        u = torch.tensor([[0, 1], [1, 0]])  # Wrong input
        code_gm = torch.tensor([[1, 0, 1], [0, 1, 0]])
        not_satisfied = torch.tensor([0, 1])

        result = stop_criterion(x, u, code_gm, not_satisfied)
        assert torch.equal(result, not_satisfied)

    def test_llr_to_bits(self):
        """Test llr_to_bits function."""
        # Test basic conversion
        x = torch.tensor([1.0, -1.0, 2.0, -0.5])
        result = llr_to_bits(x)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with zero
        x = torch.tensor([0.0])
        result = llr_to_bits(x)
        expected = torch.tensor([0.0])  # round(sigmoid(-0)) = round(0.5) = 0.0
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with batch dimension
        x = torch.tensor([[1.0, -1.0], [-2.0, 3.0]])
        result = llr_to_bits(x)
        expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with large values
        x = torch.tensor([100.0, -100.0, 1000.0, -0.001])
        result = llr_to_bits(x)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_min_sum(self):
        """Test min_sum function."""
        # Test basic operation
        x = torch.tensor([0.5, 1.0, -1.5])
        y = torch.tensor([-0.5, 2.0, -2.0])
        result = min_sum(x, y)
        expected = torch.tensor([-0.5, 1.0, 1.5])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with same signs
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([0.5, 1.5, 4.0])
        result = min_sum(x, y)
        expected = torch.tensor([0.5, 1.5, 3.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with opposite signs
        x = torch.tensor([1.0, -2.0, 3.0])
        y = torch.tensor([-0.5, 1.5, -4.0])
        result = min_sum(x, y)
        expected = torch.tensor([-0.5, -1.5, -3.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with zero values
        x = torch.tensor([0.0, 1.0, -1.0])
        y = torch.tensor([1.0, 0.0, -1.0])
        result = min_sum(x, y)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with batch dimension
        x = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
        y = torch.tensor([[0.5, -0.5], [3.0, -1.0]])
        result = min_sum(x, y)
        expected = torch.tensor([[0.5, 0.5], [2.0, 1.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_sum_product(self):
        """Test sum_product function."""
        # Test basic operation
        x = torch.tensor([0.5, 1.0, -1.5])
        y = torch.tensor([-0.5, 2.0, -2.0])
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with zero values
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([1.0, 0.0, 1.0])
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with same values
        x = torch.tensor([1.0, 2.0, -1.0])
        y = x.clone()
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with batch dimension
        x = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
        y = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with large values (should saturate)
        x = torch.tensor([10.0, -10.0])
        y = torch.tensor([5.0, -5.0])
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-5)

        # Test with small values
        x = torch.tensor([0.1, -0.1, 0.05])
        y = torch.tensor([0.2, -0.2, 0.1])
        result = sum_product(x, y)
        expected = 2 * torch.arctanh(torch.tanh(x / 2) * torch.tanh(y / 2))
        assert torch.allclose(result, expected, atol=1e-6)

    def test_integration_binary_conversion(self):
        """Test integration between binary conversion functions."""
        # Test multiple values
        for value in [0, 1, 7, 15, 255]:
            length = max(8, value.bit_length())

            # Convert to binary and back
            binary = to_binary_tensor(value, length)
            recovered = from_binary_tensor(binary)

            # Should be identical
            assert recovered == value

            # Test Hamming weight
            weight = hamming_weight(binary)
            expected_weight = bin(value).count("1")
            assert weight.item() == expected_weight

    def test_message_passing_operations(self):
        """Test message passing operations together."""
        x = torch.tensor([1.0, -0.5, 2.0, -1.5])
        y = torch.tensor([0.5, -1.0, 1.5, -2.0])

        # Test both min-sum and sum-product
        min_result = min_sum(x, y)
        sum_result = sum_product(x, y)

        # Both should have same shape
        assert min_result.shape == sum_result.shape
        assert min_result.shape == x.shape

        # For small values, they should be somewhat similar
        small_x = torch.tensor([0.1, -0.1])
        small_y = torch.tensor([0.2, -0.2])

        min_small = min_sum(small_x, small_y)
        sum_small = sum_product(small_x, small_y)

        # Should be roughly similar for small values
        assert torch.allclose(min_small, sum_small, atol=0.1)

    def test_sign_conversion_integration(self):
        """Test integration between sign and binary conversions."""
        # Test LLR to bits with sign conversion
        llr_values = torch.tensor([2.0, -1.5, 0.5, -3.0])

        # Convert LLR to bits
        bits = llr_to_bits(llr_values)

        # Convert to sign domain and back
        signs = 1 - 2 * bits  # Convert 0/1 to +1/-1
        recovered_bits = sign_to_bin(signs)

        assert torch.allclose(bits, recovered_bits, atol=1e-6)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test empty tensors
        empty = torch.tensor([])
        weight = hamming_weight(empty)
        assert weight.item() == 0

        distance = hamming_distance(empty, empty)
        assert distance.item() == 0

        # Test single element tensors
        single_zero = torch.tensor([0])
        single_one = torch.tensor([1])

        assert hamming_weight(single_zero).item() == 0
        assert hamming_weight(single_one).item() == 1

        assert hamming_distance(single_zero, single_zero).item() == 0
        assert hamming_distance(single_zero, single_one).item() == 1
        assert hamming_distance(single_one, single_one).item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test CUDA compatibility of utility functions."""
        x_cpu = torch.tensor([1, 0, 1, 0])
        y_cpu = torch.tensor([1, 1, 0, 0])

        x_cuda = x_cpu.cuda()
        y_cuda = y_cpu.cuda()

        # Test on CUDA
        distance_cuda = hamming_distance(x_cuda, y_cuda)
        weight_cuda = hamming_weight(x_cuda)

        # Compare with CPU results
        distance_cpu = hamming_distance(x_cpu, y_cpu)
        weight_cpu = hamming_weight(x_cpu)

        assert distance_cuda.cpu().item() == distance_cpu.item()
        assert weight_cuda.cpu().item() == weight_cpu.item()

        # Test float operations on CUDA
        x_float_cuda = torch.tensor([1.0, -0.5, 2.0]).cuda()
        y_float_cuda = torch.tensor([0.5, -1.0, 1.5]).cuda()

        min_cuda = min_sum(x_float_cuda, y_float_cuda)
        sum_cuda = sum_product(x_float_cuda, y_float_cuda)

        # Should not raise errors and results should be on CUDA
        assert min_cuda.device.type == "cuda"
        assert sum_cuda.device.type == "cuda"
