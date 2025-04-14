# tests/metrics/test_signal_metrics.py
"""Tests for signal metrics including BER, BLER, FER, SER, SNR."""
import pytest
import torch

from kaira.metrics.signal import (
    BER,
    BLER,
    FER,
    SER,
    SNR,
    BitErrorRate,
    BlockErrorRate,
    FrameErrorRate,
    SignalToNoiseRatio,
    SymbolErrorRate,
)

def test_ber_complex_input():
    """Test BER calculation with complex input tensors."""
    metric = BitErrorRate()

    # Create complex tensors
    transmitted_complex = torch.tensor([1+1j, 0+0j, 1+0j, 0+1j], dtype=torch.complex64)
    received_complex = torch.tensor([1+1j, 1+0j, 0+0j, 0+1j], dtype=torch.complex64) # 2 errors (real part of 2nd, real part of 3rd)

    # Expected real/imaginary concatenated tensors after internal processing
    # transmitted: [1, 0, 1, 0], [1, 0, 0, 1] -> [1, 1, 0, 0, 1, 0, 0, 1]
    # received:    [1, 1, 0, 0], [1, 0, 0, 1] -> [1, 1, 1, 0, 0, 0, 0, 1]
    # Thresholding (0.5):
    # transmitted_bits: [True, True, False, False, True, False, False, True]
    # received_bits:    [True, True, True, False, False, False, False, True]
    # Errors:           [F,    F,    T,     F,     T,     F,     F,    F] -> 2 errors

    # Test forward pass
    ber_forward = metric.forward(transmitted_complex, received_complex)
    expected_ber_forward = 2.0 / 8.0 # 2 errors out of 8 total "bits" (real+imag)
    assert torch.isclose(ber_forward, torch.tensor(expected_ber_forward)), f"Forward BER mismatch: expected {expected_ber_forward}, got {ber_forward}"

    # Test update and compute
    metric.reset()
    metric.update(transmitted_complex, received_complex)
    ber_compute = metric.compute()
    expected_ber_compute = 2.0 / 8.0
    assert torch.isclose(ber_compute, torch.tensor(expected_ber_compute)), f"Computed BER mismatch: expected {expected_ber_compute}, got {ber_compute}"

def test_ber_empty_input():
    """Test BER with empty input tensors."""
    metric = BitErrorRate()
    transmitted = torch.tensor([])
    received = torch.tensor([])
    assert metric.forward(transmitted, received) == 0.0
    metric.update(transmitted, received)
    assert metric.compute() == 0.0

def test_ber_basic_real():
    """Test basic BER calculation with real tensors."""
    metric = BitErrorRate()
    transmitted = torch.tensor([1, 0, 1, 0])
    received = torch.tensor([1, 1, 0, 0]) # 2 errors
    expected_ber = 2.0 / 4.0

    # Test forward
    ber_forward = metric.forward(transmitted, received)
    assert torch.isclose(ber_forward, torch.tensor(expected_ber)), f"Forward BER mismatch: expected {expected_ber}, got {ber_forward}"

    # Test update/compute
    metric.reset()
    metric.update(transmitted, received)
    ber_compute = metric.compute()
    assert torch.isclose(ber_compute, torch.tensor(expected_ber)), f"Computed BER mismatch: expected {expected_ber}, got {ber_compute}"

def test_ber_threshold():
    """Test BER with a different threshold."""
    metric = BitErrorRate(threshold=0.1)
    transmitted = torch.tensor([0.9, 0.05, 0.8, 0.08]) # bits: [1, 0, 1, 0]
    received = torch.tensor([0.7, 0.5, 0.05, 0.09])   # bits: [1, 1, 0, 0] -> 2 errors
    expected_ber = 2.0 / 4.0

    # Test forward
    ber_forward = metric.forward(transmitted, received)
    assert torch.isclose(ber_forward, torch.tensor(expected_ber)), f"Forward BER mismatch (threshold): expected {expected_ber}, got {ber_forward}"

    # Test update/compute
    metric.reset()
    metric.update(transmitted, received)
    ber_compute = metric.compute()
    assert torch.isclose(ber_compute, torch.tensor(expected_ber)), f"Computed BER mismatch (threshold): expected {expected_ber}, got {ber_compute}"

def test_ber_batched_input_forward_only():
    """Test BER calculation with batched input tensors (forward only)."""
    # Note: Batched processing in forward currently returns per-batch results,
    # while update/compute aggregates. This test focuses on forward.
    # TODO: Update this test if batch handling in forward changes.
    metric = BitErrorRate()

    transmitted_batch = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
    received_batch = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 1]])
    # Batch 1: 2 errors / 4 bits = 0.5
    # Batch 2: 2 errors / 4 bits = 0.5

    # Temporarily enable batched logic for testing (assuming future implementation)
    # This part might need adjustment based on the actual is_batched implementation
    # For now, we test the non-batched path as is_batched is hardcoded to False
    # If is_batched were True:
    # ber_forward_batch = metric.forward(transmitted_batch, received_batch)
    # assert torch.allclose(ber_forward_batch, expected_ber_batch), f"Forward batched BER mismatch: expected {expected_ber_batch}, got {ber_forward_batch}"

    # Test current behavior (non-batched aggregation in forward)
    # Errors: (1!=1)=F, (0!=1)=T, (1!=0)=T, (0!=0)=F -> 2 errors in batch 1
    # Errors: (1!=1)=F, (1!=0)=T, (0!=0)=F, (0!=1)=T -> 2 errors in batch 2
    # Total errors = 4, Total bits = 8
    expected_ber_aggregated = 4.0 / 8.0
    ber_forward_aggregated = metric.forward(transmitted_batch, received_batch)
    assert torch.isclose(ber_forward_aggregated, torch.tensor(expected_ber_aggregated)), f"Forward aggregated BER mismatch: expected {expected_ber_aggregated}, got {ber_forward_aggregated}"


def test_ber_update_compute_multiple_batches():
    """Test BER update/compute over multiple batches."""
    metric = BitErrorRate()

    # Batch 1
    transmitted1 = torch.tensor([1, 0, 1, 0])
    received1 = torch.tensor([1, 1, 0, 0]) # 2 errors / 4 bits
    metric.update(transmitted1, received1)
    ber1 = metric.compute()
    assert torch.isclose(ber1, torch.tensor(2.0/4.0))

    # Batch 2 (complex)
    transmitted2_complex = torch.tensor([1+1j, 0+0j], dtype=torch.complex64) # real/imag: [1, 0], [1, 0] -> [1, 1, 0, 0]
    received2_complex = torch.tensor([1+0j, 0+1j], dtype=torch.complex64)   # real/imag: [1, 0], [0, 1] -> [1, 0, 0, 1]
    # bits_t: [T, T, F, F]
    # bits_r: [T, F, F, T]
    # errors: [F, T, F, T] -> 2 errors / 4 "bits"
    metric.update(transmitted2_complex, received2_complex)
    ber2 = metric.compute()
    # Total errors = 2 (batch1) + 2 (batch2) = 4
    # Total bits = 4 (batch1) + 4 (batch2) = 8
    expected_ber2 = 4.0 / 8.0
    assert torch.isclose(ber2, torch.tensor(expected_ber2)), f"Cumulative BER mismatch after batch 2: expected {expected_ber2}, got {ber2}"

    # Reset
    metric.reset()
    assert metric.compute() == 0.0
    assert metric.total_bits == 0
    assert metric.error_bits == 0


# ===== BitErrorRate (BER) Tests =====


class TestBitErrorRate:
    """Test suite for BitErrorRate metrics."""

    def test_bit_error_rate_computation(self):
        """Test BitErrorRate computation."""
        # Create test data - 75% match, 25% errors
        transmitted = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).float()
        received = torch.tensor([0, 1, 1, 1, 0, 0, 0, 1]).float()

        # Initialize BER metric
        ber_metric = BitErrorRate()

        # Test forward computation
        error_rate = ber_metric(transmitted, received)
        assert error_rate.item() == 0.25

        # Test update and compute (accumulated statistics)
        ber_metric.reset()
        ber_metric.update(transmitted, received)
        assert ber_metric.compute().item() == 0.25

        # Test with different threshold
        ber_metric = BitErrorRate(threshold=0.7)
        received_analog = torch.tensor([0.1, 0.9, 0.6, 0.8, 0.2, 0.3, 0.1, 0.9]).float()
        error_rate = ber_metric(transmitted, received_analog)
        assert error_rate.item() == 0.125  # Only one bit (0.6) is misclassified

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_bit_error_rate_with_different_thresholds(self, threshold):
        """Test BitErrorRate with different decision thresholds."""
        transmitted = torch.tensor([0, 1, 0, 1]).float()
        received = torch.tensor([0.2, 0.6, 0.4, 0.8]).float()

        ber_metric = BitErrorRate(threshold=threshold)
        error_rate = ber_metric(transmitted, received)

        # Calculate expected errors based on threshold
        transmitted_bits = (transmitted > threshold).bool()
        received_bits = (received > threshold).bool()
        expected_errors = (transmitted_bits != received_bits).float().mean().item()

        assert error_rate.item() == expected_errors

    def test_bit_error_rate_batched(self):
        """Test BitErrorRate with batched data."""
        # Create batched test data (2 samples)
        transmitted = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]).float()
        received = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1]]).float()

        ber_metric = BitErrorRate()
        error_rate = ber_metric(transmitted, received)

        # Check shape and values
        assert error_rate.shape == torch.Size([2])
        assert error_rate[0].item() == 0.25  # 1/4 errors in first sample
        assert error_rate[1].item() == 0.5  # 2/4 errors in second sample

    def test_ber_zero_errors(self, binary_data):
        """Test BER computation with zero errors."""
        true_bits, _ = binary_data
        ber = BitErrorRate()
        ber_value = ber(true_bits, true_bits)  # Compare with itself
        assert ber_value == 0.0

    def test_ber_all_errors(self):
        """Test BER computation with all errors."""
        bits = torch.zeros((1, 100))
        inverted_bits = torch.ones((1, 100))
        ber = BitErrorRate()
        ber_value = ber(inverted_bits, bits)
        assert ber_value == 1.0

    @pytest.mark.parametrize("error_rate", [0.0, 0.1, 0.5, 1.0])
    def test_ber_specific_error_rates(self, error_rate):
        """Test BER computation with specific error rates."""
        n_bits = 1000
        true_bits = torch.zeros((1, n_bits))
        errors = torch.rand(1, n_bits) < error_rate
        received_bits = torch.logical_xor(true_bits, errors).int()

        ber = BitErrorRate()
        ber_value = ber(received_bits, true_bits)
        assert abs(ber_value.item() - error_rate) < 0.05  # Allow for statistical variation

    def test_ber_empty_tensors(self):
        """Test BER computation with empty tensors."""
        ber = BitErrorRate()

        # Create empty tensors
        empty_preds = torch.zeros((0, 10))
        empty_targets = torch.zeros((0, 10))

        # Test forward method with empty tensors
        result = ber(empty_preds, empty_targets)
        assert torch.isclose(result, torch.tensor(0.0))

        # Test update method with empty tensors
        ber.reset()
        ber.update(empty_preds, empty_targets)
        assert torch.isclose(ber.compute(), torch.tensor(0.0))


# ===== BlockErrorRate (BLER) Tests =====


class TestBlockErrorRate:
    """Test suite for BlockErrorRate metrics."""

    def test_block_error_rate_computation(self):
        """Test BlockErrorRate computation."""
        # Create test data - 1st block correct, 2nd has errors, 3rd correct, 4th has errors
        transmitted = torch.tensor(
            [
                [1, 1, 1, 1],  # Block 1
                [0, 0, 0, 0],  # Block 2
                [1, 0, 1, 0],  # Block 3
                [0, 1, 0, 1],  # Block 4
            ]
        ).float()

        received = torch.tensor(
            [
                [1, 1, 1, 1],  # Block 1 - correct
                [0, 0, 1, 0],  # Block 2 - has error
                [1, 0, 1, 0],  # Block 3 - correct
                [1, 1, 0, 1],  # Block 4 - has error
            ]
        ).float()

        # Initialize BLER metric
        bler_metric = BlockErrorRate()

        # Test forward computation
        block_error_rate = bler_metric(transmitted, received)
        assert block_error_rate.item() == 0.5  # 2 out of 4 blocks have errors

        # Test update and compute
        bler_metric.reset()
        bler_metric.update(transmitted, received)
        assert bler_metric.compute().item() == 0.5

    @pytest.mark.parametrize("block_size", [10, 50, 100])
    def test_bler_computation_with_block_size(self, binary_data, block_size):
        """Test BLER computation with different block sizes."""
        true_bits, received_bits = binary_data
        n_blocks = true_bits.size(1) // block_size
        usable_bits = n_blocks * block_size

        # Reshape into blocks
        true_blocks = true_bits[:, :usable_bits].reshape(1, n_blocks, block_size)
        received_blocks = received_bits[:, :usable_bits].reshape(1, n_blocks, block_size)

        bler = BlockErrorRate(block_size=block_size)
        # Direct computation
        bler_value = bler(received_blocks, true_blocks)
        assert isinstance(bler_value, torch.Tensor)
        assert bler_value.ndim == 0  # Scalar output
        assert 0 <= bler_value <= 1  # BLER should be between 0 and 1

        # Test the update+compute path
        bler.reset()
        bler.update(received_blocks, true_blocks)
        computed_bler = bler.compute()
        assert isinstance(computed_bler, torch.Tensor)
        assert computed_bler.ndim == 0
        assert abs(computed_bler.item() - bler_value.item()) < 1e-5  # Both methods should give same result

    def test_block_error_edge_cases(self):
        """Test BLER computation with edge cases."""
        bler = BlockErrorRate()

        # Test perfect transmission (no errors)
        perfect_blocks = torch.zeros((1, 10, 8))
        bler_value = bler(perfect_blocks, perfect_blocks)
        assert bler_value == 0.0

        # Test completely corrupted transmission (all errors)
        corrupted_blocks = torch.ones((1, 10, 8))
        clean_blocks = torch.zeros((1, 10, 8))
        bler_value = bler(corrupted_blocks, clean_blocks)
        assert bler_value == 1.0

    @pytest.mark.parametrize("block_size,error_pattern", [(10, [0]), (20, [1, 3]), (50, [-1]), (100, [0, -1])])  # Error in first block  # Errors in middle blocks  # Error in last block  # Errors in first and last blocks
    def test_bler_specific_patterns(self, block_size, error_pattern):
        """Test BLER computation with specific error patterns."""
        n_blocks = 10
        n_bits = block_size * n_blocks
        true_bits = torch.zeros((1, n_bits))
        received_bits = true_bits.clone()

        # Introduce errors in specific blocks
        for block_idx in error_pattern:
            idx = block_idx if block_idx >= 0 else n_blocks + block_idx
            start_idx = idx * block_size
            received_bits[0, start_idx] = 1  # Introduce an error in the block

        blocks = true_bits.reshape(1, n_blocks, block_size)
        received_blocks = received_bits.reshape(1, n_blocks, block_size)

        bler = BlockErrorRate(block_size=block_size)  # Explicitly set block_size
        bler.update(received_blocks, blocks)
        bler_value = bler.compute()

        # If there's any bit error in a block, the entire block is counted as an error
        expected_bler = len(error_pattern) / n_blocks
        assert abs(bler_value.item() - expected_bler) < 1e-6

    def test_bler_with_small_batch(self):
        """Test BlockErrorRate with a small batch to avoid empty tensor issues."""
        bler = BlockErrorRate(block_size=10)

        # Create small batch tensors instead of empty ones
        small_preds = torch.zeros((1, 10))
        small_target = torch.zeros((1, 10))

        # Update with small batch data
        bler.update(small_preds, small_target)

        # Compute result
        result = bler.compute()

        # Should return 0 for a perfect match
        assert isinstance(result, torch.Tensor)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_bler_reset(self):
        """Test that reset clears accumulated statistics for BLER."""
        bler = BlockErrorRate(block_size=10)

        # Create test data
        preds = torch.zeros(1, 100)
        target = torch.ones(1, 100)  # All blocks will have errors

        # First update
        bler.update(preds, target)
        first_result = bler.compute()

        # Reset
        bler.reset()

        # Update with different data
        new_preds = torch.ones(1, 100)
        new_target = torch.ones(1, 100)  # No errors
        bler.update(new_preds, new_target)
        second_result = bler.compute()

        # Results should be different
        assert not torch.isclose(first_result, second_result)
        assert torch.isclose(second_result, torch.tensor(0.0))  # Should be 0 error rate

    def test_bler_with_threshold(self):
        """Test BlockErrorRate with different thresholds."""
        # Create data with values that will give different results with different thresholds
        soft_preds = torch.tensor([[0.45, 0.55, 0.45, 0.55, 0.45, 0.55], [0.45, 0.55, 0.45, 0.55, 0.45, 0.55]])  # All close to threshold
        # Intentionally mismatch the targets from what a 0.5 threshold would give
        target = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])

        # With threshold 0.5
        bler_default = BlockErrorRate(block_size=2, threshold=0.5)
        bler_default.update(soft_preds, target)
        result_default = bler_default.compute()

        # With threshold 0.6 - this should change which predictions are considered 1s vs 0s
        bler_higher = BlockErrorRate(block_size=2, threshold=0.6)
        bler_higher.update(soft_preds, target)
        result_higher = bler_higher.compute()

        # Verify that the results are different due to different thresholds
        assert not torch.allclose(result_default, result_higher)

    def test_bler_with_different_batch_sizes(self, random_binary_data):
        """Test BlockErrorRate with different batch sizes."""
        true_bits, received_bits = random_binary_data

        # Split into single batches
        true_batch1, true_batch2 = true_bits[0:1], true_bits[1:2]
        received_batch1, received_batch2 = received_bits[0:1], received_bits[1:2]

        # Process in a single batch
        bler_single = BlockErrorRate(block_size=50)
        bler_single.update(received_bits, true_bits)
        result_single = bler_single.compute()

        # Process in multiple batches
        bler_multiple = BlockErrorRate(block_size=50)
        bler_multiple.update(received_batch1, true_batch1)
        bler_multiple.update(received_batch2, true_batch2)
        result_multiple = bler_multiple.compute()

        # Results should be the same regardless of batch processing
        assert torch.isclose(result_single, result_multiple)

    def test_bler_reshape_errors(self):
        """Test error handling in reshape_into_blocks method."""
        # Test case where input size is not divisible by block_size
        bler = BlockErrorRate(block_size=3)  # Block size that doesn't evenly divide the input

        # Create inputs with length not divisible by block_size
        preds = torch.zeros((2, 10))  # 10 is not divisible by 3
        target = torch.zeros((2, 10))

        # This should raise a ValueError
        with pytest.raises(ValueError, match="Input size .* is not divisible by"):
            bler(preds, target)

    def test_bler_multidimensional_input(self):
        """Test BlockErrorRate with multidimensional inputs."""
        # Create 3D input tensors
        bler = BlockErrorRate(block_size=2)

        # Create 3D tensors (batch, height, width) that can be reshaped into blocks
        preds = torch.zeros((2, 2, 4))  # Can be reshaped to (2, 4, 2)
        target = torch.zeros((2, 2, 4))

        # This should process without errors
        result = bler(preds, target)

        # Should return 0 for a perfect match
        assert isinstance(result, torch.Tensor)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_bler_with_different_reductions(self):
        """Test BlockErrorRate with different reduction methods."""
        # Create test data with known errors
        preds = torch.zeros((2, 6))
        target = torch.zeros((2, 6))

        # Introduce errors in specific blocks
        preds[0, 0] = 1  # Error in first block of first batch
        preds[1, 3] = 1  # Error in second block of second batch

        # Test with 'none' reduction
        bler_none = BlockErrorRate(block_size=3, reduction="none")
        result_none = bler_none(preds, target)

        # Should return a tensor with shape [2, 2] (batch_size x num_blocks)
        assert result_none.shape[0] == 2
        assert torch.allclose(result_none, torch.tensor([[1.0, 0.0], [0.0, 1.0]]).float())

        # Test with 'sum' reduction
        bler_sum = BlockErrorRate(block_size=3, reduction="sum")
        result_sum = bler_sum(preds, target)

        # Should return the sum of error blocks (2 in this case)
        assert torch.isclose(result_sum, torch.tensor(2.0))

        # Test with default 'mean' reduction for comparison
        bler_mean = BlockErrorRate(block_size=3)
        result_mean = bler_mean(preds, target)

        # Should return the average (2/4 = 0.5 in this case)
        assert torch.isclose(result_mean, torch.tensor(0.5))

    def test_bler_empty_state(self):
        """Test compute method when no updates have been made."""
        bler = BlockErrorRate(block_size=10)

        # Compute without any updates
        result = bler.compute()

        # Should return 0 when no updates have been made
        assert torch.isclose(result, torch.tensor(0.0))

        # Test with intentionally empty batches
        empty_preds = torch.zeros((0, 10))
        empty_target = torch.zeros((0, 10))

        bler.update(empty_preds, empty_target)
        result_after_empty = bler.compute()

        # Should still return 0
        assert torch.isclose(result_after_empty, torch.tensor(0.0))

    def test_bler_shape_mismatch(self):
        """Test BlockErrorRate with mismatched shapes."""
        bler = BlockErrorRate(block_size=10)

        # Create mismatched shapes
        preds = torch.zeros((2, 20))
        target = torch.zeros((2, 10))

        # This should raise a ValueError
        with pytest.raises(ValueError, match="Shape mismatch"):
            bler(preds, target)

    def test_bler_empty_tensors(self):
        """Test BLER computation with empty tensors."""
        bler = BlockErrorRate(block_size=10)

        # Create empty tensors
        empty_preds = torch.zeros((0, 10))
        empty_targets = torch.zeros((0, 10))

        # Test forward method with empty tensors
        result = bler(empty_preds, empty_targets)
        assert torch.isclose(result, torch.tensor(0.0))

        # Test update method with empty tensors
        bler.reset()
        bler.update(empty_preds, empty_targets)
        assert torch.isclose(bler.compute(), torch.tensor(0.0))

    def test_bler_reshape_with_none_block_size(self):
        """Test _reshape_into_blocks with block_size=None."""
        # Initialize BLER with block_size=None
        bler = BlockErrorRate(block_size=None)

        # Create test data
        preds = torch.zeros((3, 10))
        targets = torch.zeros((3, 10))
        targets[1, 5] = 1.0  # Error in second block

        # Call forward - this will indirectly test _reshape_into_blocks
        result = bler(preds, targets)

        # When block_size is None, each row is treated as a separate block
        # So 1 out of 3 rows/blocks has an error
        assert torch.isclose(result, torch.tensor(1 / 3))

        # Test update path as well
        bler.reset()
        bler.update(preds, targets)
        assert torch.isclose(bler.compute(), torch.tensor(1 / 3))

    def test_bler_reshape_with_empty_tensor(self):
        """Test _reshape_into_blocks with empty tensor."""
        bler = BlockErrorRate(block_size=10)

        # Create empty tensor
        empty_tensor = torch.zeros((0, 0))

        # Access the private method directly to test the exact line
        reshaped = bler._reshape_into_blocks(empty_tensor)

        # Verify shape of reshaped empty tensor
        assert reshaped.shape == torch.Size([0, 0, 0])
        assert reshaped.numel() == 0

    def test_bler_reshape_with_none_block_size_direct(self):
        """Test _reshape_into_blocks with block_size=None directly."""
        # Initialize BLER with block_size=None
        bler = BlockErrorRate(block_size=None)

        # Create test data
        test_data = torch.zeros((3, 10))

        # Call _reshape_into_blocks directly to exercise the specific line
        reshaped = bler._reshape_into_blocks(test_data)

        # When block_size is None, the method should return the input unchanged
        assert torch.equal(reshaped, test_data)
        assert reshaped.shape == test_data.shape

        # Try with more complex tensor shape
        complex_data = torch.zeros((2, 4, 5))
        reshaped_complex = bler._reshape_into_blocks(complex_data)

        # Should still return unchanged
        assert torch.equal(reshaped_complex, complex_data)
        assert reshaped_complex.shape == complex_data.shape


# ===== FrameErrorRate (FER) Tests =====


class TestFrameErrorRate:
    """Test suite for FrameErrorRate metrics."""

    def test_frame_error_rate_computation(self):
        """Test FrameErrorRate computation."""
        # Create test data - 2 frames, first has error, second is correct
        transmitted = torch.tensor(
            [
                [1, 1, 1, 1],  # Frame 1
                [0, 0, 0, 0],  # Frame 2
            ]
        ).float()

        received = torch.tensor(
            [
                [1, 0, 1, 1],  # Frame 1 - has error
                [0, 0, 0, 0],  # Frame 2 - correct
            ]
        ).float()

        # Initialize FER metric
        fer_metric = FrameErrorRate()

        # Test forward computation
        frame_error_rate = fer_metric(transmitted, received)
        assert frame_error_rate.item() == 0.5  # 1 out of 2 frames has errors

        # Test update and compute
        fer_metric.reset()
        fer_metric.update(transmitted, received)
        assert fer_metric.compute().item() == 0.5

    @pytest.mark.parametrize("frame_size", [100, 200])
    def test_fer_computation(self, binary_data, frame_size):
        """Test FER computation with different frame sizes."""
        true_bits, received_bits = binary_data
        n_frames = true_bits.size(1) // frame_size
        usable_bits = n_frames * frame_size

        # Reshape into frames
        true_frames = true_bits[:, :usable_bits].reshape(1, -1, frame_size)
        received_frames = received_bits[:, :usable_bits].reshape(1, -1, frame_size)

        fer = FrameErrorRate()
        # Direct computation
        fer_value = fer(received_frames, true_frames)
        assert isinstance(fer_value, torch.Tensor)
        assert fer_value.ndim == 0  # Scalar output
        assert 0 <= fer_value <= 1  # FER should be between 0 and 1


# ===== SymbolErrorRate (SER) Tests =====


class TestSymbolErrorRate:
    """Test suite for SymbolErrorRate metrics."""

    def test_symbol_error_rate_computation(self):
        """Test SymbolErrorRate computation."""
        # Create test data with symbols (multi-bit values)
        transmitted = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        received = torch.tensor([0, 1, 3, 3, 1, 1, 0, 3])

        # Initialize SER metric
        ser_metric = SymbolErrorRate()

        # Test forward computation
        symbol_error_rate = ser_metric(transmitted, received)
        assert symbol_error_rate.item() == 0.375  # 3 out of 8 symbols are wrong

        # Test update and compute
        ser_metric.reset()
        ser_metric.update(transmitted, received)
        assert ser_metric.compute().item() == 0.375

    @pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])
    def test_ser_computation(self, binary_data, bits_per_symbol):
        """Test SER computation with different modulation orders."""
        true_bits, received_bits = binary_data
        n_symbols = true_bits.size(1) // bits_per_symbol
        usable_bits = n_symbols * bits_per_symbol

        # Reshape bits into symbols
        true_symbols = true_bits[:, :usable_bits].reshape(1, -1, bits_per_symbol)
        received_symbols = received_bits[:, :usable_bits].reshape(1, -1, bits_per_symbol)

        # Use SymbolErrorRate (which is actually a specialized BlockErrorRate)
        ser = SymbolErrorRate()
        ser_value = ser(received_symbols, true_symbols)

        assert isinstance(ser_value, torch.Tensor)
        assert ser_value.ndim == 0  # Scalar output
        assert 0 <= ser_value <= 1  # SER should be between 0 and 1

    @pytest.mark.parametrize("error_positions", [0, -1, "middle"])
    def test_ser_single_error(self, error_positions):
        """Test SER computation with single error in different positions."""
        ser = SymbolErrorRate()
        true_symbols = torch.zeros((1, 10, 4))  # 10 symbols, 4 bits each
        received_symbols = true_symbols.clone()

        if error_positions == "middle":
            error_pos = 5
        else:
            error_pos = error_positions

        received_symbols[0, error_pos, 0] = 1  # Introduce single error
        ser_value = ser(received_symbols, true_symbols)

        assert 0 <= ser_value <= 1.0
        assert ser_value > 0  # Ensure errors are detected

    @pytest.mark.parametrize("bits_per_symbol,error_positions", [(2, [0]), (4, [1, 2]), (6, [-1])])  # QPSK with error in first symbol  # 16-QAM with errors in middle symbols  # 64-QAM with error in last symbol
    def test_ser_specific_positions(self, bits_per_symbol, error_positions):
        """Test SER computation with errors in specific positions."""
        n_symbols = 10
        true_symbols = torch.zeros((1, n_symbols, bits_per_symbol))
        received_symbols = true_symbols.clone()

        # Introduce errors at specific positions
        for pos in error_positions:
            # Flip the bits to create errors (1s instead of 0s)
            received_symbols[0, pos] = torch.ones_like(received_symbols[0, pos])

        ser = SymbolErrorRate()
        ser_value = ser(received_symbols, true_symbols)

        assert 0 <= ser_value <= 1.0  # SER should be in valid range
        assert ser_value > 0  # Should detect at least some errors


# ===== SignalToNoiseRatio (SNR) Tests =====


class TestSignalToNoiseRatio:
    """Test suite for SignalToNoiseRatio metrics."""

    def test_signal_to_noise_ratio_with_batch(self):
        """Test SignalToNoiseRatio with batched data."""
        # Create batched clean signal and noisy signal
        clean_signal = torch.tensor([[1.0, -1.0, 0.5], [0.5, -0.5, 0.0]])
        noise = torch.tensor([[0.1, -0.1, 0.05], [0.05, 0.1, -0.05]])
        noisy_signal = clean_signal + noise

        # Initialize SNR metric
        snr_metric = SignalToNoiseRatio()

        # Test forward computation
        snr_values = snr_metric(clean_signal, noisy_signal)

        # Check shape
        assert snr_values.shape == torch.Size([2])

        # Calculate expected SNR for each sample
        for i in range(2):
            signal_power = (clean_signal[i] ** 2).mean()
            noise_power = (noise[i] ** 2).mean()
            expected_snr = 10 * torch.log10(signal_power / noise_power)
            assert torch.isclose(snr_values[i], expected_snr, rtol=1e-4)

    def test_snr_complex_signals(self):
        """Test SNR calculation with complex signals."""
        snr_metric = SignalToNoiseRatio()

        # Create complex signal
        signal = torch.complex(torch.tensor([1.0, 0.0, -1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, -1.0]))

        # Create complex noise
        noise = torch.complex(torch.tensor([0.1, 0.1, 0.1, 0.1]), torch.tensor([0.1, 0.1, 0.1, 0.1]))

        noisy_signal = signal + noise

        # Calculate SNR
        snr_db = snr_metric(signal, noisy_signal)

        # Calculate expected SNR for complex signals
        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = torch.mean(torch.abs(noise) ** 2)
        expected_snr_db = 10 * torch.log10(signal_power / noise_power)

        # Check result
        assert torch.isclose(snr_db, expected_snr_db, rtol=1e-4)

    def test_snr_zero_noise(self):
        """Test SNR calculation with zero noise (perfect signal)."""
        snr_metric = SignalToNoiseRatio()

        # Create signal
        signal = torch.tensor([1.0, -1.0, 1.0, -1.0])

        # Perfect reproduction (zero noise)
        noisy_signal = signal.clone()

        # Calculate SNR
        snr_db = snr_metric(signal, noisy_signal)

        # Result should be infinity
        assert torch.isinf(snr_db)
        assert snr_db > 0  # Positive infinity

    def test_snr_zero_signal(self):
        """Test SNR calculation with zero signal."""
        snr_metric = SignalToNoiseRatio()

        # Zero signal
        signal = torch.zeros(4)

        # Some noise
        noise = torch.tensor([0.1, -0.1, 0.1, -0.1])
        noisy_signal = signal + noise

        # Calculate SNR
        snr_db = snr_metric(signal, noisy_signal)

        # Result should be very negative (approaching -infinity)
        # But due to epsilon in the calculation, it won't be exactly -infinity
        assert snr_db < -100

    @pytest.mark.parametrize("snr_db", [-10, 0, 10, 20])
    def test_snr_db_values(self, signal_data, snr_db):
        """Test SNR computation with known SNR values in dB."""
        signal, _ = signal_data

        # Create noisy signal with specific SNR using correct power calculation
        signal_power = torch.mean(signal**2).item()
        noise_power = signal_power / (10 ** (snr_db / 10))  # Calculate required noise power for desired SNR

        # Generate noise with the exact power needed
        noise = torch.randn_like(signal)
        noise_scale = torch.sqrt(torch.tensor(noise_power) / torch.mean(noise**2))
        scaled_noise = noise * noise_scale

        # Create noisy signal
        noisy_signal = signal + scaled_noise

        # Calculate SNR
        snr = SignalToNoiseRatio()
        snr_value = snr(signal, noisy_signal)

        # Check the calculated SNR is close to the expected value
        assert abs(snr_value.item() - snr_db) < 1.0  # Allow for some numerical error

    def test_snr_with_different_dimensions(self):
        """Test SNR computation with inputs of different dimensions."""
        # Create 1D signals
        signal_1d = torch.randn(100)
        noise_1d = 0.1 * torch.randn(100)
        noisy_1d = signal_1d + noise_1d

        # Create 2D signals (batch of 1D signals)
        signal_2d = torch.randn(2, 100)
        noise_2d = 0.1 * torch.randn(2, 100)
        noisy_2d = signal_2d + noise_2d

        # Create 3D signals (e.g., spectrograms)
        signal_3d = torch.randn(2, 10, 10)
        noise_3d = 0.1 * torch.randn(2, 10, 10)
        noisy_3d = signal_3d + noise_3d

        # Compute SNR for each
        snr_metric = SignalToNoiseRatio()

        snr_1d = snr_metric(signal_1d, noisy_1d)
        snr_2d = snr_metric(signal_2d, noisy_2d)
        snr_3d = snr_metric(signal_3d, noisy_3d)

        # Check that results are valid
        assert isinstance(snr_1d, torch.Tensor)
        assert isinstance(snr_2d, torch.Tensor)
        assert isinstance(snr_3d, torch.Tensor)

        # Check dimensions
        assert snr_1d.ndim == 0  # Scalar

        # Per the implementation, if signal has batch dimension > 1, output is a tensor with batch size
        assert snr_2d.ndim == 1  # Vector (one SNR per batch item)
        assert snr_3d.ndim == 1  # Vector (one SNR per batch item)

    def test_snr_compute_with_stats(self):
        """Test compute_with_stats method of SNR metric."""
        snr_metric = SignalToNoiseRatio()

        # Create a batch of signals with different SNRs
        signal = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])

        # Different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        noises = []
        for level in noise_levels:
            noise = torch.tensor([level, -level, level, -level])
            noises.append(noise)

        noisy_signal = signal.clone()
        for i, noise in enumerate(noises):
            noisy_signal[i] = signal[i] + noise

        # Calculate mean and std of SNR
        mean_snr, std_snr = snr_metric.compute_with_stats(signal, noisy_signal)

        # Calculate expected SNRs
        expected_snrs = []
        for i, level in enumerate(noise_levels):
            signal_power = torch.mean(signal[i] ** 2)
            noise_power = level**2
            expected_snr = 10 * torch.log10(signal_power / noise_power)
            expected_snrs.append(expected_snr)

        expected_mean = sum(expected_snrs) / len(expected_snrs)
        expected_std = torch.tensor(expected_snrs).std()

        # Fix: Use clone().detach() instead of torch.tensor() for tensor construction
        assert torch.isclose(mean_snr, expected_mean, rtol=1e-3)
        assert torch.isclose(std_snr, expected_std, rtol=1e-3)

    def test_signal_to_noise_ratio_db(self):
        """Test SignalToNoiseRatio with dB input/output."""
        snr = SignalToNoiseRatio(mode="db")

        # Create signal and noise
        signal = torch.ones(10) * 2.0  # Signal with power 4.0
        noise = torch.ones(10) * 1.0  # Noise with power 1.0
        noisy_signal = signal + noise

        # SNR should be 10*log10(4/1) = 6.02 dB
        result = snr(signal, noisy_signal)
        assert round(result.item(), 1) == 6.0

        # Test with zero signal (edge case)
        zero_signal = torch.zeros(10)
        result = snr(zero_signal, noise)
        assert result.item() == float("-inf")  # SNR is -infinity for zero signal

    def test_signal_to_noise_ratio_linear(self):
        """Test SignalToNoiseRatio with linear input/output."""
        snr = SignalToNoiseRatio(mode="linear")

        # Create signal and noise
        signal = torch.ones(10) * 2.0  # Signal with power 4.0
        noise = torch.ones(10) * 1.0  # Noise with power 1.0
        noisy_signal = signal + noise

        # Signal power = 4, noise power = 1
        # SNR = 4/1 = 4
        result = snr(signal, noisy_signal)
        assert result.item() == pytest.approx(4.0, abs=0.1)

    def test_snr_near_zero_noise(self):
        """Test SNR calculation with extremely small noise (approaching epsilon)."""
        snr_metric = SignalToNoiseRatio()

        # Create signal
        signal = torch.tensor([[1.0, -1.0, 1.0, -1.0]])

        # Create extremely small noise (below epsilon)
        eps = torch.finfo(torch.float32).eps
        noise = torch.ones_like(signal) * (eps / 2.0)
        noisy_signal = signal + noise

        # Calculate SNR
        snr_db = snr_metric(signal, noisy_signal)

        # Result should be infinity due to the noise_power < eps check
        assert torch.isinf(snr_db)
        assert snr_db > 0  # Positive infinity

    def test_linear_mode_batch_processing(self):
        """Test linear mode SNR with batched data to ensure correct path execution."""
        # Initialize SNR metric in linear mode
        snr_metric = SignalToNoiseRatio(mode="linear")

        # Create batched clean signal and noisy signal
        clean_signal = torch.tensor([[1.0, -1.0, 0.5], [0.5, -0.5, 0.0]])
        noise = torch.tensor([[0.1, -0.1, 0.05], [0.05, 0.1, -0.05]])
        noisy_signal = clean_signal + noise

        # Test forward computation
        snr_values = snr_metric(clean_signal, noisy_signal)

        # Check shape
        assert snr_values.shape == torch.Size([2])

        # Calculate expected linear SNR for each sample
        for i in range(2):
            signal_power = (clean_signal[i] ** 2).mean()
            noise_power = (noise[i] ** 2).mean()
            expected_snr = signal_power / noise_power  # Linear ratio, not in dB
            assert torch.isclose(snr_values[i], expected_snr, rtol=1e-4)

    def test_mode_initialization_validation(self):
        """Test that the mode validation works properly during initialization."""
        # Valid modes should initialize correctly
        SignalToNoiseRatio(mode="db")
        SignalToNoiseRatio(mode="linear")

        # Invalid mode should raise ValueError
        with pytest.raises(ValueError, match="Mode must be either 'db' or 'linear'"):
            SignalToNoiseRatio(mode="invalid")

    def test_batched_snr_with_zero_noise(self):
        """Test SNR calculation with batched data where noise is below epsilon."""
        snr_metric = SignalToNoiseRatio()

        # Create a batch of signals
        batch_size = 3
        signal = torch.ones((batch_size, 10)) * torch.tensor([1.0, 2.0, 3.0]).view(-1, 1)

        # Create extremely small noise (below epsilon)
        eps = torch.finfo(torch.float32).eps
        noise = torch.ones_like(signal) * (eps / 2.0)
        noisy_signal = signal + noise

        # Calculate SNR
        snr_values = snr_metric(signal, noisy_signal)

        # Check shape and values
        assert snr_values.shape == torch.Size([batch_size])

        # All values should be infinity due to the noise_power < eps check in batched mode
        for value in snr_values:
            assert torch.isinf(value)
            assert value > 0  # Positive infinity

    def test_batched_complex_signal_snr(self):
        """Test SNR calculation with batched complex signals."""
        snr_metric = SignalToNoiseRatio()

        # Create a batch of complex signals
        batch_size = 2
        real_part = torch.tensor([[1.0, 0.0, -1.0, 0.0], [2.0, 0.0, -2.0, 0.0]])
        imag_part = torch.tensor([[0.0, 1.0, 0.0, -1.0], [0.0, 2.0, 0.0, -2.0]])
        signal = torch.complex(real_part, imag_part)

        # Create complex noise with different levels for each batch item
        noise_levels = [0.1, 0.2]
        noise_real = torch.zeros_like(real_part)
        noise_imag = torch.zeros_like(imag_part)

        for i, level in enumerate(noise_levels):
            noise_real[i] = torch.ones(4) * level
            noise_imag[i] = torch.ones(4) * level

        noise = torch.complex(noise_real, noise_imag)
        noisy_signal = signal + noise

        # Calculate SNR
        snr_values = snr_metric(signal, noisy_signal)

        # Check shape
        assert snr_values.shape == torch.Size([batch_size])

        # Check values
        for i, level in enumerate(noise_levels):
            # Calculate expected SNR
            signal_power = torch.mean(torch.abs(signal[i]) ** 2).item()
            noise_power = torch.mean(torch.abs(noise[i]) ** 2).item()
            expected_snr = 10 * torch.log10(torch.tensor(signal_power / noise_power))

            # Check that the calculated SNR matches the expected value
            assert torch.isclose(snr_values[i], expected_snr, rtol=1e-3)


# ===== Common Tests =====


def test_metrics_aliases():
    """Test that metric aliases work properly."""
    assert BitErrorRate is BER
    assert BlockErrorRate is BLER
    assert FrameErrorRate is FER
    assert SymbolErrorRate is SER
    assert SignalToNoiseRatio is SNR
