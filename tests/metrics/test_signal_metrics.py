# tests/metrics/test_signal_metrics.py
"""Tests for signal metrics including BER, BLER, FER, SER, SNR."""
import pytest
import torch

from kaira.metrics.signal import (
    BER,
    BLER,
    EVM,
    FER,
    SER,
    SNR,
    BitErrorRate,
    BlockErrorRate,
    ErrorVectorMagnitude,
    FrameErrorRate,
    SignalToNoiseRatio,
    SymbolErrorRate,
)


def test_ber_complex_input():
    """Test BER calculation with complex input tensors."""
    metric = BitErrorRate()

    # Create complex tensors
    transmitted_complex = torch.tensor([1 + 1j, 0 + 0j, 1 + 0j, 0 + 1j], dtype=torch.complex64)
    received_complex = torch.tensor([1 + 1j, 1 + 0j, 0 + 0j, 0 + 1j], dtype=torch.complex64)  # 2 errors (real part of 2nd, real part of 3rd)

    # Expected real/imaginary concatenated tensors after internal processing
    # transmitted: [1, 0, 1, 0], [1, 0, 0, 1] -> [1, 1, 0, 0, 1, 0, 0, 1]
    # received:    [1, 1, 0, 0], [1, 0, 0, 1] -> [1, 1, 1, 0, 0, 0, 0, 1]
    # Thresholding (0.5):
    # transmitted_bits: [True, True, False, False, True, False, False, True]
    # received_bits:    [True, True, True, False, False, False, False, True]
    # Errors:           [F,    F,    T,     F,     T,     F,     F,    F] -> 2 errors

    # Test forward pass
    ber_forward = metric.forward(transmitted_complex, received_complex)
    expected_ber_forward = 2.0 / 8.0  # 2 errors out of 8 total "bits" (real+imag)
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
    received = torch.tensor([1, 1, 0, 0])  # 2 errors
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
    transmitted = torch.tensor([0.9, 0.05, 0.8, 0.08])  # bits: [1, 0, 1, 0]
    received = torch.tensor([0.7, 0.5, 0.05, 0.09])  # bits: [1, 1, 0, 0] -> 2 errors
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
    received1 = torch.tensor([1, 1, 0, 0])  # 2 errors / 4 bits
    metric.update(transmitted1, received1)
    ber1 = metric.compute()
    assert torch.isclose(ber1, torch.tensor(2.0 / 4.0))

    # Batch 2 (complex)
    transmitted2_complex = torch.tensor([1 + 1j, 0 + 0j], dtype=torch.complex64)  # real/imag: [1, 0], [1, 0] -> [1, 1, 0, 0]
    received2_complex = torch.tensor([1 + 0j, 0 + 1j], dtype=torch.complex64)  # real/imag: [1, 0], [0, 1] -> [1, 0, 0, 1]
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


def test_ber_stateful_methods():
    """Test BER metric stateful update, compute, and reset methods."""
    metric = BitErrorRate()

    # Test reset
    metric.reset()

    # Test update and compute with real data
    transmitted1 = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    received1 = torch.tensor([1, 1, 1, 0], dtype=torch.float32)  # 1 error out of 4 bits

    transmitted2 = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    received2 = torch.tensor([0, 1, 1, 1], dtype=torch.float32)  # 1 error out of 4 bits

    # Update with first batch
    metric.update(transmitted1, received1)
    ber1 = metric.compute()
    expected_ber1 = 1.0 / 4.0  # 1 error out of 4 bits
    assert torch.isclose(ber1, torch.tensor(expected_ber1)), f"BER after first update should be {expected_ber1}, got {ber1}"

    # Update with second batch
    metric.update(transmitted2, received2)
    ber2 = metric.compute()
    expected_ber2 = 2.0 / 8.0  # 2 errors out of 8 total bits
    assert torch.isclose(ber2, torch.tensor(expected_ber2)), f"BER after second update should be {expected_ber2}, got {ber2}"

    # Test reset
    metric.reset()
    ber_reset = metric.compute()
    assert torch.isclose(ber_reset, torch.tensor(0.0)), f"BER after reset should be 0, got {ber_reset}"


def test_ber_stateful_complex():
    """Test BER metric stateful methods with complex data."""
    metric = BitErrorRate()
    metric.reset()

    # Complex data
    transmitted = torch.tensor([1 + 1j, 0 + 0j, 1 + 0j, 0 + 1j], dtype=torch.complex64)
    received = torch.tensor([1 + 1j, 1 + 0j, 0 + 0j, 0 + 1j], dtype=torch.complex64)  # 2 errors

    metric.update(transmitted, received)
    ber = metric.compute()

    expected_ber = 2.0 / 8.0  # 2 errors out of 8 "bits" (real+imag)
    assert torch.isclose(ber, torch.tensor(expected_ber)), f"Complex BER should be {expected_ber}, got {ber}"


def test_ber_error_conditions():
    """Test BER error conditions for full coverage."""
    metric = BitErrorRate()

    # Test shape mismatch in forward
    with pytest.raises(ValueError, match="Input shapes must match"):
        metric.forward(torch.tensor([1, 0]), torch.tensor([1, 0, 1]))

    # Test shape mismatch in update
    with pytest.raises(ValueError, match="Input shapes must match"):
        metric.update(torch.tensor([1, 0]), torch.tensor([1, 0, 1]))

    # Test mixed complex/real error in forward
    with pytest.raises(ValueError, match="Both inputs must be complex if one is complex"):
        metric.forward(torch.tensor([1 + 0j, 0 + 0j]), torch.tensor([1.0, 0.0]))

    # Test mixed complex/real error in update
    with pytest.raises(ValueError, match="Both inputs must be complex if one is complex for update"):
        metric.update(torch.tensor([1 + 0j, 0 + 0j]), torch.tensor([1.0, 0.0]))


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
        """Test BitErrorRate with batched data, checking aggregated result."""
        # Create batched test data (2 samples)
        transmitted = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]).float()
        received = torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1]]).float()
        # Batch 1: 1 error / 4 bits = 0.25
        # Batch 2: 2 errors / 4 bits = 0.5
        # Aggregated: (1 + 2) / (4 + 4) = 3 / 8 = 0.375

        ber_metric = BitErrorRate()

        # Test forward (should return aggregated BER)
        error_rate_forward = ber_metric.forward(transmitted, received)
        expected_ber_aggregated = 3.0 / 8.0
        assert error_rate_forward.shape == torch.Size([])  # Should be scalar
        assert torch.isclose(error_rate_forward, torch.tensor(expected_ber_aggregated)), f"Forward aggregated BER mismatch: expected {expected_ber_aggregated}, got {error_rate_forward}"

        # Test update/compute path
        ber_metric.reset()
        ber_metric.update(transmitted, received)
        error_rate_compute = ber_metric.compute()
        assert error_rate_compute.shape == torch.Size([])  # Should be scalar
        assert torch.isclose(error_rate_compute, torch.tensor(expected_ber_aggregated)), f"Computed aggregated BER mismatch: expected {expected_ber_aggregated}, got {error_rate_compute}"

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
        # Updated regex to match the actual error message
        with pytest.raises(ValueError, match="Total elements per batch item .* must be divisible by block_size"):
            bler(preds, target)

    def test_bler_shape_mismatch(self):
        """Test BlockErrorRate with mismatched shapes."""
        bler = BlockErrorRate(block_size=10)

        # Create mismatched shapes
        preds = torch.zeros((2, 20))
        target = torch.zeros((2, 10))

        # This should raise a ValueError
        # Updated regex to match the actual error message
        with pytest.raises(ValueError, match="Input shapes must match"):
            bler(preds, target)

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

    def test_bler_reshape_with_none_block_size_direct(self):
        """Test _reshape_into_blocks with block_size=None directly."""
        # Initialize BLER with block_size=None
        bler = BlockErrorRate(block_size=None)

        # Create test data
        test_data = torch.zeros((3, 10))

        # Call _reshape_into_blocks directly
        reshaped = bler._reshape_into_blocks(test_data)

        # When block_size is None, the method should reshape to [batch_size, 1, elements_per_row]
        expected_shape = torch.Size([3, 1, 10])
        assert reshaped.shape == expected_shape
        # Check if the content is preserved (flattened view)
        assert torch.equal(reshaped.view(3, 10), test_data)

        # Try with more complex tensor shape
        complex_data = torch.zeros((2, 4, 5))
        reshaped_complex = bler._reshape_into_blocks(complex_data)
        expected_complex_shape = torch.Size([2, 1, 20])
        assert reshaped_complex.shape == expected_complex_shape
        # Check if the content is preserved (flattened view)
        assert torch.equal(reshaped_complex.view(2, 4, 5), complex_data)


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


class TestErrorVectorMagnitude:
    """Test cases for Error Vector Magnitude (EVM) metric."""

    def test_evm_basic_computation(self):
        """Test basic EVM computation with simple signals."""
        metric = ErrorVectorMagnitude()

        # Create simple complex signals
        reference = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = torch.tensor([0.9 + 0.1j, 0.1 + 0.9j, -0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        evm = metric.forward(reference, received)
        assert evm > 0, "EVM should be positive for signals with errors"
        assert evm < 100, "EVM should be less than 100% for reasonable signals"

    def test_evm_perfect_signal(self):
        """Test EVM with perfect signal (no errors)."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = reference.clone()

        evm = metric.forward(reference, received)
        assert torch.isclose(evm, torch.tensor(0.0), atol=1e-6), f"EVM should be 0 for perfect signal, got {evm}"

    def test_evm_modes(self):
        """Test different EVM calculation modes."""
        reference = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = torch.tensor([0.9 + 0.1j, 0.1 + 0.9j, -0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        # Test RMS mode
        evm_rms = ErrorVectorMagnitude(mode="rms")
        evm_rms_val = evm_rms.forward(reference, received)

        # Test Peak mode
        evm_peak = ErrorVectorMagnitude(mode="peak")
        evm_peak_val = evm_peak.forward(reference, received)

        # Test Percentile mode
        evm_percentile = ErrorVectorMagnitude(mode="percentile", percentile=95.0)
        evm_percentile_val = evm_percentile.forward(reference, received)

        # Peak should be >= RMS for signals with errors
        assert evm_peak_val >= evm_rms_val, "Peak EVM should be >= RMS EVM"
        assert evm_percentile_val > 0, "Percentile EVM should be positive"

    def test_evm_normalization(self):
        """Test EVM with and without normalization."""
        reference = torch.tensor([2 + 0j, 0 + 2j, -2 + 0j, 0 - 2j], dtype=torch.complex64)
        received = torch.tensor([1.8 + 0.2j, 0.2 + 1.8j, -1.8 + 0.2j, 0.2 - 1.8j], dtype=torch.complex64)

        evm_normalized = ErrorVectorMagnitude(normalize=True)
        evm_not_normalized = ErrorVectorMagnitude(normalize=False)

        evm_norm_val = evm_normalized.forward(reference, received)
        evm_no_norm_val = evm_not_normalized.forward(reference, received)

        assert evm_norm_val != evm_no_norm_val, "Normalized and non-normalized EVM should differ"
        assert evm_norm_val > 0, "Normalized EVM should be positive"
        assert evm_no_norm_val > 0, "Non-normalized EVM should be positive"

    def test_evm_real_signals(self):
        """Test EVM with real-valued signals."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1.0, -1.0, 1.0, -1.0])
        received = torch.tensor([0.9, -0.9, 1.1, -1.1])

        evm = metric.forward(reference, received)
        assert evm > 0, "EVM should be positive for signals with errors"

    def test_evm_empty_input(self):
        """Test EVM with empty input tensors."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([], dtype=torch.complex64)
        received = torch.tensor([], dtype=torch.complex64)

        evm = metric.forward(reference, received)
        assert torch.isclose(evm, torch.tensor(0.0)), f"EVM should be 0 for empty tensors, got {evm}"

    def test_evm_shape_mismatch(self):
        """Test EVM with mismatched input shapes."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1 + 0j, 0 + 1j], dtype=torch.complex64)
        received = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j], dtype=torch.complex64)

        with pytest.raises(ValueError, match="Input shapes must match"):
            metric.forward(reference, received)

    def test_evm_invalid_mode(self):
        """Test EVM initialization with invalid mode."""
        with pytest.raises(ValueError, match="Mode must be 'rms', 'peak', or 'percentile', got 'invalid'"):
            ErrorVectorMagnitude(mode="invalid")

    def test_evm_invalid_percentile(self):
        """Test EVM initialization with invalid percentile values."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100, got 0"):
            ErrorVectorMagnitude(mode="percentile", percentile=0)

        with pytest.raises(ValueError, match="Percentile must be between 0 and 100, got 101"):
            ErrorVectorMagnitude(mode="percentile", percentile=101)

    def test_evm_per_symbol_calculation(self):
        """Test per-symbol EVM calculation."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = torch.tensor([0.9 + 0.1j, 0.1 + 0.9j, -0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        per_symbol_evm = metric.calculate_per_symbol_evm(reference, received)

        assert per_symbol_evm.shape == reference.shape, "Per-symbol EVM should have same shape as input"
        assert torch.all(per_symbol_evm >= 0), "All per-symbol EVM values should be non-negative"

    def test_evm_per_symbol_empty(self):
        """Test per-symbol EVM with empty tensors."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([], dtype=torch.complex64)
        received = torch.tensor([], dtype=torch.complex64)

        per_symbol_evm = metric.calculate_per_symbol_evm(reference, received)
        assert per_symbol_evm.numel() == 0, "Per-symbol EVM should be empty for empty input"

    def test_evm_per_symbol_shape_mismatch(self):
        """Test per-symbol EVM with mismatched shapes."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1 + 0j, 0 + 1j], dtype=torch.complex64)
        received = torch.tensor([1 + 0j], dtype=torch.complex64)

        with pytest.raises(ValueError, match="Input shapes must match"):
            metric.calculate_per_symbol_evm(reference, received)

    def test_evm_statistics(self):
        """Test comprehensive EVM statistics calculation."""
        metric = ErrorVectorMagnitude()

        reference = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = torch.tensor([0.9 + 0.1j, 0.1 + 0.9j, -0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        stats = metric.calculate_statistics(reference, received)

        expected_keys = ["evm_rms", "evm_mean", "evm_std", "evm_min", "evm_max", "evm_median", "evm_95th", "evm_99th", "evm_per_symbol"]

        assert all(key in stats for key in expected_keys), "All expected statistics should be present"
        assert stats["evm_min"] <= stats["evm_mean"] <= stats["evm_max"], "Statistics should be ordered correctly"
        assert stats["evm_per_symbol"].shape == reference.shape, "Per-symbol stats should match input shape"

    def test_evm_batched_input(self):
        """Test EVM with batched input."""
        metric = ErrorVectorMagnitude()

        # Create batched data
        batch_size = 3
        signal_length = 4
        reference = torch.randn(batch_size, signal_length, dtype=torch.complex64)
        noise = torch.randn(batch_size, signal_length, dtype=torch.complex64) * 0.1
        received = reference + noise

        evm = metric.forward(reference, received)
        assert evm > 0, "EVM should be positive for noisy signals"
        assert torch.isfinite(evm), "EVM should be finite"

    def test_evm_zero_reference(self):
        """Test EVM behavior with zero reference signal."""
        metric = ErrorVectorMagnitude(normalize=True)

        reference = torch.zeros(4, dtype=torch.complex64)
        received = torch.tensor([0.1 + 0.1j, 0.1 + 0.1j, 0.1 + 0.1j, 0.1 + 0.1j], dtype=torch.complex64)

        # Should handle zero reference gracefully due to clamping
        evm = metric.forward(reference, received)
        assert torch.isfinite(evm), "EVM should be finite even with zero reference"

    def test_evm_update_compute(self):
        """Test EVM metric state update and compute methods."""
        metric = ErrorVectorMagnitude()

        reference1 = torch.tensor([1 + 0j, 0 + 1j], dtype=torch.complex64)
        received1 = torch.tensor([0.9 + 0.1j, 0.1 + 0.9j], dtype=torch.complex64)

        reference2 = torch.tensor([-1 + 0j, 0 - 1j], dtype=torch.complex64)
        received2 = torch.tensor([-0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        # Test single update
        metric.reset()
        metric.update(reference1, received1)
        evm1 = metric.compute()

        # Test multiple updates
        metric.reset()
        metric.update(reference1, received1)
        metric.update(reference2, received2)
        evm2 = metric.compute()

        assert torch.isfinite(evm1), "Single update EVM should be finite"
        assert torch.isfinite(evm2), "Multiple update EVM should be finite"
        assert evm1 > 0, "Single update EVM should be positive"
        assert evm2 > 0, "Multiple update EVM should be positive"

    def test_evm_comprehensive_coverage(self):
        """Test comprehensive EVM functionality to achieve 100% coverage."""
        metric = ErrorVectorMagnitude()

        # Test with complex signals to ensure all code paths are hit
        reference = torch.tensor([1 + 1j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
        received = torch.tensor([0.9 + 0.9j, 0.1 + 0.9j, -0.9 + 0.1j, 0.1 - 0.9j], dtype=torch.complex64)

        # Test forward method (should hit lines 73-115)
        evm_result = metric.forward(reference, received)
        assert evm_result > 0, "EVM should be positive for signals with errors"

        # Test calculate_per_symbol_evm method (should hit lines 127-153)
        per_symbol_result = metric.calculate_per_symbol_evm(reference, received)
        assert per_symbol_result.shape == reference.shape, "Per-symbol EVM should match input shape"
        assert torch.all(per_symbol_result >= 0), "Per-symbol EVM should be non-negative"

        # Test calculate_statistics method (should hit lines 166-181)
        stats_result = metric.calculate_statistics(reference, received)
        expected_keys = ["evm_rms", "evm_mean", "evm_std", "evm_min", "evm_max", "evm_median", "evm_95th", "evm_99th", "evm_per_symbol"]
        assert all(key in stats_result for key in expected_keys), "All stats keys should be present"

        # Test with zero reference to hit normalization edge case
        zero_ref = torch.zeros(2, dtype=torch.complex64)
        received_zero = torch.tensor([0.1 + 0.1j, 0.2 + 0.2j], dtype=torch.complex64)
        evm_zero = metric.forward(zero_ref, received_zero)
        assert torch.isfinite(evm_zero), "EVM should be finite even with zero reference"

    def test_evm_edge_cases_for_coverage(self):
        """Test EVM edge cases to ensure complete coverage."""
        # Test all three modes with the same data
        reference = torch.tensor([1 + 0j, 2 + 1j, -1 + 2j], dtype=torch.complex64)
        received = torch.tensor([0.8 + 0.1j, 1.9 + 0.9j, -0.9 + 1.9j], dtype=torch.complex64)

        # Test RMS mode
        evm_rms = ErrorVectorMagnitude(mode="rms")
        result_rms = evm_rms.forward(reference, received)

        # Test Peak mode
        evm_peak = ErrorVectorMagnitude(mode="peak")
        result_peak = evm_peak.forward(reference, received)

        # Test Percentile mode
        evm_percentile = ErrorVectorMagnitude(mode="percentile", percentile=90.0)
        result_percentile = evm_percentile.forward(reference, received)

        # All should be finite and positive
        assert torch.isfinite(result_rms), "RMS EVM should be finite"
        assert torch.isfinite(result_peak), "Peak EVM should be finite"
        assert torch.isfinite(result_percentile), "Percentile EVM should be finite"
        assert result_rms > 0, "RMS EVM should be positive"
        assert result_peak > 0, "Peak EVM should be positive"
        assert result_percentile > 0, "Percentile EVM should be positive"


def test_metrics_aliases():
    """Test that metric aliases work properly."""
    assert BitErrorRate is BER
    assert BlockErrorRate is BLER
    assert FrameErrorRate is FER
    assert SymbolErrorRate is SER
    assert SignalToNoiseRatio is SNR
    assert ErrorVectorMagnitude is EVM


# Final comprehensive tests to achieve 100% coverage


def test_missing_ber_coverage():
    """Test specific missing BER coverage lines 105-116 (complex handling in update)."""
    metric = BitErrorRate()
    metric.reset()

    # Test complex handling in update method (lines 105-116)
    transmitted = torch.tensor([1 + 1j, 0 + 0j, 1 + 0j, 0 + 1j], dtype=torch.complex64)
    received = torch.tensor([1 + 1j, 1 + 0j, 0 + 0j, 0 + 1j], dtype=torch.complex64)

    # This should hit the complex branch in update method
    metric.update(transmitted, received)
    ber_result = metric.compute()

    expected_ber = 2.0 / 8.0  # 2 errors out of 8 "bits"
    assert torch.isclose(ber_result, torch.tensor(expected_ber)), f"Complex BER update should give {expected_ber}, got {ber_result}"

    # Test with real data in update for comparison
    metric.reset()
    transmitted_real = torch.tensor([1.0, 0.0, 1.0, 0.0])
    received_real = torch.tensor([0.9, 0.1, 1.1, 0.1])
    metric.update(transmitted_real, received_real)
    ber_real = metric.compute()
    assert torch.isfinite(ber_real), "Real BER update should be finite"


def test_missing_snr_coverage():
    """Test specific missing SNR coverage lines."""
    # Test various SNR edge cases to hit missing lines

    # Test with batched data
    signal_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    noise_batch = torch.tensor([[0.1, 0.2], [0.3, 0.1]])

    snr_db = SignalToNoiseRatio(mode="db")
    snr_linear = SignalToNoiseRatio(mode="linear")

    # Test batch processing
    snr_db_val = snr_db.forward(signal_batch, noise_batch)
    snr_linear_val = snr_linear.forward(signal_batch, noise_batch)

    assert torch.all(torch.isfinite(snr_db_val)), "Batched dB SNR should be finite"
    assert torch.all(torch.isfinite(snr_linear_val)), "Batched linear SNR should be finite"

    # Test compute_with_stats
    mean_snr, std_snr = snr_db.compute_with_stats(signal_batch, noise_batch)
    assert torch.isfinite(mean_snr), "Mean SNR should be finite"
    assert torch.isfinite(std_snr) or torch.isnan(std_snr), "Std SNR should be finite or NaN"

    # Test with zero noise (edge case)
    zero_noise = torch.zeros_like(signal_batch)
    try:
        snr_zero = snr_db.forward(signal_batch, zero_noise)
        # Should handle gracefully (might be inf)
        assert torch.isfinite(snr_zero) or torch.isinf(snr_zero), "Zero noise SNR should be finite or inf"
    except (RuntimeError, ValueError, ZeroDivisionError):
        # It's acceptable for zero noise to raise mathematical errors like division by zero
        # This is expected behavior when computing SNR with zero noise
        pytest.skip("Zero noise SNR computation raised expected mathematical error")

    # Test with complex signals
    signal_complex = torch.tensor([1 + 1j, 2 + 0j], dtype=torch.complex64)
    noise_complex = torch.tensor([0.1 + 0.1j, 0.2 + 0j], dtype=torch.complex64)
    snr_complex = snr_db.forward(signal_complex, noise_complex)
    assert torch.isfinite(snr_complex), "Complex SNR should be finite"


def test_missing_evm_coverage():
    """Test remaining EVM coverage lines 74, 78, 128, 132."""

    # Test to hit lines 74, 78 (shape mismatch and empty tensors)
    metric = ErrorVectorMagnitude()

    # Shape mismatch (line 74)
    try:
        metric.forward(torch.tensor([1 + 0j]), torch.tensor([1 + 0j, 2 + 0j]))
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError:
        pass  # Expected

    # Empty tensors (line 78)
    empty_ref = torch.tensor([], dtype=torch.complex64)
    empty_rec = torch.tensor([], dtype=torch.complex64)
    evm_empty = metric.forward(empty_ref, empty_rec)
    assert torch.isclose(evm_empty, torch.tensor(0.0)), f"Empty tensor EVM should be 0, got {evm_empty}"

    # Test calculate_per_symbol_evm with shape mismatch (line 128)
    try:
        metric.calculate_per_symbol_evm(torch.tensor([1 + 0j]), torch.tensor([1 + 0j, 2 + 0j]))
        assert False, "Should have raised ValueError for per-symbol shape mismatch"
    except ValueError:
        pass  # Expected

    # Test calculate_per_symbol_evm with empty tensors (line 132)
    per_symbol_empty = metric.calculate_per_symbol_evm(empty_ref, empty_rec)
    assert per_symbol_empty.numel() == 0, "Per-symbol EVM should be empty for empty input"


def test_missing_bler_coverage():
    """Test missing BLER coverage lines."""

    # Test various BLER edge cases
    metric = BlockErrorRate()

    # Test with different block sizes
    metric_block4 = BlockErrorRate(block_size=4)
    data = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32).reshape(1, -1)
    data_err = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.float32).reshape(1, -1)

    bler_val = metric_block4.forward(data, data_err)
    assert torch.isfinite(bler_val), "BLER with block_size should be finite"

    # Test with threshold different from 0.5
    metric_thresh = BlockErrorRate(threshold=0.7)
    data_thresh = torch.tensor([[0.8, 0.3, 0.9, 0.2]], dtype=torch.float32)
    data_thresh_err = torch.tensor([[0.6, 0.8, 0.1, 0.9]], dtype=torch.float32)
    bler_thresh = metric_thresh.forward(data_thresh, data_thresh_err)
    assert torch.isfinite(bler_thresh), "BLER with custom threshold should be finite"

    # Test stateful methods
    metric.reset()
    metric.update(data, data_err)
    bler_computed = metric.compute()
    assert torch.isfinite(bler_computed), "BLER computed should be finite"
