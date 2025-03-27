import pytest
import torch

from kaira.metrics.signal.bler import BlockErrorRate


@pytest.fixture
def binary_data():
    """Fixture providing test data with known errors."""
    torch.manual_seed(42)
    n_bits = 1000

    # Create true bits
    true_bits = torch.randint(0, 2, (1, n_bits)).float()

    # Create received bits with some errors
    error_mask = torch.rand(1, n_bits) < 0.1  # 10% bit error rate
    received_bits = torch.logical_xor(true_bits, error_mask).float()

    return true_bits, received_bits


@pytest.mark.parametrize("block_size", [10, 50, 100])
def test_bler_computation(binary_data, block_size):
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


def test_block_error_edge_cases():
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
def test_bler_specific_patterns(block_size, error_pattern):
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

    bler = BlockErrorRate()
    bler_value = bler(received_blocks, blocks)
    expected_bler = len(error_pattern) / n_blocks
    assert abs(bler_value.item() - expected_bler) < 1e-6
