import pytest
import torch
from kaira.metrics.signal.bler import BlockErrorRate

@pytest.fixture
def binary_data():
    torch.manual_seed(42)
    true_bits = torch.randint(0, 2, (1, 1000))
    error_mask = torch.rand(1, 1000) < 0.05  # 5% error rate
    received_bits = torch.logical_xor(true_bits, error_mask).int()
    return true_bits, received_bits

@pytest.mark.parametrize("block_size", [10, 50, 100])
def test_bler_computation(binary_data, block_size):
    true_bits, received_bits = binary_data
    n_blocks = true_bits.size(1) // block_size
    usable_bits = n_blocks * block_size

    true_blocks = true_bits[:, :usable_bits].reshape(1, -1, block_size)
    received_blocks = received_bits[:, :usable_bits].reshape(1, -1, block_size)

    bler = BlockErrorRate()
    bler_value = bler(received_blocks, true_blocks)
    assert isinstance(bler_value, torch.Tensor)
    assert bler_value.ndim == 0
    assert 0 <= bler_value <= 1

    bler.reset()
    bler.update(received_blocks, true_blocks)
    bler_mean, bler_std = bler.compute()
    assert isinstance(bler_mean, torch.Tensor)
    assert isinstance(bler_std, torch.Tensor)

def test_block_error_edge_cases():
    bler = BlockErrorRate()

    perfect_blocks = torch.zeros((1, 10, 8))
    bler_value = bler(perfect_blocks, perfect_blocks)
    assert bler_value == 0.0

    corrupted_blocks = torch.ones((1, 10, 8))
    clean_blocks = torch.zeros((1, 10, 8))
    bler_value = bler(corrupted_blocks, clean_blocks)
    assert bler_value == 1.0

@pytest.mark.parametrize("block_size,error_pattern", [
    (10, [0]),
    (20, [1, 3]),
    (50, [-1]),
    (100, [0, -1])
])
def test_bler_specific_patterns(block_size, error_pattern):
    n_blocks = 10
    n_bits = block_size * n_blocks
    true_bits = torch.zeros((1, n_bits))
    received_bits = true_bits.clone()

    for block_idx in error_pattern:
        start_idx = block_idx * block_size
        received_bits[0, start_idx] = 1

    blocks = true_bits.reshape(1, n_blocks, block_size)
    received_blocks = received_bits.reshape(1, n_blocks, block_size)

    bler = BlockErrorRate()
    bler_value = bler(received_blocks, blocks)
    expected_bler = len(error_pattern) / n_blocks
    assert abs(bler_value.item() - expected_bler) < 1e-6
