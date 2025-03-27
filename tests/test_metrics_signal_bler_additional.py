import pytest
import torch

from kaira.metrics.signal.bler import BlockErrorRate


@pytest.fixture
def random_binary_data():
    """Fixture providing random binary data for testing."""
    torch.manual_seed(42)
    n_bits = 1000

    # Create true bits
    true_bits = torch.randint(0, 2, (2, n_bits)).float()

    # Create received bits with some errors
    error_mask = torch.rand(2, n_bits) < 0.1  # 10% bit error rate
    received_bits = torch.logical_xor(true_bits, error_mask).float()

    return true_bits, received_bits


def test_bler_with_small_batch():
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


def test_bler_reset():
    """Test that reset clears accumulated statistics."""
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


def test_bler_with_threshold():
    """Test BlockErrorRate with different thresholds."""
    # Create data with soft values (probabilities)
    soft_preds = torch.tensor([[0.2, 0.7, 0.3, 0.8, 0.1, 0.6], [0.9, 0.4, 0.2, 0.3, 0.7, 0.8]])
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

    # With threshold 0.5
    bler_default = BlockErrorRate(block_size=3, threshold=0.5)
    bler_default.update(soft_preds, target)
    result_default = bler_default.compute()

    # With threshold 0.6
    bler_higher = BlockErrorRate(block_size=3, threshold=0.6)
    bler_higher.update(soft_preds, target)
    result_higher = bler_higher.compute()

    # Results should be different due to different thresholds
    assert not torch.isclose(result_default, result_higher)


def test_bler_with_different_batch_sizes(random_binary_data):
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
