"""Improved tests for Block Error Rate (BLER) metrics."""
import pytest
import torch
import numpy as np
from kaira.metrics.signal.bler import BlockErrorRate, BLER, FER, SER


def test_bler_initialization():
    """Test BLER metric initialization with default and custom parameters."""
    # Default initialization
    bler_default = BlockErrorRate()
    assert bler_default.name == "BLER"
    assert bler_default.block_size is None
    assert bler_default.threshold == 0.0
    assert bler_default.reduction == "mean"
    
    # Custom parameters
    bler_custom = BlockErrorRate(
        block_size=8,
        threshold=0.1,
        reduction="sum",
        name="CustomBLER"
    )
    assert bler_custom.name == "CustomBLER"
    assert bler_custom.block_size == 8
    assert bler_custom.threshold == 0.1
    assert bler_custom.reduction == "sum"
    
    # Verify counters are initialized to zero
    assert bler_default.total_blocks.item() == 0
    assert bler_default.error_blocks.item() == 0


def test_bler_reshape_into_blocks():
    """Test the internal _reshape_into_blocks method."""
    # Create a BLER metric with specified block size
    bler = BlockErrorRate(block_size=4)
    
    # 1D input [batch=1, sequence=8]
    x = torch.arange(8).unsqueeze(0)  # Shape: [1, 8]
    reshaped = bler._reshape_into_blocks(x)
    assert reshaped.shape == (1, 2, 4)  # [batch, num_blocks, block_size]
    
    # 2D input [batch=2, sequence=8]
    x = torch.arange(16).reshape(2, 8)  # Shape: [2, 8]
    reshaped = bler._reshape_into_blocks(x)
    assert reshaped.shape == (2, 2, 4)  # [batch, num_blocks, block_size]
    
    # 3D input [batch=2, height=2, width=4]
    x = torch.arange(16).reshape(2, 2, 4)  # Shape: [2, 2, 4]
    reshaped = bler._reshape_into_blocks(x)
    assert reshaped.shape == (2, 2, 4)  # [batch, num_blocks, block_size]


def test_bler_reshape_error():
    """Test error handling when input can't be evenly divided into blocks."""
    bler = BlockErrorRate(block_size=3)  # Block size of 3
    
    # Input with size not divisible by block_size
    x = torch.arange(10).unsqueeze(0)  # Shape: [1, 10]
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        bler._reshape_into_blocks(x)


def test_bler_forward_no_block_size():
    """Test BLER calculation when block_size is None (each row is a block)."""
    bler = BlockErrorRate(block_size=None)
    
    # Create test data - each row is a block
    preds = torch.tensor([
        [1, 1, 0, 0],  # Block 1: no errors
        [1, 0, 0, 0],  # Block 2: has error
        [0, 1, 1, 1],  # Block 3: has error
    ])
    
    targets = torch.tensor([
        [1, 1, 0, 0],  # Block 1
        [1, 1, 0, 0],  # Block 2
        [0, 0, 1, 1],  # Block 3
    ])
    
    # Calculate BLER
    bler_value = bler(preds, targets)
    
    # 2 out of 3 blocks have errors
    assert bler_value.item() == pytest.approx(2/3)


def test_bler_forward_with_block_size():
    """Test BLER calculation with specified block_size."""
    bler = BlockErrorRate(block_size=2)
    
    # Create test data
    preds = torch.tensor([
        [1, 1, 0, 0, 1, 1],  # 3 blocks of size 2
        [1, 0, 0, 0, 1, 1],  # 3 blocks of size 2
    ])
    
    targets = torch.tensor([
        [1, 1, 0, 0, 1, 0],  # Last block has error
        [1, 1, 0, 0, 1, 1],  # First block has error
    ])
    
    # Calculate BLER
    bler_value = bler(preds, targets)
    
    # 2 out of 6 blocks have errors
    assert bler_value.item() == pytest.approx(2/6)


def test_bler_forward_different_reductions():
    """Test BLER with different reduction methods."""
    # Test with 'mean' reduction
    bler_mean = BlockErrorRate(block_size=2, reduction="mean")
    
    # Test with 'sum' reduction
    bler_sum = BlockErrorRate(block_size=2, reduction="sum")
    
    # Test with 'none' reduction
    bler_none = BlockErrorRate(block_size=2, reduction="none")
    
    # Create test data with 2 batches, each with 2 blocks
    preds = torch.tensor([
        [1, 1, 0, 0],  # Batch 1: 2 blocks
        [1, 0, 0, 0],  # Batch 2: 2 blocks
    ])
    
    targets = torch.tensor([
        [1, 1, 0, 0],  # Batch 1: no errors
        [1, 1, 0, 0],  # Batch 2: first block has error
    ])
    
    # Calculate with different reductions
    bler_mean_value = bler_mean(preds, targets)
    bler_sum_value = bler_sum(preds, targets)
    bler_none_value = bler_none(preds, targets)
    
    # Check results
    assert bler_mean_value.item() == 0.25  # 1 out of 4 blocks have errors
    assert bler_sum_value.item() == 1.0  # 1 block with error
    assert bler_none_value.shape == (2, 2)  # [batch, num_blocks]
    assert torch.allclose(bler_none_value, torch.tensor([[0., 0.], [1., 0.]]))


def test_bler_with_threshold():
    """Test BLER calculation with non-zero threshold."""
    # Set threshold to 0.5
    bler = BlockErrorRate(threshold=0.5)
    
    # Create test data with small deviations
    preds = torch.tensor([
        [1.0, 1.2, 0.0, 0.1],  # Row 1: small deviations
        [1.0, 0.3, 0.0, 0.0],  # Row 2: deviation > 0.5
    ])
    
    targets = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],  # Row 1
        [1.0, 1.0, 0.0, 0.0],  # Row 2
    ])
    
    # Calculate BLER
    bler_value = bler(preds, targets)
    
    # Only the second row should be counted as an error
    assert bler_value.item() == 0.5


def test_bler_update_and_compute():
    """Test BLER accumulation using update and compute methods."""
    bler = BlockErrorRate(block_size=2)
    
    # First batch
    preds1 = torch.tensor([
        [1, 1, 0, 0],  # 2 blocks
        [1, 0, 0, 0],  # 2 blocks
    ])
    
    targets1 = torch.tensor([
        [1, 1, 0, 0],  # No errors
        [1, 1, 0, 0],  # First block has error
    ])
    
    bler.update(preds1, targets1)
    
    # Verify internal state
    assert bler.total_blocks.item() == 4  # 2 batches × 2 blocks
    assert bler.error_blocks.item() == 1  # 1 block with error
    
    # Check computed BLER
    bler_value = bler.compute()
    assert bler_value.item() == pytest.approx(0.25)  # 1/4
    
    # Second batch
    preds2 = torch.tensor([
        [1, 1, 0, 1],  # 2 blocks
    ])
    
    targets2 = torch.tensor([
        [1, 1, 0, 0],  # Second block has error
    ])
    
    bler.update(preds2, targets2)
    
    # Verify internal state
    assert bler.total_blocks.item() == 6  # 4 + 2 blocks
    assert bler.error_blocks.item() == 2  # 1 + 1 blocks with errors
    
    # Check computed BLER
    bler_value = bler.compute()
    assert bler_value.item() == pytest.approx(2/6)  # 2/6 = 1/3


def test_bler_reset():
    """Test BLER metric reset functionality."""
    bler = BlockErrorRate(block_size=2)
    
    # Update with some data
    preds = torch.tensor([
        [1, 1, 0, 1],  # 2 blocks
    ])
    
    targets = torch.tensor([
        [1, 1, 0, 0],  # Second block has error
    ])
    
    bler.update(preds, targets)
    
    # Verify state before reset
    assert bler.total_blocks.item() == 2
    assert bler.error_blocks.item() == 1
    
    # Reset the metric
    bler.reset()
    
    # Verify state after reset
    assert bler.total_blocks.item() == 0
    assert bler.error_blocks.item() == 0
    
    # Verify computed value after reset
    bler_value = bler.compute()
    assert bler_value.item() == 0.0


def test_bler_multidimensional_input():
    """Test BLER with multi-dimensional input."""
    bler = BlockErrorRate(block_size=4)
    
    # Create 3D tensors (batch_size=2, height=2, width=4)
    preds = torch.zeros((2, 2, 4))
    preds[0, 0, 0] = 1.0
    preds[1, 1, 2] = 1.0
    
    targets = torch.zeros((2, 2, 4))
    targets[0, 0, 0] = 1.0
    targets[1, 1, 3] = 1.0  # Different from prediction
    
    # Calculate BLER
    bler_value = bler(preds, targets)
    
    # 1 out of 4 blocks has error (each 3D tensor is reshaped into 2 blocks)
    assert bler_value.item() == 0.25


def test_bler_empty_tensor():
    """Test BLER with empty tensors."""
    bler = BlockErrorRate()
    
    # Empty tensors
    preds = torch.tensor([])
    targets = torch.tensor([])
    
    # Should not raise error, but result might be NaN or 0
    result = bler(preds, targets)
    assert torch.isnan(result) or result.item() == 0.0


def test_bler_shape_mismatch():
    """Test BLER with shape mismatch between predictions and targets."""
    bler = BlockErrorRate()
    
    # Tensors with different shapes
    preds = torch.zeros((2, 4))
    targets = torch.zeros((2, 3))
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        bler(preds, targets)


def test_bler_aliases():
    """Test that different BLER aliases work the same way."""
    # Create the same test data for all metrics
    preds = torch.tensor([
        [1, 1, 0, 1],  # 1 error in last position
    ])
    
    targets = torch.tensor([
        [1, 1, 0, 0],  # Reference
    ])
    
    # Test with different aliases
    bler_metric = BLER()
    fer_metric = FER()
    ser_metric = SER()
    
    # All should give the same result
    bler_value = bler_metric(preds, targets)
    fer_value = fer_metric(preds, targets)
    ser_value = ser_metric(preds, targets)
    
    # They should all be the same
    assert bler_value.item() == fer_value.item()
    assert fer_value.item() == ser_value.item()


def test_bler_custom_block_size():
    """Test BLER with different custom block sizes."""
    # Test with block_size=1 (should be equivalent to BER)
    bler_block1 = BlockErrorRate(block_size=1)
    
    # Test with block_size=2
    bler_block2 = BlockErrorRate(block_size=2)
    
    # Create test data
    preds = torch.tensor([
        [1, 0, 1, 0],
    ])
    
    targets = torch.tensor([
        [1, 1, 1, 0],
    ])
    
    # With block_size=1, each element is a block, so 1 out of 4 blocks has error
    bler_value1 = bler_block1(preds, targets)
    assert bler_value1.item() == 0.25
    
    # With block_size=2, we have 2 blocks, and 1 has an error
    bler_value2 = bler_block2(preds, targets)
    assert bler_value2.item() == 0.5