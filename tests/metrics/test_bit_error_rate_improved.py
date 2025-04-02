"""Improved tests for Bit Error Rate (BER) metrics."""
import pytest
import torch
import numpy as np
from kaira.metrics.signal.ber import BitErrorRate


def test_ber_initialization():
    """Test BER metric initialization with default and custom parameters."""
    # Default initialization
    ber_default = BitErrorRate()
    assert ber_default.name == "BER"
    assert ber_default.threshold == 0.5
    
    # Custom threshold and name
    ber_custom = BitErrorRate(threshold=0.7, name="CustomBER")
    assert ber_custom.name == "CustomBER"
    assert ber_custom.threshold == 0.7
    
    # Verify counters are initialized to zero
    assert ber_default.total_bits.item() == 0
    assert ber_default.error_bits.item() == 0


def test_ber_forward():
    """Test BER forward computation with binary data."""
    ber_metric = BitErrorRate()
    
    # Perfect transmission
    transmitted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    received = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    ber_value = ber_metric(transmitted, received)
    assert ber_value.item() == pytest.approx(0.0)
    
    # One error in four bits
    received_with_error = torch.tensor([0.0, 0.0, 0.0, 1.0])  # Second bit flipped
    ber_value = ber_metric(transmitted, received_with_error)
    assert ber_value.item() == pytest.approx(0.25)
    
    # All errors
    received_all_errors = torch.tensor([1.0, 0.0, 1.0, 0.0])  # All bits flipped
    ber_value = ber_metric(transmitted, received_all_errors)
    assert ber_value.item() == pytest.approx(1.0)


def test_ber_with_soft_values():
    """Test BER calculation with soft decision values."""
    ber_metric = BitErrorRate(threshold=0.5)
    
    # Transmitted bits are hard (0 or 1)
    transmitted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    # Received bits are soft values (probabilities/LLRs)
    # Values > 0.5 are interpreted as 1, otherwise as 0
    received_soft = torch.tensor([0.2, 0.6, 0.3, 0.9])
    
    ber_value = ber_metric(transmitted, received_soft)
    assert ber_value.item() == pytest.approx(0.0)  # All decisions are correct
    
    # Some errors in soft decisions
    received_soft_with_errors = torch.tensor([0.6, 0.4, 0.7, 0.8])
    # First bit: transmitted=0, received>0.5 (error)
    # Second bit: transmitted=1, received<0.5 (error)
    # Third bit: transmitted=0, received>0.5 (error)
    # Fourth bit: transmitted=1, received>0.5 (correct)
    ber_value = ber_metric(transmitted, received_soft_with_errors)
    assert ber_value.item() == pytest.approx(0.75)  # 3 out of 4 bits are wrong


def test_ber_custom_threshold():
    """Test BER with a custom threshold value."""
    # Set a different threshold for detecting 1s
    ber_metric = BitErrorRate(threshold=0.7)
    
    transmitted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    
    # Values > 0.7 are interpreted as 1s
    received = torch.tensor([0.2, 0.6, 0.3, 0.8])
    
    # Expected interpretations:
    # First bit: 0.2 < 0.7 -> 0 (correct)
    # Second bit: 0.6 < 0.7 -> 0 (error)
    # Third bit: 0.3 < 0.7 -> 0 (correct)
    # Fourth bit: 0.8 > 0.7 -> 1 (correct)
    
    ber_value = ber_metric(transmitted, received)
    assert ber_value.item() == pytest.approx(0.25)  # 1 out of 4 bits is wrong


def test_ber_with_batched_data():
    """Test BER calculation with batched data."""
    ber_metric = BitErrorRate()
    
    # Create a batch of transmitted data (3 batches, 4 bits each)
    transmitted = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0]
    ])
    
    # Create a batch of received data with some errors
    received = torch.tensor([
        [0.0, 1.0, 1.0, 1.0],  # 1 error
        [1.0, 0.0, 0.0, 0.0],  # 1 error
        [0.0, 0.0, 1.0, 0.0]   # 1 error
    ])
    
    # Calculate BER
    ber_value = ber_metric(transmitted, received)
    
    # Expected: 3 errors out of 12 bits total
    assert ber_value.item() == pytest.approx(0.25)


def test_ber_update_and_compute():
    """Test cumulative BER calculation using update and compute methods."""
    ber_metric = BitErrorRate()
    
    # First batch
    transmitted1 = torch.tensor([0.0, 1.0, 0.0, 1.0])
    received1 = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 1 error
    
    ber_metric.update(transmitted1, received1)
    
    # Verify internal state
    assert ber_metric.total_bits.item() == 4
    assert ber_metric.error_bits.item() == 1
    
    # Check computed BER
    ber_value = ber_metric.compute()
    assert ber_value.item() == pytest.approx(0.25)
    
    # Second batch
    transmitted2 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    received2 = torch.tensor([1.0, 0.0, 0.0, 1.0])  # 2 errors
    
    ber_metric.update(transmitted2, received2)
    
    # Verify internal state
    assert ber_metric.total_bits.item() == 8
    assert ber_metric.error_bits.item() == 3
    
    # Check computed BER
    ber_value = ber_metric.compute()
    assert ber_value.item() == pytest.approx(0.375)  # 3 errors out of 8 bits


def test_ber_reset():
    """Test BER metric reset functionality."""
    ber_metric = BitErrorRate()
    
    # Update with some data
    transmitted = torch.tensor([0.0, 1.0, 0.0, 1.0])
    received = torch.tensor([1.0, 1.0, 0.0, 0.0])  # 2 errors
    
    ber_metric.update(transmitted, received)
    
    # Verify state before reset
    assert ber_metric.total_bits.item() == 4
    assert ber_metric.error_bits.item() == 2
    
    # Reset the metric
    ber_metric.reset()
    
    # Verify state after reset
    assert ber_metric.total_bits.item() == 0
    assert ber_metric.error_bits.item() == 0
    
    # Verify computed value after reset
    ber_value = ber_metric.compute()
    assert ber_value.item() == pytest.approx(0.0)


def test_ber_multidimensional_input():
    """Test BER calculation with multi-dimensional input."""
    ber_metric = BitErrorRate()
    
    # Create 3D tensors (batch, height, width) - e.g., batch of binary images
    transmitted = torch.zeros((2, 3, 3))
    transmitted[0, 1, 1] = 1.0
    transmitted[1, 0, 2] = 1.0
    transmitted[1, 2, 0] = 1.0
    
    # Copy and introduce some errors
    received = transmitted.clone()
    received[0, 1, 1] = 0.0  # Flip one bit
    received[1, 1, 1] = 1.0  # Flip another bit
    
    ber_value = ber_metric(transmitted, received)
    
    # Expected: 2 errors out of 18 bits total (2×3×3)
    assert ber_value.item() == pytest.approx(2/18)


def test_ber_with_empty_tensors():
    """Test BER calculation with empty tensors."""
    ber_metric = BitErrorRate()
    
    # Create empty tensors
    transmitted = torch.tensor([])
    received = torch.tensor([])
    
    # Forward pass with empty tensors should return nan or 0
    # (implementation dependent, but shouldn't crash)
    ber_value = ber_metric(transmitted, received)
    
    # Either NaN or 0 are acceptable results for empty inputs
    assert torch.isnan(ber_value) or ber_value.item() == 0.0


def test_ber_accumulation():
    """Test BER accumulation over multiple updates with different batch sizes."""
    ber_metric = BitErrorRate()
    
    # First update: 4 bits, 1 error
    transmitted1 = torch.tensor([0.0, 1.0, 0.0, 1.0])
    received1 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    ber_metric.update(transmitted1, received1)
    
    # Second update: 8 bits, 2 errors
    transmitted2 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    received2 = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    ber_metric.update(transmitted2, received2)
    
    # Third update: 2 bits, 0 errors
    transmitted3 = torch.tensor([1.0, 0.0])
    received3 = torch.tensor([1.0, 0.0])
    ber_metric.update(transmitted3, received3)
    
    # Check final accumulated results
    assert ber_metric.total_bits.item() == 14  # 4 + 8 + 2
    assert ber_metric.error_bits.item() == 3   # 1 + 2 + 0
    
    ber_value = ber_metric.compute()
    assert ber_value.item() == pytest.approx(3/14)