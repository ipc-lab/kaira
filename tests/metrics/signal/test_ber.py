import torch
import pytest

from kaira.metrics.signal.ber import BitErrorRate

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
    expected_ber_batch = torch.tensor([0.5, 0.5])

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

