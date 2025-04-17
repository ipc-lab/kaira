"""Tests for differentiable modulation operations."""

import torch

from kaira.modulations import (
    BPSKDemodulator,
    BPSKModulator,
    QAMDemodulator,
    QAMModulator,
    QPSKDemodulator,
    QPSKModulator,
)
from kaira.modulations.differentiable import (
    hard_decisions_with_straight_through,
    soft_bits_to_hard_symbols,
    soft_symbol_mapping,
)


class TestDifferentiableOperations:
    """Test suite for differentiable modulation operations."""

    def test_soft_symbol_mapping(self):
        """Test soft symbol mapping function."""
        # Create a simple constellation with 2 points
        constellation = torch.tensor([1 + 0j, -1 + 0j])
        bit_patterns = torch.tensor([[0.0], [1.0]])

        # Test with hard probabilities (0 and 1)
        soft_bits = torch.tensor([[0.0], [1.0]])
        symbols = soft_symbol_mapping(soft_bits, constellation, bit_patterns)
        assert symbols.shape == torch.Size([2])
        assert torch.isclose(symbols[0], torch.tensor(1 + 0j))
        assert torch.isclose(symbols[1], torch.tensor(-1 + 0j))

        # Test with soft probabilities
        soft_bits = torch.tensor([[0.3], [0.7]])
        symbols = soft_symbol_mapping(soft_bits, constellation, bit_patterns)
        assert symbols.shape == torch.Size([2])
        # Expected: 0.3 * (-1) + 0.7 * 1 = 0.4 for first symbol
        # Expected: 0.7 * (-1) + 0.3 * 1 = -0.4 for second symbol
        assert torch.isclose(symbols[0], torch.tensor(0.4 + 0j), atol=1e-6)
        assert torch.isclose(symbols[1], torch.tensor(-0.4 + 0j), atol=1e-6)

    def test_soft_bits_to_hard_symbols(self):
        """Test soft to hard symbol conversion with differentiability."""
        # Create a simple constellation with 2 points
        constellation = torch.tensor([1 + 0j, -1 + 0j])
        bit_patterns = torch.tensor([[0.0], [1.0]])

        # Test with different temperatures
        soft_bits = torch.tensor([[0.3], [0.7]])
        symbols_temp_1 = soft_bits_to_hard_symbols(soft_bits, constellation, bit_patterns, temp=1.0)
        symbols_temp_01 = soft_bits_to_hard_symbols(soft_bits, constellation, bit_patterns, temp=0.1)

        # Lower temperature should make decisions harder (closer to constellation points)
        assert torch.abs(symbols_temp_01[0] - constellation[0]) < torch.abs(symbols_temp_1[0] - constellation[0])
        assert torch.abs(symbols_temp_01[1] - constellation[1]) < torch.abs(symbols_temp_1[1] - constellation[1])

    def test_hard_decisions_with_straight_through(self):
        """Test hard decision with straight-through estimator."""
        # Create input requiring gradients
        soft_values = torch.tensor([0.3, 0.6, 0.8], requires_grad=True)

        # Apply hard decision with straight-through estimator
        hard_values = hard_decisions_with_straight_through(soft_values)

        # Check forward pass (hard decisions)
        assert hard_values.detach().tolist() == [0.0, 1.0, 1.0]

        # Check that gradients can flow through
        loss = hard_values.sum()
        loss.backward()
        assert soft_values.grad is not None
        # The gradient should be all ones since we're using straight-through
        assert torch.allclose(soft_values.grad, torch.ones_like(soft_values))


class TestDifferentiableModulators:
    """Test suite for differentiable modulators."""

    def test_bpsk_modulators_diff(self):
        """Test differentiable BPSK modulation."""
        modulator = BPSKModulator()

        # Create soft bits with gradients
        soft_bits = torch.tensor([0.1, 0.4, 0.6, 0.9], requires_grad=True)

        # Test forward_soft
        soft_symbols = modulator.forward_soft(soft_bits)

        # Verify shapes
        assert soft_symbols.shape == soft_bits.shape

        # Compute loss and check gradients
        loss = soft_symbols.real.sum()
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None
        # For BPSK the gradient should be -2 for all inputs
        assert torch.allclose(soft_bits.grad, torch.tensor([-2.0, -2.0, -2.0, -2.0]))

    def test_qpsk_modulators_diff(self):
        """Test differentiable QPSK modulation."""
        modulator = QPSKModulator()

        # Create soft bits with gradients (needs to be even number for QPSK)
        soft_bits = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], requires_grad=True)

        # Test forward_soft
        soft_symbols = modulator.forward_soft(soft_bits)

        # Verify shapes - QPSK has 2 bits per symbol
        assert soft_symbols.shape == torch.Size([3])

        # Compute loss and check gradients
        loss = soft_symbols.abs().sum()
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None

    def test_qam_modulators_diff(self):
        """Test differentiable QAM modulation."""
        modulator = QAMModulator(order=16)

        # Create soft bits with gradients (16-QAM has 4 bits per symbol)
        soft_bits = torch.rand(8, requires_grad=True)

        # Test forward_soft
        soft_symbols = modulator.forward_soft(soft_bits)

        # Verify shapes - 16-QAM has 4 bits per symbol
        assert soft_symbols.shape == torch.Size([2])

        # Compute loss and check gradients
        loss = soft_symbols.abs().sum()
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None


class TestDifferentiableDemodulators:
    """Test suite for differentiable demodulators."""

    def test_bpsk_demodulator_diff(self):
        """Test differentiable BPSK demodulation."""
        demodulator = BPSKDemodulator()

        # Create symbols with gradients
        symbols = torch.tensor([0.5 + 0j, -0.2 + 0j, 1.5 + 0j], requires_grad=True)

        # Test forward_soft with noise variance
        noise_var = 0.1
        soft_bits = demodulator.forward_soft(symbols, noise_var)

        # Verify shapes
        assert soft_bits.shape == symbols.shape

        # Verify values are between 0 and 1 (probabilities)
        assert (soft_bits >= 0).all() and (soft_bits <= 1).all()

        # Compute loss and check gradients
        loss = soft_bits.sum()
        loss.backward()

        # Verify gradients exist
        assert symbols.grad is not None

    def test_qpsk_demodulator_diff(self):
        """Test differentiable QPSK demodulation."""
        demodulator = QPSKDemodulator()

        # Create symbols with gradients
        symbols = torch.tensor([0.5 + 0.5j, -0.5 - 0.5j], requires_grad=True)

        # Test forward_soft with noise variance
        noise_var = 0.1
        soft_bits = demodulator.forward_soft(symbols, noise_var)

        # Verify shapes - QPSK has 2 bits per symbol
        assert soft_bits.shape == torch.Size([4])

        # Verify values are between 0 and 1 (probabilities)
        assert (soft_bits >= 0).all() and (soft_bits <= 1).all()

        # Compute loss and check gradients
        loss = soft_bits.sum()
        loss.backward()

        # Verify gradients exist
        assert symbols.grad is not None

    def test_qam_demodulator_diff(self):
        """Test differentiable QAM demodulation."""
        demodulator = QAMDemodulator(order=16)

        # Create symbols with gradients
        symbols = torch.tensor([1.0 + 1.0j, -1.0 - 1.0j], requires_grad=True)

        # Test forward_soft with noise variance
        noise_var = 0.1
        soft_bits = demodulator.forward_soft(symbols, noise_var)

        # Verify shapes - 16-QAM has 4 bits per symbol
        assert soft_bits.shape == torch.Size([8])

        # Verify values are between 0 and 1 (probabilities)
        assert (soft_bits >= 0).all() and (soft_bits <= 1).all()

        # Compute loss and check gradients
        loss = soft_bits.sum()
        loss.backward()

        # Verify gradients exist
        assert symbols.grad is not None


class TestEndToEndDifferentiability:
    """Test end-to-end differentiability of modulation and demodulation."""

    def test_bpsk_end_to_end(self):
        """Test end-to-end differentiability with BPSK."""
        modulator = BPSKModulator()
        demodulator = BPSKDemodulator()

        # Create soft bits with gradients
        soft_bits = torch.tensor([0.1, 0.4, 0.6, 0.9], requires_grad=True)

        # Apply modulation
        symbols = modulator.forward_soft(soft_bits)

        # Apply demodulation
        noise_var = 0.1
        decoded_bits = demodulator.forward_soft(symbols, noise_var)

        # Compute loss between original and decoded bits
        loss = torch.nn.functional.binary_cross_entropy(decoded_bits, soft_bits)
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None

    def test_qpsk_end_to_end(self):
        """Test end-to-end differentiability with QPSK."""
        modulator = QPSKModulator()
        demodulator = QPSKDemodulator()

        # Create soft bits with gradients
        soft_bits = torch.tensor([0.1, 0.9, 0.2, 0.8], requires_grad=True)

        # Apply modulation
        symbols = modulator.forward_soft(soft_bits)

        # Apply demodulation
        noise_var = 0.1
        decoded_bits = demodulator.forward_soft(symbols, noise_var)

        # Compute loss between original and decoded bits
        loss = torch.nn.functional.binary_cross_entropy(decoded_bits, soft_bits)
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None

    def test_qam_end_to_end(self):
        """Test end-to-end differentiability with 16-QAM."""
        modulator = QAMModulator(order=16)
        demodulator = QAMDemodulator(order=16)

        # Create soft bits with gradients (16-QAM has 4 bits per symbol)
        soft_bits = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6], requires_grad=True)

        # Apply modulation
        symbols = modulator.forward_soft(soft_bits)

        # Apply demodulation
        noise_var = 0.1
        decoded_bits = demodulator.forward_soft(symbols, noise_var)

        # Compute loss between original and decoded bits
        loss = torch.nn.functional.binary_cross_entropy(decoded_bits, soft_bits)
        loss.backward()

        # Verify gradients exist
        assert soft_bits.grad is not None
