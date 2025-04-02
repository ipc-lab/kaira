"""Improved tests for PSK modulation schemes."""
import pytest
import torch
import numpy as np
from kaira.modulations.psk import PSKModulator, PSKDemodulator, BPSKModulator, BPSKDemodulator
from kaira.modulations.registry import ModulationRegistry


def test_psk_modulator_instantiation():
    """Test correct instantiation of PSK modulators with different orders."""
    # Test instantiation with different orders
    psk4 = PSKModulator(order=4)
    psk8 = PSKModulator(order=8)
    psk16 = PSKModulator(order=16)
    psk32 = PSKModulator(order=32)
    psk64 = PSKModulator(order=64)
    
    # Check bits per symbol for each order
    assert psk4.bits_per_symbol == 2
    assert psk8.bits_per_symbol == 3
    assert psk16.bits_per_symbol == 4
    assert psk32.bits_per_symbol == 5
    assert psk64.bits_per_symbol == 6
    
    # Check constellation size
    assert psk4.constellation.shape == (4,)
    assert psk8.constellation.shape == (8,)
    assert psk16.constellation.shape == (16,)
    assert psk32.constellation.shape == (32,)
    assert psk64.constellation.shape == (64,)
    
    # Check that points are on unit circle
    for mod in [psk4, psk8, psk16, psk32, psk64]:
        # Check magnitude is approximately 1
        magnitudes = torch.abs(mod.constellation)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), rtol=1e-5)


def test_psk_modulator_invalid_orders():
    """Test that PSK modulators reject invalid orders."""
    # Test non-power-of-2 orders
    with pytest.raises(ValueError):
        PSKModulator(order=3)
    
    with pytest.raises(ValueError):
        PSKModulator(order=6)
    
    with pytest.raises(ValueError):
        PSKModulator(order=10)


def test_psk_gray_coding():
    """Test that Gray coding works correctly for PSK modulation."""
    # Create modulator with Gray coding
    mod_gray = PSKModulator(order=8, gray_coding=True)
    
    # Create modulator without Gray coding
    mod_no_gray = PSKModulator(order=8, gray_coding=False)
    
    # Check that the bit patterns are different
    assert not torch.equal(mod_gray.bit_patterns, mod_no_gray.bit_patterns)
    
    # Verify Gray code properties: adjacent symbols differ by exactly one bit
    for i in range(8):
        next_i = (i + 1) % 8
        # Count differing bits
        diff_bits = torch.sum(mod_gray.bit_patterns[i] != mod_gray.bit_patterns[next_i])
        assert diff_bits == 1


def test_bpsk_modulator_forward():
    """Test BPSK modulator forward pass."""
    mod = BPSKModulator()
    
    # Test with individual bits
    x = torch.tensor([0, 1, 0, 1])
    y = mod(x)
    
    assert y.shape == x.shape
    assert y.dtype == torch.complex64
    
    # Check specific values
    expected = torch.complex(torch.tensor([-1.0, 1.0, -1.0, 1.0]), torch.zeros(4))
    assert torch.allclose(y, expected)
    
    # Test with batched input
    x_batch = torch.tensor([[0, 1], [1, 0]])
    y_batch = mod(x_batch)
    
    assert y_batch.shape == x_batch.shape
    expected_batch = torch.complex(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]), torch.zeros((2, 2)))
    assert torch.allclose(y_batch, expected_batch)


def test_bpsk_demodulator_forward():
    """Test BPSK demodulator forward pass."""
    demod = BPSKDemodulator()
    
    # Test hard decisions
    y = torch.complex(torch.tensor([-1.0, 1.0, -0.2, 0.2]), torch.zeros(4))
    bits = demod(y)
    
    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.allclose(bits, expected)
    
    # Test soft decisions (LLRs)
    noise_var = 0.1
    llrs = demod(y, noise_var)
    
    # LLRs should be proportional to the real component
    expected_llrs = 2 * torch.tensor([-1.0, 1.0, -0.2, 0.2]) / noise_var
    assert torch.allclose(llrs, expected_llrs)


def test_psk_modulator_forward():
    """Test PSK modulator forward pass."""
    mod = PSKModulator(order=4)  # QPSK equivalent
    
    # Create all possible bit combinations for QPSK (2 bits per symbol)
    bits = torch.tensor([
        [0, 0],  # first symbol
        [0, 1],  # second symbol
        [1, 0],  # third symbol
        [1, 1],  # fourth symbol
    ]).flatten()
    
    # Modulate
    symbols = mod(bits)
    
    # Should get 2 complex symbols
    assert symbols.shape == (2,)
    assert symbols.dtype == torch.complex64
    
    # Check constellation points are used correctly
    used_points = set([complex(symbols[i].item()) for i in range(len(symbols))])
    all_points = set([complex(p.item()) for p in mod.constellation])
    
    # At least some of the constellation points should be used
    assert len(used_points.intersection(all_points)) > 0
    
    # Test with batch dimension
    batch_bits = bits.reshape(2, 4)  # 2 batches, 4 bits each
    batch_symbols = mod(batch_bits)
    assert batch_symbols.shape == (2, 2)  # 2 batches, 2 symbols


def test_psk_demodulator_forward_with_noise():
    """Test PSK demodulator with noisy input."""
    mod = PSKModulator(order=8)
    demod = PSKDemodulator(order=8)
    
    # Create random bits
    bits_per_symbol = 3  # log2(8)
    num_symbols = 100
    bits = torch.randint(0, 2, (num_symbols * bits_per_symbol,), dtype=torch.float)
    
    # Modulate
    symbols = mod(bits)
    
    # Add noise
    noise_level = 0.01
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * noise_level,
        torch.randn_like(symbols.imag) * noise_level
    )
    
    # Demodulate (hard decision)
    decoded_bits = demod(noisy_symbols)
    
    # Shape should match original bits
    assert decoded_bits.shape == bits.shape
    
    # With low noise, most bits should be recovered correctly
    bit_errors = (decoded_bits != bits).sum().item()
    assert bit_errors / len(bits) < 0.1  # Less than 10% error rate
    
    
def test_psk_demodulator_soft_output():
    """Test soft output (LLRs) from PSK demodulator."""
    order = 4  # QPSK
    mod = PSKModulator(order=order)
    demod = PSKDemodulator(order=order)
    
    # Create specific input for testing
    bits = torch.tensor([0, 0, 1, 0], dtype=torch.float)
    symbols = mod(bits)
    
    # Add small noise
    noise_level = 0.1
    noisy_symbols = symbols + 0.1
    
    # Demodulate with soft output
    llrs = demod(noisy_symbols, noise_var=noise_level**2)
    
    # LLRs should have correct shape
    assert llrs.shape == bits.shape
    
    # Verify that LLRs have reasonable signs
    # For high certainty, LLRs should be: 
    # - Positive for bits that are likely 0
    # - Negative for bits that are likely 1
    assert llrs[0] > 0  # First bit is 0, so LLR should be positive
    assert llrs[2] < 0  # Third bit is 1, so LLR should be negative


def test_psk_modulation_demodulation_roundtrip():
    """Test complete modulation and demodulation cycle for different PSK orders."""
    for order in [4, 8, 16]:
        # Create modulator and demodulator
        mod = PSKModulator(order=order)
        demod = PSKDemodulator(order=order)
        
        # Get bits per symbol
        bits_per_symbol = mod.bits_per_symbol
        
        # Create all possible bit patterns
        num_patterns = order
        all_patterns = []
        for i in range(num_patterns):
            binary = format(i, f'0{bits_per_symbol}b')
            pattern = [int(b) for b in binary]
            all_patterns.extend(pattern)
        
        # Convert to tensor
        all_bits = torch.tensor(all_patterns, dtype=torch.float)
        
        # Modulate
        symbols = mod(all_bits)
        
        # Demodulate (noise-free)
        recovered_bits = demod(symbols)
        
        # All bits should be recovered correctly in noise-free case
        assert torch.allclose(recovered_bits, all_bits)


def test_modulation_registry_contains_psk():
    """Test that PSK modulators and demodulators are properly registered."""
    # Check BPSK
    assert "bpskmodulator" in ModulationRegistry._modulators
    assert "bpskdemodulator" in ModulationRegistry._demodulators
    
    # Check QPSK
    assert "qpskmodulator" in ModulationRegistry._modulators
    assert "qpskdemodulator" in ModulationRegistry._demodulators
    
    # Check general PSK
    assert "pskmodulator" in ModulationRegistry._modulators
    assert "pskdemodulator" in ModulationRegistry._demodulators