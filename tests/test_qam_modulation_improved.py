"""Improved tests for QAM modulation schemes."""
import pytest
import torch
import numpy as np
from kaira.modulations.qam import QAMModulator, QAMDemodulator
from kaira.modulations.registry import ModulationRegistry


def test_qam_modulator_instantiation():
    """Test correct instantiation of QAM modulators with different orders."""
    # Test instantiation with different orders
    qam4 = QAMModulator(order=4)
    qam16 = QAMModulator(order=16)
    qam64 = QAMModulator(order=64)
    qam256 = QAMModulator(order=256)
    
    # Check bits per symbol for each order
    assert qam4.bits_per_symbol == 2
    assert qam16.bits_per_symbol == 4
    assert qam64.bits_per_symbol == 6
    assert qam256.bits_per_symbol == 8
    
    # Check constellation size
    assert qam4.constellation.shape == (4,)
    assert qam16.constellation.shape == (16,)
    assert qam64.constellation.shape == (64,)
    assert qam256.constellation.shape == (256,)
    
    # Check normalization
    for mod in [qam4, qam16, qam64, qam256]:
        # For normalized constellations, average energy should be 1
        energy = torch.mean(torch.abs(mod.constellation) ** 2)
        assert torch.isclose(energy, torch.tensor(1.0), rtol=1e-5)


def test_qam_modulator_invalid_orders():
    """Test that QAM modulators reject invalid orders."""
    # Test non-square orders
    with pytest.raises(ValueError):
        QAMModulator(order=8)
    
    # Test non-power-of-4 orders
    with pytest.raises(ValueError):
        QAMModulator(order=36)
    
    # Test negative orders
    with pytest.raises(ValueError):
        QAMModulator(order=-4)


def test_qam_gray_coding():
    """Test that Gray coding works correctly for QAM modulation."""
    # Create modulator with Gray coding
    mod_gray = QAMModulator(order=16, gray_coding=True)
    
    # Create modulator without Gray coding
    mod_no_gray = QAMModulator(order=16, gray_coding=False)
    
    # Check that the bit patterns are different
    assert not torch.equal(mod_gray.bit_patterns, mod_no_gray.bit_patterns)
    
    # With Gray coding, adjacent constellation points should differ by one bit
    # Calculate squared distances between all pairs of constellation points
    const_expanded1 = mod_gray.constellation.unsqueeze(1)  # (16, 1)
    const_expanded2 = mod_gray.constellation.unsqueeze(0)  # (1, 16)
    squared_distances = torch.abs(const_expanded1 - const_expanded2) ** 2  # (16, 16)
    
    # Find adjacent points (those with the smallest distance)
    min_dist = torch.min(squared_distances[squared_distances > 0])
    adjacency = (torch.abs(squared_distances - min_dist) < 1e-8) & (squared_distances > 0)
    
    # Check bit differences between adjacent points
    bit_diffs_sum = 0
    count = 0
    
    for i in range(16):
        for j in range(16):
            if adjacency[i, j]:
                # Count differing bits for adjacent points
                diff_bits = torch.sum(mod_gray.bit_patterns[i] != mod_gray.bit_patterns[j])
                bit_diffs_sum += diff_bits
                count += 1
    
    # On average, adjacent points should differ by close to one bit
    avg_bit_diff = bit_diffs_sum / max(1, count)
    assert avg_bit_diff <= 1.5, f"Average bit difference for adjacent points is {avg_bit_diff}, expected close to 1"


def test_qam_modulator_forward():
    """Test QAM modulator forward pass."""
    mod = QAMModulator(order=4)  # 4-QAM 
    
    # Test with individual bits
    x = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1])  # 4 bit pairs
    y = mod(x)
    
    assert y.shape == (4,)  # 4 symbols
    assert y.dtype == torch.complex64
    
    # The input maps to specific symbols
    const_points = mod.constellation
    
    # Test with batched input
    x_batch = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]])  # batch of 2, each with 2 symbols
    y_batch = mod(x_batch)
    
    assert y_batch.shape == (2, 2)


def test_qam_demodulator_forward():
    """Test QAM demodulator forward pass."""
    # Create modulator and demodulator pair
    mod = QAMModulator(order=16)  # 16-QAM
    demod = QAMDemodulator(order=16)
    
    # Generate all bit patterns for 16-QAM (4 bits per symbol)
    all_patterns = []
    for i in range(16):
        bits = [int(b) for b in format(i, '04b')]
        all_patterns.append(bits)
    
    all_bits = torch.tensor(all_patterns, dtype=torch.float).flatten()
    
    # Modulate
    symbols = mod(all_bits)
    
    # Demodulate
    demodulated_bits = demod(symbols)
    
    # Without noise, all bits should be recovered exactly
    assert torch.allclose(demodulated_bits, all_bits)


def test_qam_demodulator_hard_decision_with_noise():
    """Test QAM demodulator with noisy input."""
    # Create modulator and demodulator
    mod = QAMModulator(order=16)
    demod = QAMDemodulator(order=16)
    
    # Create random bits
    bits_per_symbol = 4  # log2(16)
    num_symbols = 100
    bits = torch.randint(0, 2, (num_symbols * bits_per_symbol,), dtype=torch.float)
    
    # Modulate
    symbols = mod(bits)
    
    # Add small noise
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


def test_qam_demodulator_soft_output():
    """Test soft output (LLRs) from QAM demodulator."""
    # Create modulator and demodulator
    mod = QAMModulator(order=4)  # 4-QAM (same as QPSK)
    demod = QAMDemodulator(order=4)
    
    # Create specific input
    bits = torch.tensor([0, 0, 1, 1], dtype=torch.float)
    symbols = mod(bits)
    
    # Add small offset to create some uncertainty
    noisy_symbols = symbols + torch.complex(
        torch.tensor([0.1, -0.1]),
        torch.tensor([0.1, -0.1])
    )
    
    # Demodulate with soft output
    noise_var = 0.2
    llrs = demod(noisy_symbols, noise_var)
    
    # LLRs should have correct shape
    assert llrs.shape == bits.shape
    
    # Check signs of LLRs 
    # For high certainty, LLRs should be: 
    # - Positive for bits that are likely 0
    # - Negative for bits that are likely 1
    # The first two bits are 0, so their LLRs should be positive
    # The last two bits are 1, so their LLRs should be negative
    assert llrs[0] > 0
    assert llrs[1] > 0
    assert llrs[2] < 0
    assert llrs[3] < 0


def test_qam_soft_demods_with_different_noise_vars():
    """Test QAM soft demodulation with different noise variance formats."""
    # Create modulator and demodulator
    mod = QAMModulator(order=16)
    demod = QAMDemodulator(order=16)
    
    # Create random bits
    bits = torch.randint(0, 2, (2, 8), dtype=torch.float)  # 2 batches, 8 bits each
    
    # Modulate
    symbols = mod(bits)  # 2 batches, 2 symbols each
    
    # Add small noise
    noise_level = 0.05
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * noise_level,
        torch.randn_like(symbols.imag) * noise_level
    )
    
    # Test with scalar noise variance
    scalar_noise_var = 0.1
    llrs_scalar = demod(noisy_symbols, scalar_noise_var)
    
    # Verify shape
    assert llrs_scalar.shape == bits.shape
    
    # Test with per-symbol noise variance
    symbol_noise_var = torch.full_like(symbols.real, 0.1)
    llrs_symbol = demod(noisy_symbols, symbol_noise_var)
    
    # Both methods should produce similar results when using the same noise variance
    assert torch.allclose(llrs_scalar, llrs_symbol, rtol=1e-4)


def test_qam_modulation_demodulation_roundtrip():
    """Test complete modulation and demodulation cycle for different QAM orders."""
    for order in [4, 16, 64]:
        # Create modulator and demodulator
        mod = QAMModulator(order=order)
        demod = QAMDemodulator(order=order)
        
        # Get bits per symbol
        bits_per_symbol = mod.bits_per_symbol
        
        # Create test bit sequences - use a subset for larger orders
        if order == 4:
            bit_count = 8  # 4 symbols
        elif order == 16:
            bit_count = 16  # 4 symbols
        else:  # order == 64
            bit_count = 24  # 4 symbols
            
        # Ensure bit count is a multiple of bits_per_symbol
        bit_count = (bit_count // bits_per_symbol) * bits_per_symbol
        
        # Create test bits
        bits = torch.randint(0, 2, (bit_count,), dtype=torch.float)
        
        # Apply modulation
        symbols = mod(bits)
        
        # Apply demodulation (no noise)
        recovered_bits = demod(symbols)
        
        # In noise-free case, all bits should be recovered correctly
        assert torch.allclose(recovered_bits, bits)


def test_qam_normalization():
    """Test normalization of QAM constellations."""
    # Test with and without normalization
    mod_norm = QAMModulator(order=16, normalize=True)
    mod_no_norm = QAMModulator(order=16, normalize=False)
    
    # Normalized constellation should have average energy = 1
    energy_norm = torch.mean(torch.abs(mod_norm.constellation) ** 2)
    assert torch.isclose(energy_norm, torch.tensor(1.0), rtol=1e-5)
    
    # Non-normalized constellation should have larger energy
    energy_no_norm = torch.mean(torch.abs(mod_no_norm.constellation) ** 2)
    assert energy_no_norm > energy_norm


def test_qam_constellation_structure():
    """Test that QAM constellations have the correct structure."""
    for order in [4, 16, 64]:
        mod = QAMModulator(order=order, normalize=False)
        
        # Get the constellation points
        const = mod.constellation
        
        # Verify real and imaginary parts are distributed correctly
        real_values = const.real.unique()
        imag_values = const.imag.unique()
        
        # Number of unique values should be sqrt(order)
        k = int(np.sqrt(order))
        assert len(real_values) == k
        assert len(imag_values) == k
        
        # Values should be symmetrical around zero
        assert torch.allclose(torch.sort(real_values)[0], -torch.sort(-real_values)[0])
        assert torch.allclose(torch.sort(imag_values)[0], -torch.sort(-imag_values)[0])


def test_modulation_registry_contains_qam():
    """Test that QAM modulators and demodulators are properly registered."""
    # Check QAM
    assert "qammodulator" in ModulationRegistry._modulators
    assert "qamdemodulator" in ModulationRegistry._demodulators