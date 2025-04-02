import torch
import pytest
import numpy as np
from kaira.modulations import PSKModulator, PSKDemodulator

@pytest.fixture
def bit_sequence():
    """Create a random bit sequence for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (200,), dtype=torch.float)

def test_psk_modulator_initialization():
    """Test initialization of PSK modulator with different parameters."""
    # Create a QPSK modulator (4-PSK)
    mod = PSKModulator(order=4, gray_coding=True)
    
    # Check attributes
    assert mod.order == 4
    assert mod.gray_coding is True
    assert mod.bits_per_symbol == 2
    
    # Check constellation size
    assert mod.constellation.shape[0] == 4
    
    # Try to create with invalid order (not a power of 2)
    with pytest.raises(ValueError):
        PSKModulator(order=7)

def test_psk_modulator_forward():
    """Test PSK modulator forward pass."""
    # Create a QPSK modulator
    mod = PSKModulator(order=4)
    
    # Test with specific bit patterns
    bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float)
    symbols = mod(bits)
    
    # Check output shape
    assert symbols.shape == torch.Size([4])  # 8 bits -> 4 QPSK symbols
    
    # Check that all symbols are on the unit circle
    magnitudes = torch.abs(symbols)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    
    # Test with empty tensor
    empty_bits = torch.tensor([], dtype=torch.float)
    empty_symbols = mod(empty_bits)
    assert empty_symbols.shape == torch.Size([0])
    
    # Test with odd length (should raise error)
    odd_bits = torch.tensor([0, 0, 0], dtype=torch.float)
    with pytest.raises(ValueError):
        mod(odd_bits)

def test_psk_demodulator_forward():
    """Test PSK demodulator forward pass with hard decisions."""
    # Create modulator and demodulator
    order = 4  # QPSK
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Generate bit sequence
    bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Demodulate (hard decision)
    recovered_bits = demodulator(symbols)
    
    # Check shape
    assert recovered_bits.shape == bits.shape
    
    # Check bit recovery (should be perfect for noise-free case)
    assert torch.allclose(recovered_bits, bits)

def test_psk_demodulation_with_noise():
    """Test PSK demodulation with noisy symbols."""
    # Create modulator and demodulator
    order = 8  # 8-PSK
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Generate bit sequence (multiple of 3 bits for 8-PSK)
    bits = torch.tensor([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Add some noise
    noisy_symbols = symbols + (torch.randn_like(symbols.real) + 1j * torch.randn_like(symbols.imag)) * 0.2
    
    # Demodulate (hard decision)
    recovered_bits = demodulator(noisy_symbols)
    
    # Check shape
    assert recovered_bits.shape == bits.shape
    
    # With reasonable noise level, some errors might occur, but not all bits should be wrong
    bit_error_rate = torch.mean((recovered_bits != bits).float())
    assert bit_error_rate < 0.5  # Less than 50% error rate

def test_psk_soft_demodulation():
    """Test PSK soft demodulation (LLR output)."""
    # Create modulator and demodulator
    order = 4  # QPSK for simplicity
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Generate bit sequence
    bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Add small noise
    noise_level = 0.05
    noisy_symbols = symbols + (torch.randn_like(symbols.real) + 1j * torch.randn_like(symbols.imag)) * noise_level
    
    # Get LLRs
    llrs = demodulator(noisy_symbols, noise_var=noise_level**2)
    
    # Check output shape
    assert llrs.shape == bits.shape
    
    # Check that LLR signs match original bits (0 -> positive LLR, 1 -> negative LLR)
    llr_sign_correct = ((llrs > 0) == (bits == 0))
    accuracy = torch.mean(llr_sign_correct.float())
    assert accuracy > 0.7  # Most LLR signs should match bit values

def test_psk_modulation_demodulation_cycle(bit_sequence):
    """Test PSK modulation and demodulation cycle preserves bits."""
    for order in [4, 8, 16]:  # Test different PSK orders
        # Create modulator and demodulator
        modulator = PSKModulator(order=order)
        demodulator = PSKDemodulator(order=order)
        
        # Get number of bits per symbol
        bits_per_symbol = modulator.bits_per_symbol
        
        # Truncate bit sequence to multiple of bits per symbol
        valid_length = (len(bit_sequence) // bits_per_symbol) * bits_per_symbol
        bits = bit_sequence[:valid_length]
        
        # Modulate
        symbols = modulator(bits)
        
        # Demodulate
        recovered_bits = demodulator(symbols)
        
        # Check shape
        assert recovered_bits.shape == bits.shape
        
        # Check bit recovery
        assert torch.allclose(recovered_bits, bits)

def test_psk_gray_coding():
    """Test PSK gray coding property."""
    # Create modulator with gray coding
    order = 8
    modulator = PSKModulator(order=order, gray_coding=True)
    
    # Check that adjacent symbols differ by only one bit
    for i in range(order):
        bits_i = modulator.bit_patterns[i]
        bits_next = modulator.bit_patterns[(i + 1) % order]
        # Count differing bits
        diff_count = torch.sum(bits_i != bits_next).item()
        # Adjacent symbols should differ by exactly one bit
        assert diff_count == 1

def test_psk_no_gray_coding():
    """Test PSK without gray coding."""
    # Create modulator without gray coding
    order = 8
    modulator = PSKModulator(order=order, gray_coding=False)
    
    # Check that the bit patterns match binary counting
    for i in range(order):
        bits = modulator.bit_patterns[i]
        # Convert bits to integer
        bits_int = 0
        for j, bit in enumerate(bits):
            bits_int += int(bit.item()) * (2 ** (len(bits) - j - 1))
        # Should match index
        assert bits_int == i

def test_psk_constellation_properties():
    """Test properties of PSK constellation."""
    # Create a PSK modulator
    order = 16
    modulator = PSKModulator(order=order)
    
    # Check that constellation points are on the unit circle
    magnitudes = torch.abs(modulator.constellation)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    
    # Check that constellation points are evenly spaced in phase
    angles = torch.angle(modulator.constellation)
    # Sort angles (they might not be in order)
    angles, _ = torch.sort(angles)
    # Calculate angle differences (with wrap-around)
    diffs = torch.diff(angles)
    # Add the wrap-around difference
    diffs = torch.cat([diffs, torch.tensor([angles[0] - angles[-1] + 2 * np.pi])])
    # Check if all differences are approximately equal
    expected_diff = 2 * np.pi / order
    assert torch.allclose(diffs, torch.ones_like(diffs) * expected_diff, atol=1e-5)

def test_psk_with_batched_input():
    """Test PSK modulator and demodulator with batched input."""
    # Create modulator and demodulator
    order = 4
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Create batched input [batch_size, sequence_length]
    batch_size = 3
    seq_length = 8  # Must be multiple of bits_per_symbol (2)
    bits = torch.randint(0, 2, (batch_size, seq_length), dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Check output shape [batch_size, sequence_length / bits_per_symbol]
    assert symbols.shape == (batch_size, seq_length // modulator.bits_per_symbol)
    
    # Demodulate
    recovered_bits = demodulator(symbols)
    
    # Check shape
    assert recovered_bits.shape == bits.shape
    
    # Check bit recovery
    assert torch.allclose(recovered_bits, bits)

def test_psk_soft_demodulation_with_batch():
    """Test PSK soft demodulation with batched input."""
    # Create modulator and demodulator
    order = 4
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Create batched input
    batch_size = 2
    seq_length = 8
    bits = torch.randint(0, 2, (batch_size, seq_length), dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Add noise
    noise_var = torch.tensor(0.1)
    noisy_symbols = symbols + (torch.randn_like(symbols.real) + 1j * torch.randn_like(symbols.imag)) * np.sqrt(noise_var)
    
    # Get LLRs with scalar noise variance
    llrs_scalar = demodulator(noisy_symbols, noise_var=noise_var)
    
    # Check shape
    assert llrs_scalar.shape == bits.shape
    
    # Get LLRs with tensor noise variance [batch_size, seq_length // bits_per_symbol]
    noise_var_tensor = torch.full((batch_size, seq_length // modulator.bits_per_symbol), noise_var)
    llrs_tensor = demodulator(noisy_symbols, noise_var=noise_var_tensor)
    
    # Both results should be the same
    assert torch.allclose(llrs_scalar, llrs_tensor)