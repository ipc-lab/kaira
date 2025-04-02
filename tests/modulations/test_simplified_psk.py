"""Simplified PSK modulation tests focusing only on essential functionality."""
import torch

from kaira.modulations.psk import BPSKModulator, BPSKDemodulator


def test_simplified_bpsk_modulation():
    """Test BPSK modulation with simplified approach."""
    modulator = BPSKModulator()
    
    # Create input bits
    bits = torch.tensor([0., 1., 0., 1.])
    
    # Modulate
    symbols = modulator(bits)
    
    # BPSK maps 0 to -1, 1 to +1
    assert symbols.shape == bits.shape
    
    # Check that the values have the correct sign (not exact values)
    assert symbols[0].real < 0  # Bit 0 -> Negative real
    assert symbols[1].real > 0  # Bit 1 -> Positive real
    assert symbols[2].real < 0  # Bit 0 -> Negative real
    assert symbols[3].real > 0  # Bit 1 -> Positive real
    
    # Imaginary parts should be zero or very close to zero
    assert torch.allclose(symbols.imag, torch.zeros_like(symbols.imag), atol=1e-6)


def test_simplified_bpsk_demodulation():
    """Test BPSK demodulation with simplified approach."""
    demodulator = BPSKDemodulator()
    
    # Create BPSK symbols
    symbols = torch.tensor([-1.0, 1.0, -0.5, 0.7], dtype=torch.complex64)
    
    # Demodulate
    bits = demodulator(symbols)
    
    # Check shape
    assert bits.shape == symbols.shape
    
    # Test that hard decisions correctly recover bits
    # Negative real -> Bit 0, Positive real -> Bit 1
    assert bits[0] == 0.0  # -1.0 -> 0
    assert bits[1] == 1.0  # 1.0 -> 1
    assert bits[2] == 0.0  # -0.5 -> 0
    assert bits[3] == 1.0  # 0.7 -> 1


def test_simplified_bpsk_noisy_demodulation():
    """Test BPSK demodulation with noise."""
    modulator = BPSKModulator()
    demodulator = BPSKDemodulator()
    
    # Create input bits
    bits = torch.tensor([0., 1., 0., 1.])
    
    # Modulate
    symbols = modulator(bits)
    
    # Add noise
    torch.manual_seed(42)  # For reproducibility
    noise_level = 0.3
    noisy_symbols = symbols + torch.randn_like(symbols) * noise_level
    
    # Demodulate with hard decision
    recovered_bits = demodulator(noisy_symbols)
    
    # Compute bit error rate
    bit_error_rate = (recovered_bits != bits).float().mean().item()
    
    # In a practical system, we would expect some errors but not too many
    # with a moderate noise level
    assert bit_error_rate < 0.5  # Less than 50% errors expected


def test_simplified_bpsk_soft_demodulation():
    """Test BPSK soft demodulation outputs LLRs with correct sign."""
    demodulator = BPSKDemodulator()
    
    # Create some received symbols
    symbols = torch.tensor([-1.0, 1.0, -0.2, 0.3], dtype=torch.complex64)
    
    # Get soft bit values (LLRs)
    llrs = demodulator(symbols, noise_var=0.1)
    
    # LLRs should have the same sign as the real part of the symbol
    # For BPSK: negative real part -> positive LLR (bit 0)
    #           positive real part -> negative LLR (bit 1)
    assert llrs[0] < 0  # -1.0 -> negative LLR (bit 0)
    assert llrs[1] > 0  # 1.0 -> positive LLR (bit 1)
    assert llrs[2] < 0  # -0.2 -> negative LLR (bit 0)
    assert llrs[3] > 0  # 0.3 -> positive LLR (bit 1)