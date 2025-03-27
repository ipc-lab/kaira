import pytest
import numpy as np
import torch
from kaira.modulations import Pi4QPSKModulator, Pi4QPSKDemodulator  # Changed from Pi4QPSKModem

@pytest.fixture
def modulator():
    """Fixture providing a Pi4QPSK modulator."""
    return Pi4QPSKModulator()

@pytest.fixture
def demodulator():
    """Fixture providing a Pi4QPSK demodulator."""
    return Pi4QPSKDemodulator()

def test_modulator_init(modulator):
    """Test initialization of Pi4QPSK modulator."""
    assert modulator.bits_per_symbol == 2  # Pi4QPSK uses 2 bits per symbol

def test_modulate(modulator):
    """Test Pi4QPSK modulation of bits to symbols."""
    bits = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1], dtype=torch.float32)
    symbols = modulator(bits)
    assert symbols.shape[0] == 4  # 8 bits → 4 symbols
    
    # If constellation is accessible, verify symbols are valid constellation points
    if hasattr(modulator, 'constellation'):
        for symbol in symbols:
            # Check if symbol is close to any constellation point
            distances = torch.abs(symbol - modulator.constellation)
            assert torch.min(distances) < 1e-5

def test_demodulate(demodulator):
    """Test Pi4QPSK demodulation of symbols back to bits."""
    # Define test symbols appropriate for Pi4QPSK
    symbols = torch.tensor([1+1j, -1-1j, 1-1j, -1+1j], dtype=torch.complex64)
    bits = demodulator(symbols)
    assert bits.shape[0] == 8  # 4 symbols → 8 bits
    assert torch.all((bits == 0) | (bits == 1))  # All values must be 0 or 1

def test_modulation_demodulation_cycle():
    """Test Pi4QPSK modulation followed by demodulation recovers original bits."""
    modulator = Pi4QPSKModulator()
    demodulator = Pi4QPSKDemodulator()
    
    # Create random bits (multiple of bits_per_symbol)
    torch.manual_seed(42)
    bits = torch.randint(0, 2, (100,), dtype=torch.float32)
    bits = bits[:len(bits) - (len(bits) % modulator.bits_per_symbol)]
    
    # Modulate bits to symbols
    symbols = modulator(bits)
    
    # Demodulate symbols back to bits
    recovered_bits = demodulator(symbols)
    
    # Check recovered bits match original bits
    assert torch.equal(recovered_bits, bits)
