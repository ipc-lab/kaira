import pytest
import numpy as np
from kaira.modulations import PAMModulator, PAMDemodulator  # Changed from PAMModulation

@pytest.fixture
def modulator():
    """Fixture providing a PAM modulator with order 4."""
    return PAMModulator(order=4)

@pytest.fixture
def demodulator():
    """Fixture providing a PAM demodulator with order 4."""
    return PAMDemodulator(order=4)

def test_modulator_init(modulator):
    """Test initialization of PAM modulator."""
    assert modulator.order == 4
    assert modulator.bits_per_symbol == 2  # log2(4)

def test_modulate(modulator):
    """Test PAM modulation of bits to symbols."""
    bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float32)
    symbols = modulator(bits)
    assert symbols.shape[0] == 4  # 8 bits → 4 symbols
    
    # Test specific symbol values if appropriate
    if hasattr(modulator, 'constellation'):
        assert torch.all(torch.isin(symbols, modulator.constellation))

def test_demodulate(demodulator):
    """Test PAM demodulation of symbols back to bits."""
    # Test with constellation points if available, otherwise use appropriate test values
    if hasattr(demodulator, 'constellation'):
        symbols = demodulator.constellation
    else:
        symbols = torch.tensor([-3, -1, 1, 3], dtype=torch.float32)
    
    bits = demodulator(symbols)
    assert bits.shape[0] == 8  # 4 symbols → 8 bits
    assert torch.all((bits == 0) | (bits == 1))  # All values must be 0 or 1

def test_modulation_demodulation_cycle():
    """Test PAM modulation followed by demodulation recovers original bits."""
    modulator = PAMModulator(order=4)
    demodulator = PAMDemodulator(order=4)
    
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
