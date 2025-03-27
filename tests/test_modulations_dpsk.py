import pytest
import torch
import matplotlib.pyplot as plt
from kaira.modulations.dpsk import (
    DPSKModulator, DPSKDemodulator, 
    DBPSKModulator, DBPSKDemodulator, 
    DQPSKModulator, DQPSKDemodulator
)

@pytest.fixture
def dpsk_modulator():
    """Fixture providing a DPSK modulator with order 4."""
    return DPSKModulator(order=4, gray_coding=True)

@pytest.fixture
def dpsk_demodulator():
    """Fixture providing a DPSK demodulator with order 4."""
    return DPSKDemodulator(order=4, gray_coding=True)

def test_dpsk_modulator_init(dpsk_modulator):
    """Test initialization of DPSK modulator."""
    assert dpsk_modulator.order == 4
    assert dpsk_modulator.gray_coding is True
    assert dpsk_modulator.bits_per_symbol == 2

def test_dpsk_modulator_create_constellation(dpsk_modulator):
    """Test constellation creation for DPSK modulator."""
    assert dpsk_modulator.constellation.shape == (4,)
    assert dpsk_modulator.bit_patterns.shape == (4, 2)

def test_dpsk_modulator_forward(dpsk_modulator):
    """Test forward method of DPSK modulator."""
    x = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.float32)
    output = dpsk_modulator(x)
    assert output.shape == (4,)  # 8 bits → 4 symbols

def test_dpsk_modulator_reset_state(dpsk_modulator):
    """Test state reset in DPSK modulator."""
    dpsk_modulator.reset_state()
    assert torch.equal(dpsk_modulator._phase_memory, torch.tensor(1.0 + 0.0j))

def test_dpsk_modulator_plot_constellation(dpsk_modulator):
    """Test constellation plotting for DPSK modulator."""
    fig = dpsk_modulator.plot_constellation()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Close to avoid memory issues

def test_dpsk_demodulator_init(dpsk_demodulator):
    """Test initialization of DPSK demodulator."""
    assert dpsk_demodulator.order == 4
    assert dpsk_demodulator.gray_coding is True
    assert dpsk_demodulator.bits_per_symbol == 2

def test_dpsk_demodulator_forward(dpsk_demodulator):
    """Test forward method of DPSK demodulator."""
    y = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
    output = dpsk_demodulator(y)
    assert output.shape == (4,)  # 2 symbols → 4 bits

def test_dpsk_demodulator_forward_with_noise(dpsk_demodulator):
    """Test forward method with noise variance for DPSK demodulator."""
    y = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
    noise_var = 0.1
    output = dpsk_demodulator(y, noise_var=noise_var)
    assert output.shape == (4,)  # 2 symbols → 4 bits

def test_dbpsk_modulator_init():
    """Test initialization of DBPSK modulator."""
    modulator = DBPSKModulator()
    assert modulator.order == 2
    assert modulator.gray_coding is True
    assert modulator.bits_per_symbol == 1

def test_dbpsk_demodulator_init():
    """Test initialization of DBPSK demodulator."""
    demodulator = DBPSKDemodulator()
    assert demodulator.order == 2
    assert demodulator.gray_coding is True
    assert demodulator.bits_per_symbol == 1

def test_dqpsk_modulator_init():
    """Test initialization of DQPSK modulator."""
    modulator = DQPSKModulator()
    assert modulator.order == 4
    assert modulator.gray_coding is True
    assert modulator.bits_per_symbol == 2

def test_dqpsk_demodulator_init():
    """Test initialization of DQPSK demodulator."""
    demodulator = DQPSKDemodulator()
    assert demodulator.order == 4
    assert demodulator.gray_coding is True
    assert demodulator.bits_per_symbol == 2

def test_dpsk_modulation_demodulation_cycle():
    """Test complete DPSK modulation and demodulation cycle."""
    modulator = DPSKModulator(order=4, gray_coding=True)
    demodulator = DPSKDemodulator(order=4, gray_coding=True)
    
    # Create test bits (multiple of bits_per_symbol)
    bits = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.float32)
    
    # Modulate bits to symbols
    symbols = modulator(bits)
    
    # Demodulate symbols back to bits
    recovered_bits = demodulator(symbols)
    
    # Check recovered bits match original bits
    assert torch.equal(recovered_bits, bits)
