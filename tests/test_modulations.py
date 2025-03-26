# tests/test_modulations.py
import pytest
import torch
import numpy as np

from kaira.modulations import (
    BPSKDemodulator,
    BPSKModulator,
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
    IdentityDemodulator,
    IdentityModulator,
    ModulationRegistry,
    OQPSKDemodulator,
    OQPSKModulator,
    PAMDemodulator,
    PAMModulator,
    PSKDemodulator,
    PSKModulator,
    Pi4QPSKDemodulator,
    Pi4QPSKModulator,
    QAMDemodulator,
    QAMModulator,
    QPSKDemodulator,
    QPSKModulator,
    binary_to_gray,
    calculate_spectral_efficiency,
    gray_to_binary,
    plot_constellation,
)


@pytest.fixture
def random_bits():
    """Fixture providing random binary sequence for testing."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (100,)).float()


def test_bpsk_modulation_demodulation(random_bits):
    """Test BPSK modulation and demodulation."""
    # Initialize modulator and demodulator
    modulator = BPSKModulator()
    demodulator = BPSKDemodulator()
    
    # Modulate bits
    symbols = modulator(random_bits)
    
    # Check output shape
    assert symbols.shape == random_bits.shape
    
    # Check BPSK constellation values (-1 and 1)
    assert torch.all((symbols == 1.0) | (symbols == -1.0))
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == random_bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == random_bits)


def test_qpsk_modulation_demodulation():
    """Test QPSK modulation and demodulation."""
    # Create known bit sequence
    bits = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1]).float()
    
    # Initialize modulator and demodulator
    modulator = QPSKModulator()
    demodulator = QPSKDemodulator()
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Check output shape (QPSK: 2 bits per symbol)
    assert symbols.shape == torch.Size([4])
    assert symbols.dtype == torch.complex64
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


@pytest.mark.parametrize("M", [4, 8, 16])
def test_psk_different_orders(M):
    """Test PSK modulation and demodulation with different constellation orders."""
    # Create random bit sequence
    n_bits = int(100 * np.log2(M))
    bits = torch.randint(0, 2, (n_bits,)).float()
    
    # Initialize modulator and demodulator
    modulator = PSKModulator(M=M)
    demodulator = PSKDemodulator(M=M)
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Check output shape (M-PSK: log2(M) bits per symbol)
    expected_n_symbols = n_bits // int(np.log2(M))
    assert symbols.shape == torch.Size([expected_n_symbols])
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


@pytest.mark.parametrize("M", [4, 16, 64])
def test_qam_different_orders(M):
    """Test QAM modulation and demodulation with different constellation orders."""
    # Create random bit sequence
    n_bits = int(100 * np.log2(M))
    bits = torch.randint(0, 2, (n_bits,)).float()
    
    # Initialize modulator and demodulator
    modulator = QAMModulator(M=M)
    demodulator = QAMDemodulator(M=M)
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Check output shape (M-QAM: log2(M) bits per symbol)
    expected_n_symbols = n_bits // int(np.log2(M))
    assert symbols.shape == torch.Size([expected_n_symbols])
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


@pytest.mark.parametrize("M", [2, 4, 8])
def test_pam_different_orders(M):
    """Test PAM modulation and demodulation with different constellation orders."""
    # Create random bit sequence
    n_bits = int(100 * np.log2(M))
    bits = torch.randint(0, 2, (n_bits,)).float()
    
    # Initialize modulator and demodulator
    modulator = PAMModulator(M=M)
    demodulator = PAMDemodulator(M=M)
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Check output shape (M-PAM: log2(M) bits per symbol)
    expected_n_symbols = n_bits // int(np.log2(M))
    assert symbols.shape == torch.Size([expected_n_symbols])
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


def test_dpsk_modulation_demodulation():
    """Test DPSK modulation and demodulation."""
    # Create known bit sequence
    bits = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1]).float()
    
    # Initialize modulator and demodulator
    modulator = DPSKModulator(M=4)  # DQPSK
    demodulator = DPSKDemodulator(M=4)
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


def test_identity_modulation_demodulation(random_bits):
    """Test Identity modulation and demodulation (passthrough)."""
    # Initialize modulator and demodulator
    modulator = IdentityModulator()
    demodulator = IdentityDemodulator()
    
    # Modulate bits
    symbols = modulator(random_bits)
    
    # Check passthrough behavior
    assert torch.all(symbols == random_bits)
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check passthrough behavior
    assert torch.all(recovered_bits == random_bits)


def test_pi4qpsk_modulation_demodulation():
    """Test Pi/4-QPSK modulation and demodulation."""
    # Create bit sequence
    bits = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]).float()
    
    # Initialize modulator and demodulator
    modulator = Pi4QPSKModulator()
    demodulator = Pi4QPSKDemodulator()
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Check output shape (Pi/4-QPSK: 2 bits per symbol)
    expected_n_symbols = len(bits) // 2
    assert symbols.shape == torch.Size([expected_n_symbols])
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation
    assert recovered_bits.shape == bits.shape
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits == bits)


def test_oqpsk_modulation_demodulation():
    """Test OQPSK modulation and demodulation."""
    # Create bit sequence
    bits = torch.tensor([0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]).float()
    
    # Initialize modulator and demodulator
    modulator = OQPSKModulator()
    demodulator = OQPSKDemodulator()
    
    # Modulate bits
    symbols = modulator(bits)
    
    # Demodulate symbols
    recovered_bits = demodulator(symbols)
    
    # Check shape preservation (after removing potential padding)
    assert recovered_bits.shape[0] >= bits.shape[0]
    
    # Check perfect recovery (noiseless case)
    assert torch.all(recovered_bits[:len(bits)] == bits)


def test_gray_binary_conversion():
    """Test Gray code and binary conversion utilities."""
    # Test scalar conversions
    for i in range(16):
        gray = binary_to_gray(i)
        assert gray_to_binary(gray) == i
    
    # Test tensor conversions for 4-bit values
    binary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    gray_codes = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8])
    
    # Test binary to Gray conversion
    for i, b in enumerate(binary):
        g = binary_to_gray(b.item())
        assert g == gray_codes[i].item()
    
    # Test Gray to binary conversion
    for i, g in enumerate(gray_codes):
        b = gray_to_binary(g.item())
        assert b == binary[i].item()


def test_spectral_efficiency():
    """Test spectral efficiency calculation."""
    # BPSK
    bpsk_eff = calculate_spectral_efficiency(modulation="bpsk", coding_rate=1.0)
    assert bpsk_eff == 1.0
    
    # QPSK with rate 1/2 coding
    qpsk_eff = calculate_spectral_efficiency(modulation="qpsk", coding_rate=0.5)
    assert qpsk_eff == 1.0
    
    # 16-QAM with rate 3/4 coding
    qam16_eff = calculate_spectral_efficiency(modulation="16qam", coding_rate=0.75)
    assert qam16_eff == 3.0
    
    # 64-QAM with rate 5/6 coding
    qam64_eff = calculate_spectral_efficiency(modulation="64qam", coding_rate=5/6)
    assert qam64_eff == pytest.approx(5.0, abs=1e-6)


def test_modulation_registry():
    """Test modulation registry functionality."""
    # Test registration
    assert "bpsk" in ModulationRegistry._modulators
    assert "qpsk" in ModulationRegistry._modulators
    assert "qam" in ModulationRegistry._modulators
    
    # Test modulator creation
    bpsk_mod = ModulationRegistry.create_modulator("bpsk")
    assert isinstance(bpsk_mod, BPSKModulator)
    
    qpsk_mod = ModulationRegistry.create_modulator("qpsk")
    assert isinstance(qpsk_mod, QPSKModulator)
    
    qam_mod = ModulationRegistry.create_modulator("qam", M=16)
    assert isinstance(qam_mod, QAMModulator)
    assert qam_mod.M == 16
    
    # Test demodulator creation
    bpsk_demod = ModulationRegistry.create_demodulator("bpsk")
    assert isinstance(bpsk_demod, BPSKDemodulator)
    
    qpsk_demod = ModulationRegistry.create_demodulator("qpsk")
    assert isinstance(qpsk_demod, QPSKDemodulator)
    
    qam_demod = ModulationRegistry.create_demodulator("qam", M=16)
    assert isinstance(qam_demod, QAMDemodulator)
    assert qam_demod.M == 16