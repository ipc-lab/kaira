import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from kaira.modulations.dpsk import (
    DBPSKDemodulator,
    DBPSKModulator,
    DPSKDemodulator,
    DPSKModulator,
    DQPSKDemodulator,
    DQPSKModulator,
)


@pytest.fixture
def dpsk_modulator():
    """Fixture providing a DPSK modulator with order 4."""
    return DPSKModulator(order=4, gray_coding=True)


@pytest.fixture
def dpsk_demodulator():
    """Fixture providing a DPSK demodulator with order 4."""
    return DPSKDemodulator(order=4, gray_coding=True)


@pytest.fixture
def dbpsk_modulator():
    """Fixture for a DBPSK modulator."""
    return DBPSKModulator()


@pytest.fixture
def dbpsk_demodulator():
    """Fixture for a DBPSK demodulator."""
    return DBPSKDemodulator()


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


def test_dpsk_modulator_initialization():
    """Test initialization of DPSK modulator."""
    # Test with different bits_per_symbol values
    mod1 = DPSKModulator(bits_per_symbol=1)  # DBPSK
    assert mod1.bits_per_symbol == 1
    assert mod1.constellation.shape == (2,)

    mod2 = DPSKModulator(bits_per_symbol=2)  # DQPSK
    assert mod2.bits_per_symbol == 2
    assert mod2.constellation.shape == (4,)

    # Test with gray coding
    mod_gray = DPSKModulator(bits_per_symbol=2, gray_coded=True)
    mod_no_gray = DPSKModulator(bits_per_symbol=2, gray_coded=False)
    assert not np.array_equal(mod_gray.constellation.numpy(), mod_no_gray.constellation.numpy())


def test_dpsk_modulator_forward(dpsk_modulator):
    """Test forward pass of DPSK modulator."""
    # Reset state before testing
    dpsk_modulator.reset_state()

    # Test with batch of integers
    x = torch.tensor([0, 1, 2, 3])
    y = dpsk_modulator(x)
    assert y.shape == torch.Size([4])
    assert y.dtype == torch.complex64

    # DPSK has memory, so consecutive calls should depend on state
    dpsk_modulator.reset_state()
    y1 = dpsk_modulator(torch.tensor([0]))
    y2 = dpsk_modulator(torch.tensor([0]))
    # Same symbol twice should result in same phase (relative to previous)
    assert torch.isclose(y1, y2)

    # Different states should produce different outputs for same input
    dpsk_modulator.reset_state()
    y3 = dpsk_modulator(torch.tensor([1]))
    y4 = dpsk_modulator(torch.tensor([1]))
    assert not torch.isclose(y3, y4)


def test_dpsk_reset_state():
    """Test resetting state for DPSK modulator."""
    mod = DPSKModulator(bits_per_symbol=1)

    # Modulate a sequence
    mod.reset_state()
    seq1 = mod(torch.tensor([1, 0, 1, 0]))

    # Reset and modulate the same sequence again
    mod.reset_state()
    seq2 = mod(torch.tensor([1, 0, 1, 0]))

    # Both sequences should be identical after reset
    assert torch.allclose(seq1, seq2)


def test_dpsk_demodulator_initialization():
    """Test initialization of DPSK demodulator."""
    # Test with different bits_per_symbol values
    demod1 = DPSKDemodulator(bits_per_symbol=1)
    assert demod1.bits_per_symbol == 1

    demod2 = DPSKDemodulator(bits_per_symbol=2)
    assert demod2.bits_per_symbol == 2


def test_dpsk_demodulator_forward():
    """Test forward pass of DPSK demodulator."""
    mod = DPSKModulator(bits_per_symbol=2)
    demod = DPSKDemodulator(bits_per_symbol=2)

    # Reset states
    mod.reset_state()
    demod.reset_state()

    # Test round trip with a sequence
    x = torch.tensor([0, 1, 2, 3, 0, 1])
    y = mod(x)
    x_hat = demod(y)

    # First symbol is used as reference, so it's lost
    assert x_hat.shape == torch.Size([5])
    assert torch.equal(x[1:], x_hat)


def test_dbpsk_modulator_forward(dbpsk_modulator):
    """Test forward pass of DBPSK modulator."""
    # Reset state
    dbpsk_modulator.reset_state()

    # Test with a sequence
    x = torch.tensor([0, 1, 0, 0, 1])
    y = dbpsk_modulator(x)
    assert y.shape == torch.Size([5])
    assert y.dtype == torch.complex64

    # The first output is the reference symbol
    assert torch.isclose(y[0], torch.tensor(1.0 + 0.0j, dtype=torch.complex64))


def test_dbpsk_roundtrip():
    """Test round trip encoding and decoding with DBPSK."""
    mod = DBPSKModulator()
    demod = DBPSKDemodulator()

    # Reset states
    mod.reset_state()
    demod.reset_state()

    # Test with a bit sequence
    x = torch.tensor([0, 1, 0, 0, 1, 1, 0])
    y = mod(x)
    x_hat = demod(y)

    # First symbol is reference
    assert x_hat.shape == torch.Size([6])
    assert torch.equal(x[1:], x_hat)

    # Test with separate emissions
    mod.reset_state()
    demod.reset_state()

    emissions = []
    for bit in [0, 1, 0, 1]:
        emissions.append(mod(torch.tensor([bit])))

    # Concatenate emissions
    y_seq = torch.cat(emissions)

    # Demodulate the sequence
    demod.reset_state()
    x_hat_seq = demod(y_seq)

    # Should match the original sequence (minus reference)
    assert torch.equal(torch.tensor([1, 0, 1]), x_hat_seq)
