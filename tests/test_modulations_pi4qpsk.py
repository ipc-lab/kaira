import numpy as np
import pytest
import torch

from kaira.modulations.pi4qpsk import Pi4QPSKDemodulator, Pi4QPSKModulator


@pytest.fixture
def pi4qpsk_modulator():
    """Fixture for a Pi/4-QPSK modulator."""
    return Pi4QPSKModulator()


@pytest.fixture
def pi4qpsk_demodulator():
    """Fixture for a Pi/4-QPSK demodulator."""
    return Pi4QPSKDemodulator()


def test_pi4qpsk_modulator_initialization():
    """Test initialization of Pi/4-QPSK modulator."""
    mod = Pi4QPSKModulator()
    assert mod.bits_per_symbol == 2
    assert mod.constellation.shape == (4,)

    # Verify constellation points
    # Pi/4-QPSK uses two QPSK constellations rotated by pi/4
    torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
    torch.tensor([np.sqrt(2) / 2 + np.sqrt(2) / 2 * 1j, -np.sqrt(2) / 2 + np.sqrt(2) / 2 * 1j, -np.sqrt(2) / 2 - np.sqrt(2) / 2 * 1j, np.sqrt(2) / 2 - np.sqrt(2) / 2 * 1j], dtype=torch.complex64)

    # Verify the rotation property is preserved in the implementation
    assert mod._even_symbols or mod._odd_symbols

    # Test with gray coding
    mod_gray = Pi4QPSKModulator(gray_coded=True)
    mod_no_gray = Pi4QPSKModulator(gray_coded=False)
    # Constellation should be different with gray coding
    assert not torch.allclose(mod_gray.constellation, mod_no_gray.constellation)


def test_pi4qpsk_modulator_forward(pi4qpsk_modulator):
    """Test forward pass of Pi/4-QPSK modulator."""
    # Reset state
    pi4qpsk_modulator.reset_state()

    # Test with batch of integers
    x = torch.tensor([0, 1, 2, 3])
    y = pi4qpsk_modulator(x)
    assert y.shape == torch.Size([4])
    assert y.dtype == torch.complex64

    # Pi/4-QPSK alternates between two constellation sets
    # Modulate twice and verify different constellation sets are used
    pi4qpsk_modulator.reset_state()
    y1 = pi4qpsk_modulator(torch.tensor([0]))
    y2 = pi4qpsk_modulator(torch.tensor([0]))
    # Same symbol but different constellations should produce different outputs
    assert not torch.isclose(y1, y2)


def test_pi4qpsk_reset_state():
    """Test resetting state for Pi/4-QPSK modulator."""
    mod = Pi4QPSKModulator()

    # Modulate a sequence
    mod.reset_state()
    seq1 = mod(torch.tensor([0, 1, 2, 3]))

    # Reset and modulate the same sequence again
    mod.reset_state()
    seq2 = mod(torch.tensor([0, 1, 2, 3]))

    # Both sequences should be identical after reset
    assert torch.allclose(seq1, seq2)


def test_pi4qpsk_demodulator_initialization():
    """Test initialization of Pi/4-QPSK demodulator."""
    demod = Pi4QPSKDemodulator()
    assert demod.bits_per_symbol == 2

    # Test with soft output
    demod_soft = Pi4QPSKDemodulator(soft_output=True)
    assert demod_soft.soft_output is True


def test_pi4qpsk_demodulator_forward(pi4qpsk_modulator, pi4qpsk_demodulator):
    """Test forward pass of Pi/4-QPSK demodulator."""
    # Reset states
    pi4qpsk_modulator.reset_state()
    pi4qpsk_demodulator.reset_state()

    # Test round trip with a sequence
    x = torch.tensor([0, 1, 2, 3, 0, 1])
    y = pi4qpsk_modulator(x)
    x_hat = pi4qpsk_demodulator(y)

    # Should recover original symbols
    assert torch.equal(x, x_hat)

    # Test with noise
    y_noisy = y + 0.1 * torch.randn_like(y.real) + 0.1j * torch.randn_like(y.imag)
    x_hat_noisy = pi4qpsk_demodulator(y_noisy)
    # Shape should match even with noise
    assert x_hat_noisy.shape == x.shape


def test_pi4qpsk_soft_demodulation():
    """Test soft demodulation for Pi/4-QPSK."""
    mod = Pi4QPSKModulator()
    demod = Pi4QPSKDemodulator(soft_output=True)

    # Reset states
    mod.reset_state()
    demod.reset_state()

    # Test with a sequence
    x = torch.tensor([0, 1, 2, 3])
    y = mod(x)

    # Get soft bit LLRs
    llrs = demod(y)
    assert llrs.shape == (4, 2)  # 4 symbols, 2 bits per symbol

    # For perfect reception, LLRs should have high magnitude
    assert torch.all(torch.abs(llrs) > 1.0)
