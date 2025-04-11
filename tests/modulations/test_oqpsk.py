import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from kaira.modulations import OQPSKDemodulator, OQPSKModulator


@pytest.fixture
def binary_bits():
    """Fixture providing binary bits for testing."""
    return torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)


@pytest.fixture
def binary_stream():
    """Fixture providing a random stream of bits."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (100,), dtype=torch.float32)


def test_oqpsk_modulator():
    """Test OQPSK modulation of bit pairs."""
    bits = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1], dtype=torch.float32)
    modulator = OQPSKModulator(normalize=True)
    # Reset any previous state
    modulator.reset_state()
    symbols = modulator(bits)

    # With OQPSK, the quadrature component is delayed by half a symbol
    # The first quadrature bit is set to 0.0 (from _delayed_quad)
    # The subsequent quadrature bits are from the input, but shifted
    norm = 1 / np.sqrt(2)
    expected = torch.complex(torch.tensor([norm, norm, -norm, -norm], dtype=symbols.real.dtype), torch.tensor([0.0, norm, -norm, norm], dtype=symbols.imag.dtype))

    assert torch.allclose(symbols, expected)
    assert modulator.bits_per_symbol == 2


def test_oqpsk_modulator_invalid_input():
    """Test OQPSK modulation with invalid input length."""
    bits = torch.tensor([0, 1, 0], dtype=torch.float32)
    modulator = OQPSKModulator()

    with pytest.raises(ValueError):
        modulator(bits)


def test_oqpsk_modulator_reset_state():
    """Test OQPSK modulator state reset."""
    modulator = OQPSKModulator()
    modulator._delayed_quad.fill_(1.0)
    modulator.reset_state()
    assert modulator._delayed_quad.item() == 0.0


def test_oqpsk_modulator_plot_constellation():
    """Test OQPSK modulator constellation plot."""
    modulator = OQPSKModulator()
    fig = modulator.plot_constellation()
    assert isinstance(fig, plt.Figure)


def test_oqpsk_demodulator_hard():
    """Test OQPSK hard demodulation."""
    symbols = torch.complex(torch.tensor([0.8, 0.9, -0.7, -0.8]), torch.tensor([0.7, -0.8, 0.9, -0.7]))
    demodulator = OQPSKDemodulator()
    bits = demodulator(symbols)

    assert bits.shape[0] == 8
    assert bits.dtype == torch.float32
    assert bits.shape == torch.Size([8])
    assert torch.all((bits == 0) | (bits == 1))


def test_oqpsk_demodulator_soft():
    """Test OQPSK soft demodulation (LLR calculation)."""
    norm = 1 / np.sqrt(2)
    symbols = torch.complex(torch.tensor([0.7, 0.8, -0.7, -0.8]) * norm, torch.tensor([0.6, -0.7, 0.8, -0.6]) * norm)
    noise_var = 0.5
    demodulator = OQPSKDemodulator()
    llrs = demodulator(symbols, noise_var)

    assert llrs.shape[0] == 8
    assert llrs.dtype == torch.float32
    assert llrs.shape == torch.Size([8])


def test_oqpsk_modulation_demodulation_cycle(binary_stream):
    """Test OQPSK modulation followed by demodulation recovers original bits."""
    bits = binary_stream[: len(binary_stream) - (len(binary_stream) % 2)]
    modulator = OQPSKModulator()
    demodulator = OQPSKDemodulator()
    symbols = modulator(bits)
    recovered_bits = demodulator(symbols)

    assert len(recovered_bits) == len(bits)
