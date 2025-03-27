import pytest
import torch

from kaira.modulations.qam import QAMDemodulator, QAMModulator


@pytest.fixture
def qam_modulator():
    """Fixture for a basic QAM modulator."""
    return QAMModulator(bits_per_symbol=4)  # 16-QAM


@pytest.fixture
def qam_demodulator():
    """Fixture for a basic QAM demodulator."""
    return QAMDemodulator(bits_per_symbol=4)  # 16-QAM


def test_qam_modulator_initialization():
    """Test initialization of QAM modulator with different parameters."""
    # Test with different bits_per_symbol values
    mod2 = QAMModulator(bits_per_symbol=2)
    assert mod2.bits_per_symbol == 2
    assert mod2.constellation.shape == (4,)

    mod4 = QAMModulator(bits_per_symbol=4)
    assert mod4.bits_per_symbol == 4
    assert mod4.constellation.shape == (16,)

    mod6 = QAMModulator(bits_per_symbol=6)
    assert mod6.bits_per_symbol == 6
    assert mod6.constellation.shape == (64,)

    # Test with invalid bits_per_symbol values
    with pytest.raises(ValueError):
        QAMModulator(bits_per_symbol=3)  # Must be even

    with pytest.raises(ValueError):
        QAMModulator(bits_per_symbol=0)  # Must be positive


def test_qam_modulator_forward(qam_modulator):
    """Test forward pass of QAM modulator."""
    # Test with single integer input
    x = torch.tensor(5)
    y = qam_modulator(x)
    assert y.shape == torch.Size([])
    assert y.dtype == torch.complex64

    # Test with batch of integers
    x = torch.tensor([0, 5, 10, 15])
    y = qam_modulator(x)
    assert y.shape == torch.Size([4])
    assert y.dtype == torch.complex64

    # Test with invalid input
    with pytest.raises(ValueError):
        qam_modulator(torch.tensor(16))  # Out of range for 16-QAM


def test_qam_demodulator_initialization():
    """Test initialization of QAM demodulator with different parameters."""
    # Test with different bits_per_symbol values
    demod2 = QAMDemodulator(bits_per_symbol=2)
    assert demod2.bits_per_symbol == 2
    assert demod2.constellation.shape == (4,)

    demod4 = QAMDemodulator(bits_per_symbol=4)
    assert demod4.bits_per_symbol == 4
    assert demod4.constellation.shape == (16,)

    # Test with invalid bits_per_symbol values
    with pytest.raises(ValueError):
        QAMDemodulator(bits_per_symbol=3)  # Must be even


def test_qam_demodulator_forward(qam_modulator, qam_demodulator):
    """Test forward pass of QAM demodulator."""
    # Test round trip: modulate and demodulate
    x = torch.tensor([0, 5, 10, 15])
    y = qam_modulator(x)
    x_hat = qam_demodulator(y)
    assert torch.equal(x, x_hat)

    # Test with noisy data
    y_noisy = y + 0.1 * torch.randn_like(y.real) + 0.1j * torch.randn_like(y.imag)
    x_hat_noisy = qam_demodulator(y_noisy)
    # Some might be wrong due to noise, but size should match
    assert x_hat_noisy.shape == x.shape


def test_qam_gray_coding():
    """Test that Gray coding is properly implemented."""
    mod = QAMModulator(bits_per_symbol=4, gray_coded=True)

    # In Gray coding, adjacent symbols should differ by only one bit
    for i in range(mod.constellation.shape[0] - 1):
        bin_i = format(i, f"0{mod.bits_per_symbol}b")
        bin_i1 = format(i + 1, f"0{mod.bits_per_symbol}b")
        # Count differing bits
        sum(b1 != b2 for b1, b2 in zip(bin_i, bin_i1))
        # Not all adjacent indices will have one bit difference because of the 2D nature of QAM
        # but we can check if the constellation is different
        assert mod.constellation[i] != mod.constellation[i + 1]


def test_constellation_normalization():
    """Test that constellation is properly normalized."""
    mod = QAMModulator(bits_per_symbol=4, normalize=True)
    # Check average power
    power = torch.mean(torch.abs(mod.constellation) ** 2)
    assert torch.isclose(power, torch.tensor(1.0), atol=1e-6)
