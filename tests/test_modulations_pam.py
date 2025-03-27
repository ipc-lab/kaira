import pytest
import torch

from kaira.modulations.pam import PAMDemodulator, PAMModulator


@pytest.fixture
def pam_modulator():
    """Fixture for a PAM modulator."""
    return PAMModulator(bits_per_symbol=2)  # 4-PAM


@pytest.fixture
def pam_demodulator():
    """Fixture for a PAM demodulator."""
    return PAMDemodulator(bits_per_symbol=2)  # 4-PAM


def test_pam_modulator_initialization():
    """Test initialization of PAM modulator with different parameters."""
    # Test with different bits_per_symbol values
    mod1 = PAMModulator(bits_per_symbol=1)  # 2-PAM (Same as BPSK)
    assert mod1.bits_per_symbol == 1
    assert mod1.constellation.shape == (2,)

    mod2 = PAMModulator(bits_per_symbol=2)  # 4-PAM
    assert mod2.bits_per_symbol == 2
    assert mod2.constellation.shape == (4,)

    mod3 = PAMModulator(bits_per_symbol=3)  # 8-PAM
    assert mod3.bits_per_symbol == 3
    assert mod3.constellation.shape == (8,)

    # Test with invalid bits_per_symbol
    with pytest.raises(ValueError):
        PAMModulator(bits_per_symbol=0)

    # Test with gray coding
    mod_gray = PAMModulator(bits_per_symbol=2, gray_coded=True)
    mod_no_gray = PAMModulator(bits_per_symbol=2, gray_coded=False)
    # They should have different constellation mappings
    assert not torch.allclose(mod_gray.constellation, mod_no_gray.constellation)

    # Test with normalization
    mod_norm = PAMModulator(bits_per_symbol=2, normalize=True)
    # Average power should be 1.0
    power = torch.mean(torch.abs(mod_norm.constellation) ** 2)
    assert torch.isclose(power, torch.tensor(1.0), atol=1e-6)


def test_pam_modulator_forward(pam_modulator):
    """Test forward pass of PAM modulator."""
    # Test with single integer input
    x = torch.tensor(2)
    y = pam_modulator(x)
    assert y.shape == torch.Size([])
    assert y.dtype == torch.float32

    # Test with batch of integers
    x = torch.tensor([0, 1, 2, 3])
    y = pam_modulator(x)
    assert y.shape == torch.Size([4])
    assert y.dtype == torch.float32

    # Test with invalid input
    with pytest.raises(ValueError):
        pam_modulator(torch.tensor(4))  # Out of range for 4-PAM


def test_pam_demodulator_initialization():
    """Test initialization of PAM demodulator."""
    # Test with different bits_per_symbol values
    demod1 = PAMDemodulator(bits_per_symbol=1)
    assert demod1.bits_per_symbol == 1
    assert demod1.constellation.shape == (2,)

    demod2 = PAMDemodulator(bits_per_symbol=2)
    assert demod2.bits_per_symbol == 2
    assert demod2.constellation.shape == (4,)

    # Test with gray coding
    demod_gray = PAMDemodulator(bits_per_symbol=2, gray_coded=True)
    demod_no_gray = PAMDemodulator(bits_per_symbol=2, gray_coded=False)
    assert not torch.allclose(demod_gray.constellation, demod_no_gray.constellation)

    # Test with soft output
    demod_soft = PAMDemodulator(bits_per_symbol=2, soft_output=True)
    assert demod_soft.soft_output is True


def test_pam_demodulator_forward(pam_modulator, pam_demodulator):
    """Test forward pass of PAM demodulator."""
    # Test round trip: modulate and demodulate
    x = torch.tensor([0, 1, 2, 3])
    y = pam_modulator(x)
    x_hat = pam_demodulator(y)
    assert torch.equal(x, x_hat)

    # Test with noisy data
    y_noisy = y + 0.1 * torch.randn_like(y)
    x_hat_noisy = pam_demodulator(y_noisy)
    # Some might be wrong due to noise, but size should match
    assert x_hat_noisy.shape == x.shape


def test_pam_soft_demodulation():
    """Test soft demodulation for PAM."""
    # Create a demodulator with soft output
    demod = PAMDemodulator(bits_per_symbol=2, soft_output=True)

    # Modulate some data
    mod = PAMModulator(bits_per_symbol=2)
    x = torch.tensor([0, 1, 2, 3])
    y = mod(x)

    # Get soft bit LLRs
    llrs = demod(y)
    assert llrs.shape == (4, 2)  # 4 symbols, 2 bits per symbol

    # For perfect reception, LLRs should have high magnitude
    # Positive for 0 bits, negative for 1 bits
    # Check first symbol (usually mapped to all zeros in bit representation)
    assert llrs[0, 0] > 0  # First bit should be 0 (positive LLR)
    assert llrs[0, 1] > 0  # Second bit should be 0 (positive LLR)

    # Test with a noisy point exactly between two constellation points
    # This should give a small LLR for the bit that differs between the two points
    mid_point = (mod.constellation[0] + mod.constellation[1]) / 2
    mid_llrs = demod(mid_point.unsqueeze(0))
    # The LLR for the bit that differs should be small
    assert torch.abs(mid_llrs[0, 1]) < 1.0
