import pytest
import torch
import numpy as np
from kaira.modulations.pam import PAMModulator, PAMDemodulator
from kaira.modulations.utils import binary_to_gray

@pytest.fixture
def device():
    """Fixture providing the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def orders():
    """Fixture providing PAM orders to test."""
    return [2, 4, 8, 16]

@pytest.mark.parametrize("order", [2, 4, 8, 16, 32, 64])
def test_pam_modulator_initialization(order):
    """Test that PAM modulator initializes correctly with various orders."""
    # With gray coding (default)
    mod = PAMModulator(order=order)
    assert mod.order == order
    assert mod.gray_coding is True
    assert mod.normalize is True
    assert mod.bits_per_symbol == int(np.log2(order))
    
    # With binary coding
    mod = PAMModulator(order=order, gray_coding=False)
    assert mod.gray_coding is False
    
    # Without normalization
    mod = PAMModulator(order=order, normalize=False)
    assert mod.normalize is False

def test_pam_modulator_invalid_order():
    """Test that PAM modulator raises an error for invalid orders."""
    with pytest.raises(ValueError):
        PAMModulator(order=3)  # Not a power of 2
    
    with pytest.raises(ValueError):
        PAMModulator(order=7)  # Not a power of 2

def test_pam_constellation_creation():
    """Test that PAM modulator creates the correct constellation."""
    # Test 4-PAM with binary coding
    mod = PAMModulator(order=4, gray_coding=False, normalize=False)
    expected_levels = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    assert torch.allclose(mod.levels, expected_levels)
    
    # Test 4-PAM with gray coding
    mod = PAMModulator(order=4, gray_coding=True, normalize=False)
    # Gray coding should rearrange the levels based on gray code pattern
    expected_bit_patterns = torch.tensor([
        [0., 0.],  # Gray code: 00
        [0., 1.],  # Gray code: 01
        [1., 1.],  # Gray code: 11
        [1., 0.],  # Gray code: 10
    ])
    assert torch.allclose(mod.bit_patterns, expected_bit_patterns)
    
    # Test normalization
    mod = PAMModulator(order=4, normalize=True)
    energy = torch.mean(mod.levels**2)
    assert torch.isclose(energy, torch.tensor(1.0), atol=1e-5)

def test_binary_to_gray():
    """Test the binary to Gray code conversion utility."""
    # Test a few known conversions
    assert binary_to_gray(0) == 0
    assert binary_to_gray(1) == 1
    assert binary_to_gray(2) == 3
    assert binary_to_gray(3) == 2
    assert binary_to_gray(4) == 6
    assert binary_to_gray(5) == 7
    assert binary_to_gray(6) == 5
    assert binary_to_gray(7) == 4

@pytest.mark.parametrize("order", [2, 4, 8, 16])
def test_pam_modulation(order, device):
    """Test PAM modulation with different orders."""
    mod = PAMModulator(order=order).to(device)
    bits_per_symbol = mod.bits_per_symbol
    
    # Create random bits
    batch_size = 10
    n_symbols = 100
    bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
    
    # Modulate
    symbols = mod(bits)
    
    # Check output shape
    assert symbols.shape == (batch_size, n_symbols)
    
    # Check output is complex
    assert symbols.dtype == torch.complex64 or symbols.dtype == torch.complex128
    
    # Check all symbols are in constellation
    real_part = symbols.real.reshape(-1)
    imag_part = symbols.imag.reshape(-1)
    
    # All symbols should have imaginary part as zero
    assert torch.allclose(imag_part, torch.zeros_like(imag_part))
    
    # All real parts should match one of the levels
    for level in mod.levels:
        mask = torch.isclose(real_part, level, atol=1e-5)
        real_part = real_part[~mask]
    
    # After removing all matching symbols, tensor should be empty
    assert real_part.numel() == 0

def test_pam_modulation_input_validation():
    """Test that PAM modulator validates input dimensions."""
    mod = PAMModulator(order=4)  # 2 bits per symbol
    
    # Valid input: multiple of bits_per_symbol
    valid_input = torch.randint(0, 2, (10, 100)).float()
    mod(valid_input)  # Should not raise
    
    # Invalid input: not a multiple of bits_per_symbol
    invalid_input = torch.randint(0, 2, (10, 101)).float()
    with pytest.raises(ValueError):
        mod(invalid_input)

@pytest.mark.parametrize("order", [2, 4, 8, 16])
def test_pam_demodulator_initialization(order):
    """Test that PAM demodulator initializes correctly."""
    demod = PAMDemodulator(order=order)
    assert demod.order == order
    assert demod.gray_coding is True
    assert demod.normalize is True
    assert demod.bits_per_symbol == int(np.log2(order))
    
    # Check modulator reference
    assert demod.modulator.order == order
    assert demod.modulator.gray_coding is True
    assert demod.modulator.normalize is True

@pytest.mark.parametrize("order", [2, 4, 8, 16])
def test_pam_demodulation_hard_decision(order, device):
    """Test PAM hard decision demodulation."""
    mod = PAMModulator(order=order).to(device)
    demod = PAMDemodulator(order=order).to(device)
    
    # Create random bits
    batch_size = 10
    n_symbols = 100
    bits_per_symbol = mod.bits_per_symbol
    original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
    
    # Modulate and then demodulate without noise
    symbols = mod(original_bits)
    demodulated_bits = demod(symbols)
    
    # Check output shape
    assert demodulated_bits.shape == original_bits.shape
    
    # Without noise, demodulated bits should match original bits
    assert torch.allclose(demodulated_bits, original_bits)

@pytest.mark.parametrize("order", [2, 4, 8])
def test_pam_demodulation_with_noise(order, device):
    """Test PAM demodulation with additive noise."""
    mod = PAMModulator(order=order).to(device)
    demod = PAMDemodulator(order=order).to(device)
    
    # Create random bits
    batch_size = 10
    n_symbols = 100
    bits_per_symbol = mod.bits_per_symbol
    original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
    
    # Modulate
    symbols = mod(original_bits)
    
    # Add noise (low SNR)
    noise_std = 0.1
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * noise_std,
        torch.zeros_like(symbols.imag)
    )
    
    # Hard decision demodulation
    demodulated_bits = demod(noisy_symbols)
    
    # Check output shape
    assert demodulated_bits.shape == original_bits.shape
    
    # With some noise, we expect some errors but not 100% error
    bit_error_rate = torch.mean((demodulated_bits != original_bits).float())
    assert bit_error_rate > 0.0  # Some errors
    assert bit_error_rate < 0.5  # But not complete random guessing

@pytest.mark.parametrize("order", [2, 4, 8])
def test_pam_soft_demodulation(order, device):
    """Test PAM soft demodulation (LLR output)."""
    mod = PAMModulator(order=order).to(device)
    demod = PAMDemodulator(order=order).to(device)
    
    # Create random bits
    batch_size = 10
    n_symbols = 100
    bits_per_symbol = mod.bits_per_symbol
    original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
    
    # Modulate
    symbols = mod(original_bits)
    
    # Add noise
    noise_var = 0.1
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * np.sqrt(noise_var),
        torch.zeros_like(symbols.imag)
    )
    
    # Soft demodulation
    llrs = demod(noisy_symbols, noise_var=noise_var)
    
    # Check output shape
    assert llrs.shape == original_bits.shape
    
    # Check LLR properties:
    # 1. When original bit is 0, LLR should tend to be positive
    # 2. When original bit is 1, LLR should tend to be negative
    mask_0 = original_bits == 0
    mask_1 = original_bits == 1
    
    avg_llr_0 = torch.mean(llrs[mask_0])
    avg_llr_1 = torch.mean(llrs[mask_1])
    
    assert avg_llr_0 > 0  # Positive LLR for bit 0
    assert avg_llr_1 < 0  # Negative LLR for bit 1

@pytest.mark.parametrize("order", [2, 4, 8])
def test_pam_modulation_demodulation_consistency(order, device):
    """Test end-to-end consistency of PAM modulation and demodulation."""
    # Initialize with various settings
    test_cases = [
        {"gray_coding": True, "normalize": True},
        {"gray_coding": False, "normalize": True},
        {"gray_coding": True, "normalize": False},
        {"gray_coding": False, "normalize": False}
    ]
    
    batch_size = 10
    n_symbols = 100
    
    for config in test_cases:
        mod = PAMModulator(order=order, **config).to(device)
        demod = PAMDemodulator(order=order, **config).to(device)
        
        bits_per_symbol = mod.bits_per_symbol
        original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
        
        # Modulate and demodulate without noise
        symbols = mod(original_bits)
        demodulated_bits = demod(symbols)
        
        # Check reconstruction
        assert torch.allclose(demodulated_bits, original_bits)

def test_pam_plot_constellation():
    """Test that PAM constellation plotting works."""
    mod = PAMModulator(order=4)
    fig = mod.plot_constellation()
    assert fig is not None

def test_pam_min_distance_calculation():
    """Test the minimum distance calculation used in soft demodulation."""
    demod = PAMDemodulator(order=4)
    
    # Create sample input
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    levels = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    noise_var = torch.tensor([0.1, 0.1, 0.1, 0.1])
    
    # Calculate min distances
    min_dist = demod._min_distance_to_levels(y, levels, noise_var)
    
    # Each symbol should be closest to its corresponding level
    assert min_dist.shape == y.shape
    
    # Test with different noise variances
    varying_noise = torch.tensor([0.1, 0.2, 0.3, 0.4])
    min_dist_varying = demod._min_distance_to_levels(y, levels, varying_noise)
    assert min_dist_varying.shape == y.shape
    
    # Higher noise variance should result in less negative distance
    assert min_dist_varying[0] < min_dist_varying[3]