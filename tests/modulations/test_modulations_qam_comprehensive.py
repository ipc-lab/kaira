import pytest
import torch
import numpy as np
from kaira.modulations.qam import QAMModulator, QAMDemodulator
from kaira.modulations.utils import binary_to_gray

@pytest.fixture
def device():
    """Fixture providing the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def orders():
    """Fixture providing QAM orders to test."""
    return [4, 16, 64, 256]

@pytest.mark.parametrize("order", [4, 16, 64, 256])
def test_qam_modulator_initialization(order):
    """Test that QAM modulator initializes correctly with various orders."""
    # With gray coding (default)
    mod = QAMModulator(order=order)
    assert mod.order == order
    assert mod.gray_coding is True
    assert mod.normalize is True
    assert mod.bits_per_symbol == int(np.log2(order))
    
    # With binary coding
    mod = QAMModulator(order=order, gray_coding=False)
    assert mod.gray_coding is False
    
    # Without normalization
    mod = QAMModulator(order=order, normalize=False)
    assert mod.normalize is False

def test_qam_modulator_invalid_order():
    """Test that QAM modulator raises an error for invalid orders."""
    with pytest.raises(ValueError):
        QAMModulator(order=8)  # Not a perfect square and power of 4
    
    with pytest.raises(ValueError):
        QAMModulator(order=36)  # Not a power of 4

def test_qam_constellation_creation():
    """Test that QAM modulator creates the correct constellation."""
    # Test 16-QAM with binary coding
    mod = QAMModulator(order=16, gray_coding=False, normalize=False)
    
    # Check constellation size
    assert mod.constellation.shape == (16,)
    
    # Check the real and imaginary parts cover all combinations of [-3, -1, 1, 3]
    real_vals = mod.constellation.real
    imag_vals = mod.constellation.imag
    
    unique_real = torch.unique(real_vals)
    unique_imag = torch.unique(imag_vals)
    
    assert len(unique_real) == 4
    assert len(unique_imag) == 4
    
    expected_vals = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    assert torch.allclose(torch.sort(unique_real)[0], expected_vals)
    assert torch.allclose(torch.sort(unique_imag)[0], expected_vals)
    
    # Test normalization
    mod_norm = QAMModulator(order=16, normalize=True)
    energy = torch.mean(torch.abs(mod_norm.constellation) ** 2)
    assert torch.isclose(energy, torch.tensor(1.0), atol=1e-5)

def test_qam_gray_coding():
    """Test QAM Gray coding bit mapping."""
    # Create a 16-QAM modulator with Gray coding
    mod = QAMModulator(order=16, gray_coding=True)
    
    # In Gray coding, adjacent constellation points should differ by only one bit
    # Test a few known bit patterns to confirm proper Gray mapping
    bit_patterns = mod.bit_patterns
    
    # Find constellation points that are adjacent (differ by 2 in one dimension)
    for i in range(16):
        point_i = mod.constellation[i]
        for j in range(i+1, 16):
            point_j = mod.constellation[j]
            
            # Check if they're adjacent (considering normalization)
            # Adjacent points have a distance equal to the smallest grid spacing
            diff = torch.abs(point_i - point_j)
            is_adjacent = (diff.real == 0 and diff.imag > 0) or (diff.real > 0 and diff.imag == 0)
            
            if is_adjacent:
                # Adjacent points in Gray coding should differ by exactly one bit
                bit_diff = torch.sum(bit_patterns[i] != bit_patterns[j])
                assert bit_diff == 1

@pytest.mark.parametrize("order", [4, 16, 64])
def test_qam_modulation(order, device):
    """Test QAM modulation with different orders."""
    mod = QAMModulator(order=order).to(device)
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
    symbols_flat = symbols.reshape(-1)
    
    # Each symbol should be one of the constellation points
    for sym in symbols_flat:
        # Find if this symbol matches any constellation point
        distances = torch.abs(sym - mod.constellation)
        min_dist = torch.min(distances)
        # Should be very close to one of the constellation points
        assert min_dist < 1e-5

def test_qam_modulation_input_validation():
    """Test that QAM modulator validates input dimensions."""
    mod = QAMModulator(order=16)  # 4 bits per symbol
    
    # Valid input: multiple of bits_per_symbol
    valid_input = torch.randint(0, 2, (10, 100)).float()
    mod(valid_input)  # Should not raise
    
    # Invalid input: not a multiple of bits_per_symbol
    invalid_input = torch.randint(0, 2, (10, 101)).float()
    with pytest.raises(ValueError):
        mod(invalid_input)

@pytest.mark.parametrize("order", [4, 16, 64])
def test_qam_demodulator_initialization(order):
    """Test that QAM demodulator initializes correctly."""
    demod = QAMDemodulator(order=order)
    assert demod.order == order
    assert demod.gray_coding is True
    assert demod.normalize is True
    assert demod.bits_per_symbol == int(np.log2(order))
    
    # Check modulator reference
    assert demod.modulator.order == order
    assert demod.modulator.gray_coding is True
    assert demod.modulator.normalize is True

@pytest.mark.parametrize("order", [4, 16, 64])
def test_qam_demodulation_hard_decision(order, device):
    """Test QAM hard decision demodulation."""
    mod = QAMModulator(order=order).to(device)
    demod = QAMDemodulator(order=order).to(device)
    
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

@pytest.mark.parametrize("order", [4, 16, 64])
def test_qam_demodulation_with_noise(order, device):
    """Test QAM demodulation with additive noise."""
    mod = QAMModulator(order=order).to(device)
    demod = QAMDemodulator(order=order).to(device)
    
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
        torch.randn_like(symbols.imag) * noise_std
    )
    
    # Hard decision demodulation
    demodulated_bits = demod(noisy_symbols)
    
    # Check output shape
    assert demodulated_bits.shape == original_bits.shape
    
    # With some noise, we expect some errors but not 100% error
    bit_error_rate = torch.mean((demodulated_bits != original_bits).float())
    assert bit_error_rate > 0.0  # Some errors
    assert bit_error_rate < 0.5  # But not complete random guessing

@pytest.mark.parametrize("order", [4, 16])
def test_qam_soft_demodulation(order, device):
    """Test QAM soft demodulation (LLR output)."""
    mod = QAMModulator(order=order).to(device)
    demod = QAMDemodulator(order=order).to(device)
    
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
        torch.randn_like(symbols.imag) * np.sqrt(noise_var)
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

@pytest.mark.parametrize("order", [4, 16])
def test_qam_modulation_demodulation_consistency(order, device):
    """Test end-to-end consistency of QAM modulation and demodulation."""
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
        mod = QAMModulator(order=order, **config).to(device)
        demod = QAMDemodulator(order=order, **config).to(device)
        
        bits_per_symbol = mod.bits_per_symbol
        original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
        
        # Modulate and demodulate without noise
        symbols = mod(original_bits)
        demodulated_bits = demod(symbols)
        
        # Check reconstruction
        assert torch.allclose(demodulated_bits, original_bits)

def test_qam_plot_constellation():
    """Test that QAM constellation plotting works."""
    mod = QAMModulator(order=16)
    fig = mod.plot_constellation()
    assert fig is not None

def test_qam_min_distance_calculation():
    """Test the minimum distance calculation used in soft demodulation."""
    demod = QAMDemodulator(order=16)
    
    # Create sample input
    y = torch.tensor([1.0+1.0j, -1.0-1.0j, 1.0-1.0j, -1.0+1.0j], dtype=torch.complex64)
    points = torch.tensor([1.0+1.0j, -1.0-1.0j, 3.0+3.0j], dtype=torch.complex64)
    noise_var = torch.tensor([0.1, 0.1, 0.1, 0.1])
    
    # Calculate min distances
    min_dist = demod._min_distance_to_points(y, points, noise_var)
    
    # Each symbol should be closest to its corresponding point
    assert min_dist.shape == y.shape
    
    # Test with different noise variances
    varying_noise = torch.tensor([0.1, 0.2, 0.3, 0.4])
    min_dist_varying = demod._min_distance_to_points(y, points, varying_noise)
    assert min_dist_varying.shape == y.shape
    
    # Higher noise variance should result in less negative distance
    assert min_dist_varying[0] < min_dist_varying[3]

def test_qam_soft_demodulation_with_tensor_noise_var(device):
    """Test QAM soft demodulation with tensor noise variance."""
    mod = QAMModulator(order=16).to(device)
    demod = QAMDemodulator(order=16).to(device)
    
    # Create random bits
    batch_size = 10
    n_symbols = 100
    bits_per_symbol = mod.bits_per_symbol
    original_bits = torch.randint(0, 2, (batch_size, n_symbols * bits_per_symbol), device=device).float()
    
    # Modulate
    symbols = mod(original_bits)
    
    # Add varying noise
    base_noise_var = 0.1
    # Create varying noise variance for each symbol
    noise_var = torch.ones(batch_size, n_symbols, device=device) * base_noise_var
    noise_var[:, ::10] = 0.2  # Every 10th symbol has higher noise
    
    # Generate noise
    noise = torch.complex(
        torch.randn_like(symbols.real) * torch.sqrt(noise_var),
        torch.randn_like(symbols.imag) * torch.sqrt(noise_var)
    )
    noisy_symbols = symbols + noise
    
    # Soft demodulation with tensor noise variance
    llrs = demod(noisy_symbols, noise_var=noise_var)
    
    # Check output shape
    assert llrs.shape == original_bits.shape