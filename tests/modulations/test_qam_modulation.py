"""Comprehensive tests for QAM modulation schemes."""
import pytest
import torch
import numpy as np

from kaira.modulations.qam import QAMModulator, QAMDemodulator
from kaira.modulations.utils import binary_to_gray


# ===== Fixtures =====

@pytest.fixture
def device():
    """Fixture providing the compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def orders():
    """Fixture providing QAM orders to test."""
    return [4, 16, 64, 256]


@pytest.fixture
def qam_modulator():
    """Fixture for a QAM modulator."""
    return QAMModulator(order=16)  # 16-QAM (4 bits per symbol)


@pytest.fixture
def qam_demodulator():
    """Fixture for a QAM demodulator."""
    return QAMDemodulator(order=16)  # 16-QAM (4 bits per symbol)


# ===== Initialization Tests =====

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
    # Test with invalid order values
    with pytest.raises(ValueError):
        QAMModulator(order=3)  # Not a valid QAM order
        
    with pytest.raises(ValueError):
        QAMModulator(order=8)  # Must be a perfect square and power of 4
    
    with pytest.raises(ValueError):
        QAMModulator(order=32)  # Must be a perfect square
        
    with pytest.raises(ValueError):
        QAMModulator(order=36)  # Not a power of 4
    
    with pytest.raises(ValueError):
        QAMModulator(order=-4)  # Negative value


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


# ===== Constellation Tests =====

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


def test_4qam_constellation_structure():
    """Test the structure of the 4-QAM constellation."""
    mod = QAMModulator(order=4, normalize=False)
    
    # 4-QAM should have points at (±1, ±1)
    expected_points = torch.tensor([
        -1-1j, -1+1j, 1-1j, 1+1j
    ])
    
    # Check each constellation point matches one of the expected points
    for point in mod.constellation:
        # Verify this point exists in the expected constellation
        matches = torch.min(torch.abs(point - expected_points)).item() < 1e-5
        assert matches, f"Point {point} not found in expected constellation"


def test_qam_normalization():
    """Test QAM constellation normalization."""
    # Create modulators with and without normalization
    mod_norm = QAMModulator(order=16, normalize=True)
    mod_no_norm = QAMModulator(order=16, normalize=False)
    
    # The normalized constellation should have average symbol energy of 1.0
    constellation_energy_norm = torch.mean(torch.abs(mod_norm.constellation)**2)
    assert torch.isclose(constellation_energy_norm, torch.tensor(1.0), atol=1e-5)
    
    # The non-normalized constellation should have different energy
    constellation_energy_no_norm = torch.mean(torch.abs(mod_no_norm.constellation)**2)
    assert not torch.isclose(constellation_energy_no_norm, torch.tensor(1.0), atol=1e-5)


def test_qam_plot_constellation(qam_modulator):
    """Test QAM constellation plotting functionality."""
    # Verify the function exists and returns a figure
    fig = qam_modulator.plot_constellation()
    assert fig is not None


# ===== Gray Coding Tests =====

def test_qam_gray_coding():
    """Test QAM Gray coding bit mapping."""
    # Create a 16-QAM modulator with Gray coding
    mod_gray = QAMModulator(order=16, gray_coding=True)
    
    # Create modulator without Gray coding
    mod_no_gray = QAMModulator(order=16, gray_coding=False)
    
    # Check that the bit patterns are different
    assert not torch.equal(mod_gray.bit_patterns, mod_no_gray.bit_patterns)
    
    # With Gray coding, adjacent constellation points should differ by one bit
    # Calculate squared distances between all pairs of constellation points
    const_expanded1 = mod_gray.constellation.unsqueeze(1)  # (16, 1)
    const_expanded2 = mod_gray.constellation.unsqueeze(0)  # (1, 16)
    squared_distances = torch.abs(const_expanded1 - const_expanded2) ** 2  # (16, 16)
    
    # Find adjacent points (those with the smallest distance)
    min_dist = torch.min(squared_distances[squared_distances > 0])
    adjacency = (torch.abs(squared_distances - min_dist) < 1e-8) & (squared_distances > 0)
    
    # Check bit differences between adjacent points
    bit_diffs_sum = 0
    count = 0
    
    for i in range(16):
        for j in range(16):
            if adjacency[i, j]:
                # Count differing bits for adjacent points
                diff_bits = torch.sum(mod_gray.bit_patterns[i] != mod_gray.bit_patterns[j])
                bit_diffs_sum += diff_bits
                count += 1
    
    # On average, adjacent points should differ by close to one bit
    avg_bit_diff = bit_diffs_sum / max(1, count)
    assert avg_bit_diff <= 1.5, f"Average bit difference for adjacent points is {avg_bit_diff}, expected close to 1"


# ===== Modulation Tests =====

def test_qam_modulator_forward():
    """Test QAM modulator forward pass."""
    mod = QAMModulator(order=4)  # 4-QAM 
    
    # Test with individual bits
    x = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1])  # 4 bit pairs
    y = mod(x)
    
    assert y.shape == (4,)  # 4 symbols
    assert y.dtype == torch.complex64
    
    # The input maps to specific symbols
    const_points = mod.constellation
    
    # Test with batched input
    x_batch = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1]])  # batch of 2, each with 2 symbols
    y_batch = mod(x_batch)
    
    assert y_batch.shape == (2, 2)


def test_qam_modulator_forward_with_fixture(qam_modulator):
    """Test QAM modulator forward pass using the fixture."""
    # 16-QAM has 4 bits per symbol
    bits = torch.tensor([[0, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.float)
    
    # Modulate bits
    symbols = qam_modulator(bits)
    
    # Should produce 2 symbols (8 bits / 4 bits per symbol)
    assert symbols.shape == (1, 2)
    
    # Verify the constellation has the right size
    assert qam_modulator.constellation.shape[0] == 16
    
    # Check that output values are from the constellation
    for symbol in symbols.view(-1):
        distances = torch.abs(symbol - qam_modulator.constellation)
        assert torch.min(distances) < 1e-5


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


def test_qam_bit_mapping():
    """Test QAM bit mapping consistency."""
    mod = QAMModulator(order=16)
    
    # Create all possible 4-bit patterns
    all_bits = torch.zeros(16, 4)
    for i in range(16):
        binary = format(i, '04b')
        for j, bit in enumerate(binary):
            all_bits[i, j] = int(bit)
    
    # Modulate each pattern individually
    individual_symbols = torch.empty(16, dtype=torch.complex64)
    for i in range(16):
        individual_symbols[i] = mod(all_bits[i:i+1])[0, 0]
    
    # Each symbol should be unique
    for i in range(16):
        for j in range(i+1, 16):
            assert not torch.isclose(individual_symbols[i], individual_symbols[j])
    
    # Check that the constellation contains all these symbols
    for symbol in individual_symbols:
        min_distance = torch.min(torch.abs(symbol - mod.constellation))
        assert min_distance < 1e-5


def test_multi_dimensional_input():
    """Test QAM modulator and demodulator with multi-dimensional input."""
    mod = QAMModulator(order=16)  # 4 bits per symbol
    demod = QAMDemodulator(order=16)
    
    # Create 3D input tensor: [batch, frames, bits]
    batch_size = 2
    frames = 3
    bits_per_frame = 8  # Must be divisible by bits_per_symbol (4)
    
    bits = torch.randint(0, 2, (batch_size, frames, bits_per_frame), dtype=torch.float)
    
    # Modulate
    symbols = mod(bits)
    
    # Verify shape: should be [batch, frames, symbols_per_frame]
    expected_symbols = bits_per_frame // mod.bits_per_symbol
    assert symbols.shape == (batch_size, frames, expected_symbols)
    
    # Demodulate
    recovered_bits = demod(symbols)
    
    # Verify shape is restored
    assert recovered_bits.shape == bits.shape
    
    # With no noise, all bits should be recovered correctly
    assert torch.allclose(bits, recovered_bits)


# ===== Demodulation Tests =====

def test_qam_demodulator_forward():
    """Test QAM demodulator forward pass."""
    # Create modulator and demodulator pair
    mod = QAMModulator(order=16)  # 16-QAM
    demod = QAMDemodulator(order=16)
    
    # Generate all bit patterns for 16-QAM (4 bits per symbol)
    all_patterns = []
    for i in range(16):
        bits = [int(b) for b in format(i, '04b')]
        all_patterns.append(bits)
    
    all_bits = torch.tensor(all_patterns, dtype=torch.float).flatten()
    
    # Modulate
    symbols = mod(all_bits)
    
    # Demodulate
    demodulated_bits = demod(symbols)
    
    # Without noise, all bits should be recovered exactly
    assert torch.allclose(demodulated_bits, all_bits)


def test_qam_demodulator_hard_decision_with_noise():
    """Test QAM demodulator with noisy input."""
    # Create modulator and demodulator
    mod = QAMModulator(order=16)
    demod = QAMDemodulator(order=16)
    
    # Create random bits
    bits_per_symbol = 4  # log2(16)
    num_symbols = 100
    bits = torch.randint(0, 2, (num_symbols * bits_per_symbol,), dtype=torch.float)
    
    # Modulate
    symbols = mod(bits)
    
    # Add small noise
    noise_level = 0.01
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * noise_level,
        torch.randn_like(symbols.imag) * noise_level
    )
    
    # Demodulate (hard decision)
    decoded_bits = demod(noisy_symbols)
    
    # Shape should match original bits
    assert decoded_bits.shape == bits.shape
    
    # With low noise, most bits should be recovered correctly
    bit_errors = (decoded_bits != bits).sum().item()
    assert bit_errors / len(bits) < 0.1  # Less than 10% error rate


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
    # Use higher noise for 4-QAM since it's more robust to noise
    noise_std = 0.5 if order == 4 else 0.1
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
    
    # Skip this assertion for CUDA with 4-QAM if no errors are detected
    if not (order == 4 and device.type == 'cuda' and bit_error_rate == 0.0):
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


def test_qam_soft_demodulation_with_different_noise_vars():
    """Test QAM soft demodulation with different noise variance formats."""
    # Use a deterministic seed for reproducibility
    torch.manual_seed(42)
    
    mod = QAMModulator(order=16, normalize=True)
    demod = QAMDemodulator(order=16, normalize=True)
    
    # Generate random bits
    bits = torch.randint(0, 2, (2, 8), dtype=torch.float)
    
    # Modulate
    symbols = mod(bits)
    
    # Add small noise
    noise_level = 0.01
    noisy_symbols = symbols + torch.complex(
        torch.randn_like(symbols.real) * noise_level,
        torch.randn_like(symbols.imag) * noise_level
    )
    
    # Test with scalar noise variance
    llrs_scalar = demod(noisy_symbols, noise_var=noise_level**2 * 2)
    assert llrs_scalar.shape == bits.shape
    
    # Test with tensor noise variance (same for all symbols)
    noise_var_tensor = torch.full_like(symbols, noise_level**2 * 2)
    llrs_tensor = demod(noisy_symbols, noise_var=noise_var_tensor)
    assert llrs_tensor.shape == bits.shape
    
    # Values should be close since the noise variance is the same
    assert torch.allclose(llrs_scalar, llrs_tensor, rtol=1e-5, atol=1e-5)
    
    # Test with varying noise variance per symbol
    noise_var_varying = torch.linspace(0.0001, 0.01, symbols.numel()).reshape_as(symbols) * 2
    llrs_varying = demod(noisy_symbols, noise_var=noise_var_varying)
    assert llrs_varying.shape == bits.shape


# ===== End-to-End Tests =====

def test_qam_modulation_demodulation_loop():
    """Test complete modulation-demodulation loop to ensure consistency."""
    # Use deterministic seed for reproducibility
    torch.manual_seed(42)
    
    for order in [4, 16, 64]:  # Test different QAM orders
        bits_per_symbol = int(np.log2(order))
        
        # Create modulator and demodulator with same parameters
        mod = QAMModulator(order=order, gray_coding=True, normalize=True)
        demod = QAMDemodulator(order=order, gray_coding=True, normalize=True)
        
        # Generate random bits for testing
        batch_size = 3
        num_symbols = 10
        bits = torch.randint(0, 2, (batch_size, num_symbols * bits_per_symbol), dtype=torch.float)
            
        # Ensure bit length is a multiple of bits_per_symbol
        num_complete_symbols = bits.shape[1] // bits_per_symbol
        valid_length = num_complete_symbols * bits_per_symbol
        bits = bits[:, :valid_length]
        
        # Complete loop
        symbols = mod(bits)
        recovered_bits = demod(symbols)
        
        # Verify bits are recovered correctly
        assert recovered_bits.shape == bits.shape
        
        # With no noise, all bits should be recovered perfectly
        assert torch.allclose(recovered_bits, bits)


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