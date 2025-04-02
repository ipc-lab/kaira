"""Functional tests for PSK modulation focusing on minimum distance properties."""
import numpy as np
import torch

from kaira.modulations.psk import PSKModulator, PSKDemodulator, QPSKModulator, QPSKDemodulator


def test_psk_constellation_distances():
    """Test that PSK constellation points have equal distances from origin."""
    for order in [4, 8, 16]:
        modulator = PSKModulator(order=order)
        
        # Get constellation points
        constellation = modulator.constellation
        
        # Calculate magnitudes (distances from origin)
        magnitudes = torch.abs(constellation)
        
        # All points should be at the same distance from origin (unit circle)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)


def test_psk_demodulation_finds_closest():
    """Test that PSK demodulation finds a close constellation point."""
    torch.manual_seed(42)  # For reproducibility
    
    for order in [4, 8, 16]:
        modulator = PSKModulator(order=order)
        demodulator = PSKDemodulator(order=order)
        
        # Get constellation points
        constellation = modulator.constellation
        
        # For each constellation point, test that demodulation works
        for i, point in enumerate(constellation):
            # Create a noisy version of the point (small noise)
            noisy_point = point + 0.05 * (torch.randn(()) + 1j * torch.randn(()))
            
            # Demodulate the noisy point
            demodulated_bits = demodulator(noisy_point.unsqueeze(0))
            
            # Remodulate the bits
            remodulated = modulator(demodulated_bits)
            
            # The remodulated point should be close to the original point
            # We're checking that the distance is within a reasonable threshold
            distance = torch.abs(remodulated.squeeze(0) - point)
            
            # Since constellation points are on the unit circle, a reasonable threshold 
            # would be less than the minimum distance between adjacent constellation points
            # For PSK with order M, this is approximately 2*sin(π/M)
            min_distance_between_points = 2 * np.sin(np.pi / order)
            
            # We expect the distance to be much smaller than this
            max_allowed_error = min_distance_between_points * 0.5
            
            assert distance < max_allowed_error


def test_qpsk_functional_correctness():
    """Test QPSK modulation focusing on the quadrant structure."""
    modulator = QPSKModulator()
    
    # Generate bit patterns for all QPSK symbols
    bits = torch.tensor([
        [0., 0.],  # First quadrant
        [0., 1.],  # Fourth quadrant
        [1., 0.],  # Second quadrant
        [1., 1.]   # Third quadrant
    ])
    
    # Flatten for modulation
    flat_bits = bits.flatten()
    
    # Modulate
    symbols = modulator(flat_bits)
    
    # Check symbol quadrants (based on sign of real and imaginary parts)
    quadrants = [
        (symbols[0].real > 0 and symbols[0].imag > 0),  # First quadrant
        (symbols[1].real > 0 and symbols[1].imag < 0),  # Fourth quadrant
        (symbols[2].real < 0 and symbols[2].imag > 0),  # Second quadrant
        (symbols[3].real < 0 and symbols[3].imag < 0)   # Third quadrant
    ]
    
    # All symbols should be in their expected quadrants
    assert all(quadrants)


def test_psk_bit_error_rate_with_noise():
    """Test that PSK bit error rate increases with noise level."""
    order = 8
    modulator = PSKModulator(order=order)
    demodulator = PSKDemodulator(order=order)
    
    # Generate random bits for multiple symbols
    bits_per_symbol = int(np.log2(order))
    num_symbols = 100
    bits = torch.randint(0, 2, (num_symbols * bits_per_symbol,), dtype=torch.float)
    
    # Modulate
    symbols = modulator(bits)
    
    # Test with different noise levels
    noise_levels = [0.01, 0.1, 0.5]
    error_rates = []
    
    for noise_level in noise_levels:
        # Add noise
        noisy_symbols = symbols + noise_level * (torch.randn_like(symbols.real) + 1j * torch.randn_like(symbols.imag))
        
        # Demodulate
        recovered_bits = demodulator(noisy_symbols)
        
        # Calculate bit error rate
        errors = (recovered_bits != bits).float().mean().item()
        error_rates.append(errors)
    
    # Error rate should increase with noise level
    assert error_rates[0] <= error_rates[1]
    assert error_rates[1] <= error_rates[2]