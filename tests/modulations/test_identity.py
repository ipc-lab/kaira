"""Comprehensive tests for the Identity modulation scheme."""

import pytest
import torch

from kaira.modulations import (
    IdentityModulator, 
    IdentityDemodulator
)


class TestIdentityModulationComprehensive:
    """Comprehensive tests for Identity modulation."""

    def test_identity_modulator_constructor(self):
        """Test Identity modulator constructor."""
        # Test default constructor
        mod = IdentityModulator()
        assert mod.bits_per_symbol == 1  # Default
        assert hasattr(mod, 'constellation')
        # Verify constellation is [0, 1]
        expected = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        assert torch.all(mod.constellation == expected)

    def test_identity_demodulator_constructor(self):
        """Test Identity demodulator constructor."""
        # Test default constructor
        demod = IdentityDemodulator()
        assert demod.bits_per_symbol == 1  # Default
        assert hasattr(demod, 'constellation')
        # Verify constellation is [0, 1]
        expected = torch.tensor([0.0, 1.0], dtype=torch.complex64)
        assert torch.all(demod.constellation == expected)

    def test_identity_modulator_forward(self):
        """Test identity modulator forward pass."""
        mod = IdentityModulator()
        
        # Test with various input shapes
        inputs = [
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])
        ]
        
        for x in inputs:
            y = mod(x)
            # Output shape should be the same as input
            assert y.shape == x.shape
            # Output values should be identical
            assert torch.all(y == x)

    def test_identity_demodulator_forward(self):
        """Test identity demodulator forward pass."""
        demod = IdentityDemodulator()
        
        # Test with various input shapes
        inputs = [
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])
        ]
        
        for x in inputs:
            y = demod(x)
            # Output shape should be the same as input
            assert y.shape == x.shape
            # Output values should be identical
            assert torch.all(y == x)

    def test_identity_modulator_demodulator_roundtrip(self):
        """Test roundtrip conversion with Identity modulation."""
        mod = IdentityModulator()
        demod = IdentityDemodulator()
        
        # Test with various input patterns
        inputs = [
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 1.0, 0.0]),
            torch.randn(10)  # Even non-binary values should pass through
        ]
        
        for x in inputs:
            # Forward through modulator
            modulated = mod(x)
            
            # Forward through demodulator
            demodulated = demod(modulated)
            
            # Roundtrip should preserve values
            assert torch.all(torch.isclose(demodulated, x))

    def test_identity_with_float_values(self):
        """Test Identity modulation with float values."""
        mod = IdentityModulator()
        demod = IdentityDemodulator()
        
        # Test with float values
        x = torch.tensor([0.1, 0.5, 0.9])
        
        # Modulate and demodulate
        modulated = mod(x)
        demodulated = demod(modulated)
        
        # Values should pass through unchanged
        assert torch.all(torch.isclose(demodulated, x))

    def test_identity_with_noise(self):
        """Test Identity demodulator with noise."""
        demod = IdentityDemodulator()
        
        # Original signal
        x = torch.tensor([0.0, 1.0, 0.0, 1.0])
        
        # Add noise
        noise = torch.randn_like(x) * 0.1
        noisy_x = x + noise
        
        # Test demodulation with noise_var parameter
        result_with_noise = demod(noisy_x, noise_var=0.1)
        
        # With Identity demodulator, the noise_var parameter is ignored
        # and the output should be the same as the input
        assert torch.all(torch.isclose(result_with_noise, noisy_x))

    def test_identity_plot_constellation(self):
        """Test constellation plotting for Identity modulator."""
        mod = IdentityModulator()
        
        # Test plot constellation method exists
        assert hasattr(mod, 'plot_constellation')
        
        # Calling plot_constellation should not error
        fig = mod.plot_constellation()
        assert fig is not None

    def test_identity_with_complex_values(self):
        """Test Identity modulation with complex values."""
        mod = IdentityModulator()
        demod = IdentityDemodulator()
        
        # Test with complex values
        x = torch.tensor([1+1j, 0+0j, -1-1j], dtype=torch.complex64)
        
        # Modulate and demodulate
        modulated = mod(x)
        demodulated = demod(modulated)
        
        # Values should pass through unchanged
        assert torch.all(torch.isclose(demodulated, x))

    def test_identity_with_direct_access_methods(self):
        """Test direct access to modulate/demodulate methods."""
        mod = IdentityModulator()
        demod = IdentityDemodulator()
        
        x = torch.tensor([0, 1, 0, 1])
        
        # Test direct modulate method
        modulated_direct = mod.modulate(x)
        assert torch.all(modulated_direct == x)
        
        # Test direct demodulate method
        demodulated_direct = demod.demodulate(modulated_direct)
        assert torch.all(demodulated_direct == x)
        
        # Test direct soft_demodulate method
        soft_demodulated = demod.soft_demodulate(modulated_direct, noise_var=0.1)
        assert torch.all(soft_demodulated == x)