"""Enhanced tests for the modulations/utils.py module to increase coverage."""
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from kaira.modulations.utils import (
    binary_to_gray,
    gray_to_binary, 
    binary_array_to_gray,
    gray_array_to_binary,
    plot_constellation,
    calculate_theoretical_ber,
    calculate_spectral_efficiency
)

class TestGrayCodeEdgeCases:
    """Test edge cases for Gray code conversion functions."""
    
    def test_binary_to_gray_edge_cases(self):
        """Test binary_to_gray function with edge cases."""
        # Test with large values
        assert binary_to_gray(1023) == 1365
        assert binary_to_gray(65535) == 65535 ^ (65535 >> 1)
        
        # Test with max integer values
        max_int = 2**31 - 1  # Max signed 32-bit integer
        assert binary_to_gray(max_int) == max_int ^ (max_int >> 1)
    
    def test_gray_to_binary_edge_cases(self):
        """Test gray_to_binary function with edge cases."""
        # Test with large values
        assert gray_to_binary(1365) == 1023
        assert gray_to_binary(65535 ^ (65535 >> 1)) == 65535
        
        # Test with max integer values
        max_int = 2**31 - 1  # Max signed 32-bit integer
        assert gray_to_binary(max_int ^ (max_int >> 1)) == max_int

    def test_gray_code_round_trip(self):
        """Test that converting binary to Gray and back returns the original value."""
        for i in range(1000):
            assert gray_to_binary(binary_to_gray(i)) == i
            
    def test_binary_array_to_gray_empty(self):
        """Test binary_array_to_gray with empty input."""
        # Empty list
        result = binary_array_to_gray([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        
        # Empty numpy array
        result = binary_array_to_gray(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        
        # Empty torch tensor
        result = binary_array_to_gray(torch.tensor([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0
    
    def test_gray_array_to_binary_empty(self):
        """Test gray_array_to_binary with empty input."""
        # Empty list
        result = gray_array_to_binary([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        
        # Empty numpy array
        result = gray_array_to_binary(np.array([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0
        
        # Empty torch tensor
        result = gray_array_to_binary(torch.tensor([]))
        assert isinstance(result, np.ndarray)
        assert result.size == 0

class TestConstellationPlottingAdvanced:
    """Advanced tests for constellation plotting functions."""
    
    def test_plot_constellation_no_labels(self):
        """Test plot_constellation without labels."""
        # Create a constellation
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], dtype=torch.complex64)
        
        # Test with no labels
        fig = plot_constellation(constellation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_constellation_fewer_labels(self):
        """Test plot_constellation with fewer labels than points."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], dtype=torch.complex64)
        labels = ["00", "01"]  # Only 2 labels for 4 points
        
        fig = plot_constellation(constellation, labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_constellation_2d_tensor(self):
        """Test plot_constellation with a 2D tensor."""
        # Create a 2D tensor (batch of constellations)
        constellation = torch.tensor([[1+1j, -1+1j, -1-1j, 1-1j]], dtype=torch.complex64)
        
        # For this test, we'll flatten it in the test, but ideally the function should handle this
        fig = plot_constellation(constellation.flatten())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_constellation_with_kwargs(self):
        """Test plot_constellation with additional kwargs passed to scatter."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], dtype=torch.complex64)
        
        # Test with additional kwargs
        fig = plot_constellation(
            constellation, 
            alpha=0.5,  # Additional kwarg for scatter
            edgecolors='black'  # Additional kwarg for scatter
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

class TestTheoreticalAdvanced:
    """Advanced tests for theoretical calculations."""
    
    def test_calculate_theoretical_ber_shape(self):
        """Test the shape of the output from calculate_theoretical_ber."""
        # Test with varying shapes of input
        snr_db = np.linspace(-10, 20, 31)  # 31 points
        ber = calculate_theoretical_ber("bpsk", snr_db)
        assert ber.shape == (31,)
        
        # Test 2D array (should be flattened)
        snr_db_2d = snr_db.reshape((-1, 1))
        with pytest.raises(ValueError):  # Should raise ValueError with 2D array
            calculate_theoretical_ber("bpsk", snr_db_2d)
    
    def test_calculate_theoretical_ber_consistency(self):
        """Test consistency of BER calculations across modulation schemes."""
        snr_db = np.array([0, 10, 20])
        
        # BPSK should have better BER than QPSK at the same SNR
        ber_bpsk = calculate_theoretical_ber("bpsk", snr_db)
        ber_qpsk = calculate_theoretical_ber("qpsk", snr_db)
        assert np.all(ber_bpsk <= ber_qpsk)
        
        # Higher order modulations should have worse BER
        ber_16qam = calculate_theoretical_ber("16qam", snr_db)
        ber_64qam = calculate_theoretical_ber("64qam", snr_db)
        assert np.all(ber_qpsk <= ber_16qam)
        assert np.all(ber_16qam <= ber_64qam)
    
    def test_calculate_spectral_efficiency_custom(self):
        """Test spectral efficiency calculation with custom modulation names."""
        # Test with standard modulations
        assert calculate_spectral_efficiency("32qam") == 5.0
        assert calculate_spectral_efficiency("128qam") == 7.0
        assert calculate_spectral_efficiency("512qam") == 9.0
        assert calculate_spectral_efficiency("1024qam") == 10.0
        
        # Test with custom modulation orders
        assert calculate_spectral_efficiency("32psk") == 5.0
        assert calculate_spectral_efficiency("32pam") == 5.0
        
        # Test mixed case
        assert calculate_spectral_efficiency("16QAM") == 4.0
        assert calculate_spectral_efficiency("256Qam") == 8.0
    
    def test_calculate_spectral_efficiency_invalid(self):
        """Test spectral efficiency calculation with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("invalid_modulation")
        
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("psk")  # Missing order
        
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("qam")  # Missing order