"""Tests for the modulations/utils.py module."""
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

class TestBinaryToGray:
    """Test case for binary to Gray code conversion functions."""
    
    def test_binary_to_gray(self):
        """Test binary_to_gray function."""
        assert binary_to_gray(0) == 0
        assert binary_to_gray(1) == 1
        assert binary_to_gray(2) == 3
        assert binary_to_gray(3) == 2
        assert binary_to_gray(4) == 6
        assert binary_to_gray(5) == 7
        assert binary_to_gray(6) == 5
        assert binary_to_gray(7) == 4
        assert binary_to_gray(8) == 12
    
    def test_gray_to_binary(self):
        """Test gray_to_binary function."""
        assert gray_to_binary(0) == 0
        assert gray_to_binary(1) == 1
        assert gray_to_binary(3) == 2
        assert gray_to_binary(2) == 3
        assert gray_to_binary(6) == 4
        assert gray_to_binary(7) == 5
        assert gray_to_binary(5) == 6
        assert gray_to_binary(4) == 7
        assert gray_to_binary(12) == 8
    
    def test_binary_array_to_gray(self):
        """Test binary_array_to_gray function with different input types."""
        # Test with list input
        binary_list = [0, 1, 2, 3, 4, 5, 6, 7]
        expected = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        np.testing.assert_array_equal(binary_array_to_gray(binary_list), expected)
        
        # Test with numpy array input
        binary_np = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(binary_array_to_gray(binary_np), expected)
        
        # Test with torch tensor input
        binary_torch = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(binary_array_to_gray(binary_torch), expected)
        
        # Test with float values
        binary_float = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        np.testing.assert_array_equal(binary_array_to_gray(binary_float), expected)
    
    def test_gray_array_to_binary(self):
        """Test gray_array_to_binary function with different input types."""
        # Test with list input
        gray_list = [0, 1, 3, 2, 6, 7, 5, 4]
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(gray_array_to_binary(gray_list), expected)
        
        # Test with numpy array input
        gray_np = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        np.testing.assert_array_equal(gray_array_to_binary(gray_np), expected)
        
        # Test with torch tensor input
        gray_torch = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4])
        np.testing.assert_array_equal(gray_array_to_binary(gray_torch), expected)
        
        # Test with float values
        gray_float = np.array([0.0, 1.0, 3.0, 2.0, 6.0, 7.0, 5.0, 4.0])
        np.testing.assert_array_equal(gray_array_to_binary(gray_float), expected)


class TestConstellationPlotting:
    """Test case for constellation plotting functions."""
    
    def test_plot_constellation(self):
        """Test the plot_constellation function."""
        # Create a simple constellation
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], dtype=torch.complex64)
        labels = ["00", "01", "11", "10"]
        
        # Basic test to ensure no errors
        fig = plot_constellation(constellation, labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with custom parameters
        fig = plot_constellation(
            constellation, 
            labels=labels, 
            title="Test Constellation",
            figsize=(6, 6), 
            annotate=True, 
            grid=False, 
            axis_labels=False,
            marker="x", 
            marker_size=50, 
            color="red"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with existing axis
        fig, ax = plt.subplots()
        fig2 = plot_constellation(constellation, labels=labels, ax=ax)
        assert fig is fig2  # Should be the same figure object
        plt.close(fig)


class TestTheoretical:
    """Test case for theoretical calculations."""
    
    def test_calculate_theoretical_ber(self):
        """Test the calculation of theoretical BER."""
        # Test with single SNR value
        snr_db = 10.0
        ber_bpsk = calculate_theoretical_ber("bpsk", snr_db)
        assert isinstance(ber_bpsk, np.ndarray)
        assert ber_bpsk.shape == (1,)
        assert 0 <= ber_bpsk[0] <= 1  # BER should be a probability
        
        # Test with array of SNR values
        snr_db_array = np.array([-10, 0, 10, 20])
        ber_qpsk = calculate_theoretical_ber("qpsk", snr_db_array)
        assert ber_qpsk.shape == (4,)
        assert np.all(0 <= ber_qpsk) and np.all(ber_qpsk <= 1)
        
        # Test with list of SNR values
        snr_db_list = [-10, 0, 10, 20]
        ber_16qam = calculate_theoretical_ber("16qam", snr_db_list)
        assert ber_16qam.shape == (4,)
        
        # Test with torch tensor
        snr_db_torch = torch.tensor([-10, 0, 10, 20])
        ber_64qam = calculate_theoretical_ber("64qam", snr_db_torch)
        assert ber_64qam.shape == (4,)
        
        # Test additional modulation schemes
        calculate_theoretical_ber("4pam", snr_db)
        calculate_theoretical_ber("8pam", snr_db)
        calculate_theoretical_ber("dpsk", snr_db)
        calculate_theoretical_ber("dqpsk", snr_db)
        
        # Test invalid modulation scheme
        with pytest.raises(ValueError):
            calculate_theoretical_ber("invalid_modulation", snr_db)
    
    def test_calculate_spectral_efficiency(self):
        """Test the calculation of spectral efficiency."""
        # Test exact values for common modulations
        assert calculate_spectral_efficiency("bpsk") == 1.0
        assert calculate_spectral_efficiency("qpsk") == 2.0
        assert calculate_spectral_efficiency("4qam") == 2.0
        assert calculate_spectral_efficiency("8psk") == 3.0
        assert calculate_spectral_efficiency("16qam") == 4.0
        assert calculate_spectral_efficiency("64qam") == 6.0
        assert calculate_spectral_efficiency("256qam") == 8.0
        assert calculate_spectral_efficiency("4pam") == 2.0
        assert calculate_spectral_efficiency("8pam") == 3.0
        assert calculate_spectral_efficiency("16pam") == 4.0
        
        # Test case insensitivity
        assert calculate_spectral_efficiency("BPSK") == 1.0
        assert calculate_spectral_efficiency("QPSK") == 2.0
        
        # Test auto-detection of order from name
        assert calculate_spectral_efficiency("32qam") == 5.0
        assert calculate_spectral_efficiency("128qam") == 7.0
        
        # Test invalid modulation scheme
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("invalid_modulation")