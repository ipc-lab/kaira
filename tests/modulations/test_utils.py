"""Comprehensive tests for modulation utility functions."""

import pytest
import torch
import numpy as np
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


class TestModulationsUtilsComprehensive:
    """Comprehensive tests for modulation utility functions."""

    def test_binary_to_gray_edge_cases(self):
        """Test binary to gray conversion with edge cases."""
        # Test with zero and powers of 2
        assert binary_to_gray(0) == 0
        assert binary_to_gray(1) == 1
        assert binary_to_gray(2) == 3
        assert binary_to_gray(4) == 6
        assert binary_to_gray(8) == 12
        
        # Test with large numbers
        assert binary_to_gray(1023) == 1365
        assert binary_to_gray(2**16 - 1) == (2**16 - 1) ^ ((2**16 - 1) >> 1)
        
        # Test with negative numbers (should raise ValueError)
        with pytest.raises(ValueError):
            binary_to_gray(-1)

    def test_gray_to_binary_edge_cases(self):
        """Test gray to binary conversion with edge cases."""
        # Test with zero and simple cases
        assert gray_to_binary(0) == 0
        assert gray_to_binary(1) == 1
        assert gray_to_binary(3) == 2
        assert gray_to_binary(6) == 4
        assert gray_to_binary(12) == 8
        
        # Test with large numbers
        assert gray_to_binary(1365) == 1023
        
        # Test with negative numbers (should raise ValueError)
        with pytest.raises(ValueError):
            gray_to_binary(-1)

    def test_binary_gray_roundtrip(self):
        """Test round trip conversion between binary and gray."""
        for i in range(1000):
            gray = binary_to_gray(i)
            binary = gray_to_binary(gray)
            assert binary == i

    def test_binary_array_to_gray_edge_cases(self):
        """Test binary array to gray array conversion with edge cases."""
        # Test with empty tensor
        empty = torch.tensor([], dtype=torch.int64)
        assert torch.equal(binary_array_to_gray(empty), empty)
        
        # Test with tensor of zeros
        zeros = torch.zeros(5, dtype=torch.int64)
        assert torch.equal(binary_array_to_gray(zeros), zeros)
        
        # Test with ones
        ones = torch.ones(5, dtype=torch.int64)
        assert torch.equal(binary_array_to_gray(ones), ones)
        
        # Test with mixed values
        mixed = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4, 12, 13])
        assert torch.equal(binary_array_to_gray(mixed), expected)

    def test_gray_array_to_binary_edge_cases(self):
        """Test gray array to binary array conversion with edge cases."""
        # Test with empty tensor
        empty = torch.tensor([], dtype=torch.int64)
        assert torch.equal(gray_array_to_binary(empty), empty)
        
        # Test with zeros
        zeros = torch.zeros(5, dtype=torch.int64)
        assert torch.equal(gray_array_to_binary(zeros), zeros)
        
        # Test with ones
        ones = torch.ones(5, dtype=torch.int64)
        assert torch.equal(gray_array_to_binary(ones), ones)
        
        # Test with mixed values
        gray = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4, 12, 13])
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert torch.equal(gray_array_to_binary(gray), expected)

    def test_binary_gray_array_roundtrip(self):
        """Test round trip conversion between binary and gray arrays."""
        # Test with random integers
        for _ in range(5):
            original = torch.randint(0, 1000, (100,))
            gray = binary_array_to_gray(original)
            binary = gray_array_to_binary(gray)
            assert torch.equal(binary, original)

    def test_calculate_theoretical_ber(self):
        """Test theoretical BER calculation."""
        # Test BPSK in AWGN
        snr_db_range = torch.linspace(0, 20, 5)
        ber_bpsk = calculate_theoretical_ber(snr_db_range, 'bpsk')
        
        # BER should decrease as SNR increases
        assert torch.all(ber_bpsk[:-1] > ber_bpsk[1:])
        
        # Test QPSK in AWGN (should be same as BPSK for same Eb/N0)
        ber_qpsk = calculate_theoretical_ber(snr_db_range, 'qpsk')
        assert torch.allclose(ber_bpsk, ber_qpsk)
        
        # Test 16QAM
        ber_16qam = calculate_theoretical_ber(snr_db_range, '16qam')
        # 16QAM should have worse BER than BPSK at same SNR
        assert torch.all(ber_16qam > ber_bpsk)
        
        # Test 64QAM
        ber_64qam = calculate_theoretical_ber(snr_db_range, '64qam')
        # 64QAM should have worse BER than 16QAM at same SNR
        assert torch.all(ber_64qam > ber_16qam)
        
        # Test with invalid modulation
        with pytest.raises(ValueError):
            calculate_theoretical_ber(snr_db_range, 'invalid_mod')

    def test_calculate_spectral_efficiency(self):
        """Test spectral efficiency calculation."""
        # Test for common modulation schemes
        assert calculate_spectral_efficiency("bpsk") == 1.0
        assert calculate_spectral_efficiency("qpsk") == 2.0
        assert calculate_spectral_efficiency("8psk") == 3.0
        assert calculate_spectral_efficiency("16qam") == 4.0
        assert calculate_spectral_efficiency("64qam") == 6.0
        assert calculate_spectral_efficiency("256qam") == 8.0
        
        # Test with coding rate
        assert calculate_spectral_efficiency("bpsk", coding_rate=0.5) == 0.5
        assert calculate_spectral_efficiency("qpsk", coding_rate=0.75) == 1.5
        assert calculate_spectral_efficiency("16qam", coding_rate=0.9) == 3.6
        
        # Test with invalid modulation
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("invalid_mod")
        
        # Test with invalid coding rate
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("bpsk", coding_rate=1.5)
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("bpsk", coding_rate=-0.1)

    def test_plot_constellation(self):
        """Test constellation plotting function."""
        # Create a simple constellation
        constellation = torch.tensor([1+0j, 0+1j, -1+0j, 0-1j])
        
        # Test with default parameters
        fig, ax = plot_constellation(constellation)
        plt.close(fig)  # Close the figure to avoid display
        
        # Test with labels
        labels = ["00", "01", "11", "10"]
        fig, ax = plot_constellation(constellation, labels=labels)
        plt.close(fig)
        
        # Test with title
        fig, ax = plot_constellation(constellation, title="Test Constellation")
        plt.close(fig)
        
        # Test with different marker size
        fig, ax = plot_constellation(constellation, marker_size=100)
        plt.close(fig)
        
        # Test with grid
        fig, ax = plot_constellation(constellation, grid=True)
        plt.close(fig)
        
        # Test with figsize
        fig, ax = plot_constellation(constellation, figsize=(10, 8))
        plt.close(fig)
        
        # Test with empty constellation
        empty_const = torch.tensor([])
        with pytest.raises(ValueError):
            plot_constellation(empty_const)