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

"""Tests for modulation utility functions."""

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

from kaira.modulations.utils import (
    binary_array_to_gray,
    gray_array_to_binary,
    plot_constellation,
    calculate_theoretical_ber,
    calculate_spectral_efficiency,
)


class TestBinaryGrayConversion:
    """Tests for binary <-> Gray code conversion functions."""

    def test_empty_array_handling(self):
        """Test that empty arrays are handled correctly."""
        # Test empty array for binary_array_to_gray
        empty_list = []
        empty_np = np.array([])
        empty_tensor = torch.tensor([])

        # Test with different empty input types
        for empty_input in [empty_list, empty_np, empty_tensor]:
            result = binary_array_to_gray(empty_input)
            assert isinstance(result, torch.Tensor)
            assert result.numel() == 0
            assert result.shape == torch.Size([0])
        
        # Test empty array for gray_array_to_binary
        for empty_input in [empty_list, empty_np, empty_tensor]:
            result = gray_array_to_binary(empty_input)
            assert isinstance(result, torch.Tensor)
            assert result.numel() == 0
            assert result.shape == torch.Size([0])
    
    def test_device_and_dtype_preservation(self):
        """Test that empty arrays preserve device and dtype."""
        # Only run if CUDA is available
        if torch.cuda.is_available():
            # Create empty tensor with specific device and dtype
            empty_tensor = torch.tensor([], dtype=torch.float32, device="cuda")
            
            # Check binary to gray
            result = binary_array_to_gray(empty_tensor)
            assert result.device.type == "cuda"
            assert result.dtype == torch.float32
            
            # Check gray to binary
            result = gray_array_to_binary(empty_tensor)
            assert result.device.type == "cuda"
            assert result.dtype == torch.float32


class TestTheoreticalBER:
    """Tests for theoretical BER calculation functions."""
    
    def test_modulation_case_insensitivity(self):
        """Test that modulation scheme names are case-insensitive."""
        snr_db = 10.0
        
        # Test lowercase
        ber_lower = calculate_theoretical_ber(snr_db, "bpsk")
        # Test uppercase
        ber_upper = calculate_theoretical_ber(snr_db, "BPSK")
        # Test mixed case
        ber_mixed = calculate_theoretical_ber(snr_db, "BpSk")
        
        # All should be equal
        assert torch.isclose(ber_lower, ber_upper)
        assert torch.isclose(ber_lower, ber_mixed)
        
        # Test another modulation scheme
        ber_qpsk_lower = calculate_theoretical_ber(snr_db, "qpsk")
        ber_qpsk_upper = calculate_theoretical_ber(snr_db, "QPSK")
        assert torch.isclose(ber_qpsk_lower, ber_qpsk_upper)


class TestConstellationPlotting:
    """Tests for constellation diagram plotting functions."""
    
    def test_annotation_handling(self):
        """Test that constellation points are correctly annotated."""
        # Create a simple constellation
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        labels = ["00", "01", "11", "10"]  # QPSK Gray-coded labels
        
        # Create plot with annotations
        fig, ax = plot_constellation(
            constellation, 
            labels=labels, 
            annotate=True,
            title="Test Constellation"
        )
        
        # Verify annotations exist and have correct text
        annotations = [child for child in ax.get_children() if isinstance(child, Annotation)]
        assert len(annotations) == 4  # Should have 4 annotations
        
        # Get annotation texts
        texts = [a.get_text() for a in annotations]
        
        # Verify all labels are in the annotations
        for label in labels:
            assert label in texts
        
        # Test with fewer labels than points
        short_labels = ["00", "01"]
        fig, ax = plot_constellation(
            constellation, 
            labels=short_labels, 
            annotate=True
        )
        
        # Should use index numbers for missing labels
        annotations = [child for child in ax.get_children() if isinstance(child, Annotation)]
        texts = [a.get_text() for a in annotations]
        assert "00" in texts
        assert "01" in texts
        assert "2" in texts  # Index for the third point
        assert "3" in texts  # Index for the fourth point
        
        # Clean up
        plt.close(fig)
        
    def test_no_annotation(self):
        """Test that no annotations are added when annotate=False."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        labels = ["00", "01", "11", "10"]
        
        # Create plot with annotations disabled
        fig, ax = plot_constellation(
            constellation, 
            labels=labels, 
            annotate=False
        )
        
        # Verify no annotations exist
        annotations = [child for child in ax.get_children() if isinstance(child, Annotation)]
        assert len(annotations) == 0
        
        # Clean up
        plt.close(fig)


class TestSpectralEfficiency:
    """Tests for spectral efficiency calculation functions."""
    
    def test_auto_order_extraction(self):
        """Test automatic extraction of order from modulation scheme names."""
        # Test standard cases
        assert calculate_spectral_efficiency("16QAM") == 4.0
        assert calculate_spectral_efficiency("64PSK") == 6.0
        assert calculate_spectral_efficiency("32PAM") == 5.0
        
        # Test with different case
        assert calculate_spectral_efficiency("16qam") == 4.0
        assert calculate_spectral_efficiency("64psk") == 6.0
        assert calculate_spectral_efficiency("32pam") == 5.0
        
        # Test with unusual formatting
        assert calculate_spectral_efficiency("QAM16") == 4.0
        assert calculate_spectral_efficiency("psk-64") == 6.0
        assert calculate_spectral_efficiency("PAM_32") == 5.0
        
        # Test with coding rate
        assert calculate_spectral_efficiency("16QAM", coding_rate=0.5) == 2.0
        assert calculate_spectral_efficiency("64PSK", coding_rate=0.75) == 4.5
    
    def test_extraction_error(self):
        """Test that ValueError is raised for undefined schemes."""
        # Test with invalid modulation scheme
        with pytest.raises(ValueError, match="Spectral efficiency for 'invalid' not defined"):
            calculate_spectral_efficiency("invalid")
        
        # Test with non-standard format that can't be parsed
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("QAMX")
    
    def test_auto_order_extraction_comprehensive(self):
        """Test automatic extraction of order from various modulation scheme name formats."""
        # Standard formats with different schemes
        assert calculate_spectral_efficiency("16QAM") == 4.0
        assert calculate_spectral_efficiency("64PSK") == 6.0
        assert calculate_spectral_efficiency("32PAM") == 5.0
        
        # Mixed case variations
        assert calculate_spectral_efficiency("16qam") == 4.0
        assert calculate_spectral_efficiency("64pSk") == 6.0
        assert calculate_spectral_efficiency("32Pam") == 5.0
        
        # Non-standard formats with scheme first
        assert calculate_spectral_efficiency("QAM16") == 4.0
        assert calculate_spectral_efficiency("PSK64") == 6.0
        assert calculate_spectral_efficiency("PAM32") == 5.0
        
        # Formats with separators
        assert calculate_spectral_efficiency("QAM-16") == 4.0
        assert calculate_spectral_efficiency("PSK_64") == 6.0
        assert calculate_spectral_efficiency("32-PAM") == 5.0
        
        # Unusual power-of-2 orders
        assert calculate_spectral_efficiency("128QAM") == 7.0
        assert calculate_spectral_efficiency("512PSK") == 9.0
        
        # With coding rates
        assert calculate_spectral_efficiency("16QAM", coding_rate=0.5) == 2.0
        assert calculate_spectral_efficiency("PSK32", coding_rate=0.75) == 3.75
        
        # Verify non-power-of-2 orders
        assert calculate_spectral_efficiency("24QAM") == pytest.approx(np.log2(24))
        assert calculate_spectral_efficiency("10PSK") == pytest.approx(np.log2(10))
        
        # Test multi-digit orders
        assert calculate_spectral_efficiency("1024QAM") == 10.0
        assert calculate_spectral_efficiency("4096PSK") == 12.0
