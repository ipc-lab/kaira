"""Comprehensive tests for modulation utility functions."""
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt

from kaira.modulations.utils import (
    binary_to_gray,
    gray_to_binary,
    calculate_theoretical_ber,
    calculate_spectral_efficiency,
    plot_constellation
)


# ===== Binary/Gray Code Tests =====

class TestBinaryGrayConversion:
    """Tests for binary to Gray code and Gray to binary conversion utilities."""
    
    def test_gray_to_binary_conversion(self):
        """Test Gray to binary code conversion."""
        # Test with a few known values
        assert gray_to_binary(0) == 0
        assert gray_to_binary(1) == 1
        assert gray_to_binary(3) == 2
        assert gray_to_binary(2) == 3
        assert gray_to_binary(6) == 4
        assert gray_to_binary(7) == 5
        assert gray_to_binary(5) == 7
        assert gray_to_binary(4) == 6
    
    def test_binary_to_gray_conversion(self):
        """Test binary to Gray code conversion."""
        # Test with a few known values
        assert binary_to_gray(0) == 0
        assert binary_to_gray(1) == 1
        assert binary_to_gray(2) == 3
        assert binary_to_gray(3) == 2
        assert binary_to_gray(4) == 6
        assert binary_to_gray(5) == 7
        assert binary_to_gray(6) == 5
        assert binary_to_gray(7) == 4
    
    def test_roundtrip_consistency(self):
        """Test that binary to Gray and back preserves the original value."""
        # Test round trip for a range of values
        for i in range(1000):
            assert gray_to_binary(binary_to_gray(i)) == i
            assert binary_to_gray(gray_to_binary(i)) == i
    
    def test_gray_code_adjacency_property(self):
        """Test that adjacent Gray codes differ by only one bit."""
        # Get Gray code patterns for 4-bit values (0-15)
        n_bits = 4
        max_val = 2**n_bits - 1
        
        gray_codes = [binary_to_gray(i) for i in range(max_val + 1)]
        
        # Check that adjacent values differ by exactly one bit
        for i in range(max_val):
            # Count the differing bits between successive Gray codes
            diff = gray_codes[i] ^ gray_codes[i+1]
            # Count the number of 1s in the binary representation
            num_diff_bits = bin(diff).count('1')
            assert num_diff_bits == 1
        
        # Check that first and last values also differ by exactly one bit
        # (cyclic property of Gray codes)
        diff = gray_codes[0] ^ gray_codes[-1]
        num_diff_bits = bin(diff).count('1')
        assert num_diff_bits == 1


# ===== Plotting Tests =====

class TestPlotConstellation:
    """Tests for constellation plotting function."""

    def test_plot_constellation_basic(self):
        """Test basic constellation plotting."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_with_labels(self):
        """Test constellation plotting with labels."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        labels = ["00", "01", "11", "10"]
        fig = plot_constellation(constellation, labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_without_annotations(self):
        """Test constellation plotting without annotations."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation, annotate=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_without_grid(self):
        """Test constellation plotting without grid."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation, grid=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_without_axis_labels(self):
        """Test constellation plotting without axis labels."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation, axis_labels=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_with_custom_style(self):
        """Test constellation plotting with custom styling options."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(
            constellation,
            title="Custom Title",
            figsize=(6, 6),
            marker="x",
            marker_size=50,
            color="red",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_with_existing_axis(self):
        """Test constellation plotting with a provided axis."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig, ax = plt.subplots()
        fig2 = plot_constellation(constellation, ax=ax)
        assert fig is fig2  # Should return the same figure
        plt.close(fig)

    def test_plot_constellation_with_additional_kwargs(self):
        """Test constellation plotting with additional kwargs for scatter."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation, alpha=0.5, edgecolors="black")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ===== Theoretical BER Tests =====

class TestTheoreticalBER:
    """Tests for theoretical BER calculation functions."""
    
    def test_calculate_theoretical_ber_shape(self):
        """Test the shape of the output from calculate_theoretical_ber."""
        # Test with varying shapes of input
        snr_db = np.linspace(-10, 20, 31)  # 31 points
        ber = calculate_theoretical_ber("bpsk", snr_db)
        assert ber.shape == (31,)
        
        # Test 2D array (should be flattened or raise an error depending on implementation)
        snr_db_2d = snr_db.reshape((-1, 1))
        try:
            calculate_theoretical_ber("bpsk", snr_db_2d)
        except ValueError:
            # Either it works by flattening or it raises a ValueError, both are acceptable
            pass
    
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

    @pytest.mark.parametrize("modulation", [
        "bpsk", "qpsk", "4qam", "16qam", "64qam", "4pam", "8pam", "dpsk", "dbpsk", "dqpsk"
    ])
    def test_calculate_theoretical_ber_single_value(self, modulation):
        """Test calculating theoretical BER for a single SNR value."""
        ber = calculate_theoretical_ber(modulation, 10.0)
        assert isinstance(ber, np.ndarray)
        assert ber.shape == (1,)
        assert 0.0 <= ber[0] <= 1.0  # BER should be between 0 and 1

    @pytest.mark.parametrize("modulation", [
        "bpsk", "qpsk", "4qam", "16qam", "64qam", "4pam", "8pam", "dpsk", "dbpsk", "dqpsk"
    ])
    def test_calculate_theoretical_ber_multiple_values(self, modulation):
        """Test calculating theoretical BER for multiple SNR values."""
        snrs = [0.0, 5.0, 10.0, 15.0, 20.0]
        ber = calculate_theoretical_ber(modulation, snrs)
        assert isinstance(ber, np.ndarray)
        assert ber.shape == (5,)
        assert np.all((0.0 <= ber) & (ber <= 1.0))  # All BERs should be between 0 and 1

    @pytest.mark.parametrize("modulation", [
        "bpsk", "qpsk", "4qam", "16qam", "64qam", "4pam", "8pam", "dpsk", "dbpsk", "dqpsk"
    ])
    def test_calculate_theoretical_ber_torch_input(self, modulation):
        """Test calculating theoretical BER with PyTorch tensor input."""
        snrs = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0])
        ber = calculate_theoretical_ber(modulation, snrs)
        assert isinstance(ber, np.ndarray)
        assert ber.shape == (5,)
        assert np.all((0.0 <= ber) & (ber <= 1.0))

    @pytest.mark.parametrize("modulation", [
        "bpsk", "qpsk", "4qam", "16qam", "64qam", "4pam", "8pam", "dpsk", "dbpsk", "dqpsk"
    ])
    def test_calculate_theoretical_ber_numpy_input(self, modulation):
        """Test calculating theoretical BER with NumPy array input."""
        snrs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        ber = calculate_theoretical_ber(modulation, snrs)
        assert isinstance(ber, np.ndarray)
        assert ber.shape == (5,)
        assert np.all((0.0 <= ber) & (ber <= 1.0))

    def test_calculate_theoretical_ber_decreases_with_snr(self):
        """Test that BER decreases as SNR increases."""
        snrs = np.linspace(0, 20, 10)
        for modulation in ["bpsk", "qpsk", "16qam"]:
            ber = calculate_theoretical_ber(modulation, snrs)
            # Check that BER is monotonically decreasing
            assert np.all(np.diff(ber) <= 0)

    def test_calculate_theoretical_ber_invalid_modulation(self):
        """Test that an invalid modulation raises a ValueError."""
        with pytest.raises(ValueError):
            calculate_theoretical_ber("invalid_modulation", 10.0)

    def test_calculate_theoretical_ber_relative_performance(self):
        """Test that higher-order modulations have worse BER at the same SNR."""
        snr = 10.0
        ber_bpsk = calculate_theoretical_ber("bpsk", snr)
        ber_qpsk = calculate_theoretical_ber("qpsk", snr)
        ber_16qam = calculate_theoretical_ber("16qam", snr)
        ber_64qam = calculate_theoretical_ber("64qam", snr)
        
        # Higher-order modulations should have higher BER at the same SNR
        assert ber_bpsk <= ber_qpsk
        assert ber_qpsk <= ber_16qam
        assert ber_16qam <= ber_64qam


# ===== Spectral Efficiency Tests =====

class TestSpectralEfficiency:
    """Tests for spectral efficiency calculation functions."""

    @pytest.mark.parametrize("modulation,expected", [
        ("bpsk", 1.0),
        ("qpsk", 2.0),
        ("4qam", 2.0),
        ("pi4qpsk", 2.0),
        ("oqpsk", 2.0),
        ("dqpsk", 2.0),
        ("8psk", 3.0),
        ("16qam", 4.0),
        ("64qam", 6.0),
        ("256qam", 8.0),
        ("4pam", 2.0),
        ("8pam", 3.0),
        ("16pam", 4.0),
    ])
    def test_calculate_spectral_efficiency_known_modulations(self, modulation, expected):
        """Test calculating spectral efficiency for known modulation schemes."""
        efficiency = calculate_spectral_efficiency(modulation)
        assert efficiency == expected

    def test_calculate_spectral_efficiency_case_insensitive(self):
        """Test that spectral efficiency calculation is case-insensitive."""
        assert calculate_spectral_efficiency("BPSK") == calculate_spectral_efficiency("bpsk")
        assert calculate_spectral_efficiency("Qpsk") == calculate_spectral_efficiency("qpsk")
        assert calculate_spectral_efficiency("16QAM") == calculate_spectral_efficiency("16qam")

    def test_calculate_spectral_efficiency_auto_extract(self):
        """Test automatic extraction of order from modulation name."""
        assert calculate_spectral_efficiency("32qam") == 5.0
        assert calculate_spectral_efficiency("128qam") == 7.0
        assert calculate_spectral_efficiency("512qam") == 9.0
        assert calculate_spectral_efficiency("1024qam") == 10.0
        
        assert calculate_spectral_efficiency("16psk") == 4.0
        assert calculate_spectral_efficiency("32pam") == 5.0

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

    def test_calculate_spectral_efficiency_invalid_modulation(self):
        """Test that an invalid modulation raises a ValueError."""
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("invalid_modulation")