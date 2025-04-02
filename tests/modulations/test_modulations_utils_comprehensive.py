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


class TestBinaryGrayConversion:
    """Tests for binary to Gray code conversion and vice versa."""

    @pytest.mark.parametrize("binary,gray", [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 2),
        (4, 6),
        (5, 7),
        (6, 5),
        (7, 4),
        (8, 12),
        (9, 13),
        (10, 15),
        (11, 14),
        (15, 10),
        (16, 24),
        (31, 17),
        (127, 64),
        (255, 128),
    ])
    def test_binary_to_gray(self, binary, gray):
        """Test binary to Gray code conversion for specific values."""
        assert binary_to_gray(binary) == gray

    @pytest.mark.parametrize("gray,binary", [
        (0, 0),
        (1, 1),
        (3, 2),
        (2, 3),
        (6, 4),
        (7, 5),
        (5, 6),
        (4, 7),
        (12, 8),
        (13, 9),
        (15, 10),
        (14, 11),
        (10, 15),
        (24, 16),
        (17, 31),
        (64, 127),
        (128, 255),
    ])
    def test_gray_to_binary(self, gray, binary):
        """Test Gray code to binary conversion for specific values."""
        assert gray_to_binary(gray) == binary

    def test_binary_gray_roundtrip(self):
        """Test that converting binary to Gray and back returns the original."""
        for i in range(256):
            assert gray_to_binary(binary_to_gray(i)) == i

    def test_binary_array_to_gray_numpy(self):
        """Test converting a NumPy array of binary values to Gray code."""
        binary = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        expected = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        gray = binary_array_to_gray(binary)
        np.testing.assert_array_equal(gray, expected)

    def test_binary_array_to_gray_torch(self):
        """Test converting a PyTorch tensor of binary values to Gray code."""
        binary = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        expected = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        gray = binary_array_to_gray(binary)
        np.testing.assert_array_equal(gray, expected)

    def test_binary_array_to_gray_list(self):
        """Test converting a list of binary values to Gray code."""
        binary = [0, 1, 2, 3, 4, 5, 6, 7]
        expected = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        gray = binary_array_to_gray(binary)
        np.testing.assert_array_equal(gray, expected)

    def test_binary_array_to_gray_float(self):
        """Test converting float values (interpreted as ints) to Gray code."""
        binary = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        expected = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        gray = binary_array_to_gray(binary)
        np.testing.assert_array_equal(gray, expected)

    def test_gray_array_to_binary_numpy(self):
        """Test converting a NumPy array of Gray code values to binary."""
        gray = np.array([0, 1, 3, 2, 6, 7, 5, 4])
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        binary = gray_array_to_binary(gray)
        np.testing.assert_array_equal(binary, expected)

    def test_gray_array_to_binary_torch(self):
        """Test converting a PyTorch tensor of Gray code values to binary."""
        gray = torch.tensor([0, 1, 3, 2, 6, 7, 5, 4])
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        binary = gray_array_to_binary(gray)
        np.testing.assert_array_equal(binary, expected)

    def test_gray_array_to_binary_list(self):
        """Test converting a list of Gray code values to binary."""
        gray = [0, 1, 3, 2, 6, 7, 5, 4]
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        binary = gray_array_to_binary(gray)
        np.testing.assert_array_equal(binary, expected)

    def test_gray_array_to_binary_float(self):
        """Test converting float values (interpreted as ints) from Gray code to binary."""
        gray = np.array([0.0, 1.0, 3.0, 2.0, 6.0, 7.0, 5.0, 4.0])
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        binary = gray_array_to_binary(gray)
        np.testing.assert_array_equal(binary, expected)

    def test_gray_binary_array_roundtrip(self):
        """Test that converting binary array to Gray and back returns the original."""
        binary = np.arange(64)
        gray = binary_array_to_gray(binary)
        binary_again = gray_array_to_binary(gray)
        np.testing.assert_array_equal(binary, binary_again)


class TestPlotConstellation:
    """Tests for constellation plotting function."""

    def test_plot_constellation_basic(self):
        """Test basic constellation plotting functionality."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        fig = plot_constellation(constellation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_with_labels(self):
        """Test constellation plotting with point labels."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        labels = ["00", "01", "11", "10"]
        fig = plot_constellation(constellation, labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_constellation_without_annotations(self):
        """Test constellation plotting without annotations."""
        constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j])
        labels = ["00", "01", "11", "10"]
        fig = plot_constellation(constellation, labels=labels, annotate=False)
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


class TestTheoreticalBER:
    """Tests for theoretical BER calculation functions."""

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
        assert calculate_spectral_efficiency("16psk") == 4.0
        assert calculate_spectral_efficiency("32pam") == 5.0

    def test_calculate_spectral_efficiency_invalid_modulation(self):
        """Test that an invalid modulation raises a ValueError."""
        with pytest.raises(ValueError):
            calculate_spectral_efficiency("invalid_modulation")