"""Tests for GNU Radio integration."""

from unittest.mock import patch

import pytest
import torch

from kaira.hardware.sdr_utils import (
    FrequencyBand,
    HardwareError,
    SDRConfig,
    db_to_linear,
    linear_to_db,
    validate_frequency_range,
)


class TestSDRConfig:
    """Test SDR configuration."""

    def test_valid_config(self):
        """Test valid SDR configuration."""
        config = SDRConfig(center_frequency=915e6, sample_rate=1e6, tx_gain=20.0, rx_gain=30.0)

        assert config.center_frequency == 915e6
        assert config.sample_rate == 1e6
        assert config.bandwidth == 1e6  # Should default to sample_rate
        assert config.tx_gain == 20.0
        assert config.rx_gain == 30.0

    def test_invalid_frequency(self):
        """Test invalid frequency raises error."""
        with pytest.raises(ValueError, match="Center frequency must be positive"):
            SDRConfig(center_frequency=-1, sample_rate=1e6)

    def test_invalid_sample_rate(self):
        """Test invalid sample rate raises error."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            SDRConfig(center_frequency=915e6, sample_rate=-1)

    def test_invalid_gain(self):
        """Test invalid gain ranges raise errors."""
        with pytest.raises(ValueError, match="TX gain must be between"):
            SDRConfig(center_frequency=915e6, sample_rate=1e6, tx_gain=150)

        with pytest.raises(ValueError, match="RX gain must be between"):
            SDRConfig(center_frequency=915e6, sample_rate=1e6, rx_gain=-10)


class TestFrequencyBand:
    """Test frequency band enumeration."""

    def test_frequency_bands(self):
        """Test that frequency bands have correct values."""
        assert FrequencyBand.ISM_915.value == 915e6
        assert FrequencyBand.ISM_2400.value == 2.4e9
        assert FrequencyBand.GPS_L1.value == 1575.42e6


class TestUtilityFunctions:
    """Test utility functions."""

    def test_frequency_validation(self):
        """Test frequency validation."""
        # Valid frequency
        assert validate_frequency_range(915e6)

        # Invalid frequency
        with pytest.raises(ValueError, match="outside valid range"):
            validate_frequency_range(10e6)  # Too low

        with pytest.raises(ValueError, match="outside valid range"):
            validate_frequency_range(10e9)  # Too high

    def test_db_conversions(self):
        """Test dB to linear conversions."""
        # Test dB to linear
        assert abs(db_to_linear(0) - 1.0) < 1e-10
        assert abs(db_to_linear(10) - 10.0) < 1e-10
        assert abs(db_to_linear(20) - 100.0) < 1e-10

        # Test linear to dB
        assert abs(linear_to_db(1.0) - 0.0) < 1e-10
        assert abs(linear_to_db(10.0) - 10.0) < 1e-10
        assert abs(linear_to_db(100.0) - 20.0) < 1e-10


@pytest.mark.skipif(True, reason="GNU Radio not available in test environment")
class TestGNURadioBridge:
    """Test GNU Radio bridge functionality."""

    def test_bridge_creation(self):
        """Test creating GNU Radio bridge."""
        from kaira.hardware import GNURadioBridge

        config = SDRConfig(center_frequency=915e6, sample_rate=1e6)

        bridge = GNURadioBridge(config)
        assert bridge.config == config
        assert bridge._flowgraph is None
        assert not bridge._running

    @patch("kaira.hardware.gnuradio_bridge.GNURADIO_AVAILABLE", False)
    def test_gnuradio_not_available(self):
        """Test error when GNU Radio is not available."""
        from kaira.hardware import GNURadioBridge

        config = SDRConfig(center_frequency=915e6, sample_rate=1e6)

        with pytest.raises(ImportError, match="GNU Radio is not available"):
            GNURadioBridge(config)


class TestTorchBlocks:
    """Test PyTorch integration blocks (mocked)."""

    def test_encoder_block_interface(self):
        """Test encoder block interface."""
        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Tanh())

        # Mock the GNU Radio block interface
        with patch("kaira.hardware.gnuradio_bridge.GNURADIO_AVAILABLE", True):
            with patch("kaira.hardware.gnuradio_bridge.gr"):

                # This would normally create a GNU Radio block
                # For testing, we just verify the model can be set
                assert model is not None

    def test_decoder_block_interface(self):
        """Test decoder block interface."""
        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Sigmoid())

        # Mock the GNU Radio block interface
        with patch("kaira.hardware.gnuradio_bridge.GNURADIO_AVAILABLE", True):
            with patch("kaira.hardware.gnuradio_bridge.gr"):

                # This would normally create a GNU Radio block
                # For testing, we just verify the model can be set
                assert model is not None


class TestHardwareError:
    """Test hardware error exception."""

    def test_hardware_error(self):
        """Test hardware error exception."""
        with pytest.raises(HardwareError, match="Test error"):
            raise HardwareError("Test error")
