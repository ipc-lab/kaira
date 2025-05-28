"""Tests for Error Vector Magnitude (EVM) metric."""

import numpy as np
import pytest
import torch

from kaira.metrics.signal.evm import EVM, ErrorVectorMagnitude


class TestEVM:
    """Test class for Error Vector Magnitude metric."""

    def test_evm_basic_functionality(self):
        """Test basic EVM calculation with simple complex signals."""
        metric = ErrorVectorMagnitude()

        # Simple test case: ideal vs noisy signal
        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=torch.complex64)
        received = torch.tensor([1.1 + 0.1j, 0.1 + 1.1j, -1.1 + 0.1j, 0.1 - 1.1j], dtype=torch.complex64)

        evm = metric.forward(transmitted, received)

        # Manual calculation for verification
        error_vector = received - transmitted
        error_power = torch.abs(error_vector) ** 2
        reference_power = torch.abs(transmitted) ** 2
        expected_evm = torch.sqrt(torch.mean(error_power / reference_power)) * 100

        assert torch.isclose(evm, expected_evm, rtol=1e-5), f"Expected {expected_evm}, got {evm}"

    def test_evm_perfect_signal(self):
        """Test EVM with perfect signal (no error)."""
        metric = ErrorVectorMagnitude()

        signal = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j], dtype=torch.complex64)
        evm = metric.forward(signal, signal)

        assert torch.isclose(evm, torch.tensor(0.0)), f"Perfect signal should have 0% EVM, got {evm}"

    def test_evm_different_modes(self):
        """Test EVM calculation with different modes (RMS, peak, percentile)."""
        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=torch.complex64)
        # Create received signal with varying error levels
        received = torch.tensor([1.1 + 0.05j, 0.05 + 1.2j, -1.05 + 0.05j, 0.05 - 1.05j], dtype=torch.complex64)

        # Test RMS mode
        metric_rms = ErrorVectorMagnitude(mode="rms")
        evm_rms = metric_rms.forward(transmitted, received)

        # Test Peak mode
        metric_peak = ErrorVectorMagnitude(mode="peak")
        evm_peak = metric_peak.forward(transmitted, received)

        # Test Percentile mode
        metric_percentile = ErrorVectorMagnitude(mode="percentile", percentile=95.0)
        evm_percentile = metric_percentile.forward(transmitted, received)

        # Peak EVM should be >= RMS EVM >= some percentile values
        assert evm_peak >= evm_rms, f"Peak EVM ({evm_peak}) should be >= RMS EVM ({evm_rms})"
        assert evm_percentile > 0, f"Percentile EVM should be > 0, got {evm_percentile}"

    def test_evm_normalization_modes(self):
        """Test EVM with and without normalization."""
        transmitted = torch.tensor([2.0 + 0.0j, 0.0 + 2.0j], dtype=torch.complex64)
        received = torch.tensor([2.2 + 0.2j, 0.2 + 2.2j], dtype=torch.complex64)

        # With normalization (default)
        metric_normalized = ErrorVectorMagnitude(normalize=True)
        evm_normalized = metric_normalized.forward(transmitted, received)

        # Without normalization
        metric_unnormalized = ErrorVectorMagnitude(normalize=False)
        evm_unnormalized = metric_unnormalized.forward(transmitted, received)

        # Both should be positive, but different values
        assert evm_normalized > 0
        assert evm_unnormalized > 0
        assert not torch.isclose(evm_normalized, evm_unnormalized)

    def test_evm_real_signals(self):
        """Test EVM with real-valued signals."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
        received = torch.tensor([1.1, -1.1, 0.9, -0.9], dtype=torch.float32)

        evm = metric.forward(transmitted, received)
        assert evm > 0, f"EVM should be positive for noisy signal, got {evm}"

    def test_evm_shape_mismatch(self):
        """Test that EVM raises error for mismatched input shapes."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
        received = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)

        with pytest.raises(ValueError, match="Input shapes must match"):
            metric.forward(transmitted, received)

    def test_evm_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Mode must be"):
            ErrorVectorMagnitude(mode="invalid")

    def test_evm_invalid_percentile(self):
        """Test that invalid percentile raises ValueError."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            ErrorVectorMagnitude(percentile=150)

        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            ErrorVectorMagnitude(percentile=0)

    def test_evm_per_symbol_calculation(self):
        """Test per-symbol EVM calculation."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j], dtype=torch.complex64)
        received = torch.tensor([1.1 + 0.1j, 0.1 + 1.1j, -1.1 + 0.1j], dtype=torch.complex64)

        per_symbol_evm = metric.calculate_per_symbol_evm(transmitted, received)

        # Should have same length as input
        assert per_symbol_evm.shape == transmitted.shape
        # All values should be positive
        assert torch.all(per_symbol_evm >= 0)

    def test_evm_statistics(self):
        """Test comprehensive EVM statistics calculation."""
        metric = ErrorVectorMagnitude()

        # Create signal with known properties
        np.random.seed(42)
        torch.manual_seed(42)

        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j] * 10, dtype=torch.complex64)
        noise = 0.1 * (torch.randn_like(transmitted, dtype=torch.float32) + 1j * torch.randn_like(transmitted, dtype=torch.float32))
        received = transmitted + noise

        stats = metric.calculate_statistics(transmitted, received)

        # Check that all expected keys are present
        expected_keys = ["evm_rms", "evm_mean", "evm_std", "evm_min", "evm_max", "evm_median", "evm_95th", "evm_99th", "evm_per_symbol"]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

        # Check that statistics make sense
        assert stats["evm_min"] <= stats["evm_median"] <= stats["evm_max"]
        assert stats["evm_mean"] > 0
        assert stats["evm_std"] >= 0
        assert stats["evm_95th"] >= stats["evm_median"]
        assert stats["evm_99th"] >= stats["evm_95th"]

    def test_evm_batched_signals(self):
        """Test EVM with batched signals."""
        metric = ErrorVectorMagnitude()

        # Create batched signals [batch_size, sequence_length]
        batch_size, seq_len = 3, 4
        transmitted = torch.randn(batch_size, seq_len, dtype=torch.complex64)
        received = transmitted + 0.1 * torch.randn_like(transmitted)

        evm = metric.forward(transmitted, received)
        assert evm > 0, f"Batched EVM should be positive, got {evm}"

    def test_evm_zero_reference_signal(self):
        """Test EVM behavior with zero reference signal."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.zeros(4, dtype=torch.complex64)
        received = torch.tensor([0.1 + 0.1j, 0.0, 0.2, 0.0], dtype=torch.complex64)

        # Should handle zero reference gracefully due to clamping
        evm = metric.forward(transmitted, received)
        assert not torch.isnan(evm), "EVM should not be NaN with zero reference"
        assert not torch.isinf(evm), "EVM should not be infinite with zero reference"

    def test_evm_edge_cases(self):
        """Test EVM with edge cases."""
        metric = ErrorVectorMagnitude()

        # Empty tensors
        empty_transmitted = torch.tensor([], dtype=torch.complex64)
        empty_received = torch.tensor([], dtype=torch.complex64)

        # Should return 0.0 for empty tensors
        evm_empty = metric.forward(empty_transmitted, empty_received)
        assert torch.isclose(evm_empty, torch.tensor(0.0)), f"Empty tensors should return 0.0 EVM, got {evm_empty}"

    def test_evm_very_small_signals(self):
        """Test EVM with very small signal amplitudes."""
        metric = ErrorVectorMagnitude()

        # Very small signals
        transmitted = torch.tensor([1e-6 + 1e-6j, 1e-6 - 1e-6j], dtype=torch.complex64)
        received = transmitted + 1e-7 * torch.randn_like(transmitted)

        evm = metric.forward(transmitted, received)
        assert not torch.isnan(evm), "EVM should not be NaN with very small signals"
        assert not torch.isinf(evm), "EVM should not be infinite with very small signals"

    def test_evm_large_signals(self):
        """Test EVM with large signal amplitudes."""
        metric = ErrorVectorMagnitude()

        # Large signals
        transmitted = torch.tensor([1000 + 1000j, 1000 - 1000j], dtype=torch.complex64)
        received = transmitted + 10 * torch.randn_like(transmitted)

        evm = metric.forward(transmitted, received)
        assert evm > 0, f"EVM should be positive with large signals, got {evm}"
        assert evm < 100, f"EVM should be reasonable even with large signals, got {evm}"

    def test_evm_alias(self):
        """Test that EVM alias works correctly."""
        metric1 = ErrorVectorMagnitude()
        metric2 = EVM()

        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex64)
        received = torch.tensor([1.1 + 0.1j, 0.1 + 1.1j], dtype=torch.complex64)

        evm1 = metric1.forward(transmitted, received)
        evm2 = metric2.forward(transmitted, received)

        assert torch.isclose(evm1, evm2), "EVM alias should produce same results"

    def test_evm_metric_properties(self):
        """Test EVM metric properties."""
        metric = ErrorVectorMagnitude()

        # Check metric properties
        assert metric.is_differentiable
        assert not metric.higher_is_better
        assert metric.name == "EVM"

    def test_evm_custom_name(self):
        """Test EVM with custom name."""
        custom_name = "Custom_EVM"
        metric = ErrorVectorMagnitude(name=custom_name)
        assert metric.name == custom_name

    def test_evm_percentile_modes(self):
        """Test different percentile values."""
        transmitted = torch.randn(100, dtype=torch.complex64)
        received = transmitted + 0.1 * torch.randn_like(transmitted)

        # Test different percentiles
        percentiles = [50, 90, 95, 99]
        evms = []

        for p in percentiles:
            metric = ErrorVectorMagnitude(mode="percentile", percentile=p)
            evm = metric.forward(transmitted, received)
            evms.append(evm)

        # Higher percentiles should generally give higher EVM values
        for i in range(len(evms) - 1):
            assert evms[i + 1] >= evms[i], f"EVM should increase with percentile: {percentiles[i]}%={evms[i]}, {percentiles[i+1]}%={evms[i+1]}"

    def test_evm_multidimensional_signals(self):
        """Test EVM with multi-dimensional signal arrays."""
        metric = ErrorVectorMagnitude()

        # Test with 2D signals (e.g., MIMO or multi-channel)
        transmitted = torch.randn(4, 8, dtype=torch.complex64)  # 4 channels, 8 symbols each
        received = transmitted + 0.05 * torch.randn_like(transmitted)

        evm = metric.forward(transmitted, received)
        assert evm > 0, f"Multi-dimensional EVM should be positive, got {evm}"

        # Test per-symbol EVM shape preservation
        per_symbol_evm = metric.calculate_per_symbol_evm(transmitted, received)
        assert per_symbol_evm.shape == transmitted.shape

    def test_evm_numerical_stability(self):
        """Test EVM numerical stability with extreme values."""
        metric = ErrorVectorMagnitude()

        # Test with very large error
        transmitted = torch.tensor([1.0 + 0.0j], dtype=torch.complex64)
        received = torch.tensor([100.0 + 100.0j], dtype=torch.complex64)

        evm = metric.forward(transmitted, received)
        assert not torch.isnan(evm), "EVM should not be NaN with large errors"
        assert not torch.isinf(evm), "EVM should not be infinite with large errors"
        assert evm > 100, f"Large error should produce high EVM, got {evm}"

    def test_evm_consistent_across_calls(self):
        """Test that EVM is consistent across multiple calls with same input."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j], dtype=torch.complex64)
        received = torch.tensor([1.1 + 0.1j, 0.1 + 1.1j, -1.1 + 0.1j], dtype=torch.complex64)

        evm1 = metric.forward(transmitted, received)
        evm2 = metric.forward(transmitted, received)

        assert torch.isclose(evm1, evm2), f"EVM should be consistent: {evm1} vs {evm2}"


class TestEVMIntegration:
    """Integration tests for EVM with realistic communication scenarios."""

    def test_evm_qam_constellation(self):
        """Test EVM with QAM constellation points."""
        metric = ErrorVectorMagnitude()

        # 16-QAM constellation points (simplified)
        qam16_points = torch.tensor([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j, 1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, 3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j], dtype=torch.complex64)

        # Add AWGN noise
        noise_std = 0.1
        noise = noise_std * (torch.randn_like(qam16_points, dtype=torch.float32) + 1j * torch.randn_like(qam16_points, dtype=torch.float32))
        received = qam16_points + noise

        evm = metric.forward(qam16_points, received)

        # For AWGN with std=0.1 on normalized constellation, EVM should be reasonable
        assert 0 < evm < 50, f"QAM EVM should be reasonable, got {evm}%"

    def test_evm_with_phase_noise(self):
        """Test EVM with phase noise."""
        metric = ErrorVectorMagnitude()

        # QPSK constellation
        qpsk_points = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64)
        qpsk_points = qpsk_points / torch.abs(qpsk_points[0])  # Normalize

        # Add phase noise
        phase_noise_std = 0.1  # radians
        phase_noise = phase_noise_std * torch.randn(len(qpsk_points))
        received = qpsk_points * torch.exp(1j * phase_noise)

        evm = metric.forward(qpsk_points, received)
        assert evm > 0, f"Phase noise should increase EVM, got {evm}%"

    def test_evm_with_amplitude_distortion(self):
        """Test EVM with amplitude distortion."""
        metric = ErrorVectorMagnitude()

        transmitted = torch.tensor([1.0, 0.5, 1.0, 0.5], dtype=torch.complex64)

        # Apply amplitude distortion (non-linear amplifier effect)
        amplitude = torch.abs(transmitted)
        distorted_amplitude = amplitude * (1 + 0.1 * amplitude**2)  # AM/AM distortion
        phase = torch.angle(transmitted)
        received = distorted_amplitude * torch.exp(1j * phase)

        evm = metric.forward(transmitted, received)
        assert evm > 0, f"Amplitude distortion should increase EVM, got {evm}%"
