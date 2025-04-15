# tests/utils/test_utils.py
"""Consolidated tests for utility functions."""
import os
import random

import numpy as np
import pytest
import torch

from kaira.utils import (
    calculate_num_filters_factor_image,
    seed_everything,
    snr_db_to_linear,
    snr_linear_to_db,
    to_tensor,
)
from kaira.utils.snr import (
    add_noise_for_snr,
    calculate_snr,
    estimate_signal_power,
    noise_power_to_snr,
    snr_to_noise_power,
)

# ===== Fixtures =====


@pytest.fixture
def random_signal():
    """Fixture providing a random signal for testing."""
    torch.manual_seed(42)
    return torch.randn(100)


@pytest.fixture
def complex_signal():
    """Fixture providing a random complex signal for testing."""
    torch.manual_seed(42)
    real = torch.randn(100)
    imag = torch.randn(100)
    return torch.complex(real, imag)


# ===== Basic Utility Tests =====


class TestBasicUtils:
    """Tests for basic utility functions."""

    @pytest.mark.parametrize(
        "x, expected_type",
        [
            (torch.tensor([1, 2, 3]), torch.Tensor),
            (1, torch.Tensor),
            (1.0, torch.Tensor),
            ([1, 2, 3], torch.Tensor),
            (np.array([1, 2, 3]), torch.Tensor),
        ],
    )
    def test_to_tensor(self, x, expected_type):
        """Test to_tensor function with various inputs."""
        tensor = to_tensor(x)
        assert isinstance(tensor, expected_type)

    def test_to_tensor_device(self):
        """Test to_tensor function with device argument."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tensor = to_tensor([1, 2, 3], device=device)
            # Check the device type matches, not necessarily the exact device object
            assert tensor.device.type == "cuda"

    def test_to_tensor_with_mixed_dtype(self):
        """Test to_tensor function with mixed data types."""
        # Test with floating point values
        float_value = 3.14
        float_tensor = to_tensor(float_value)
        assert isinstance(float_tensor, torch.Tensor)
        assert float_tensor.dtype == torch.float32
        assert float_tensor.item() == pytest.approx(3.14)

        # Test with integer values
        int_value = 42
        int_tensor = to_tensor(int_value)
        assert isinstance(int_tensor, torch.Tensor)
        assert int_tensor.item() == 42

        # Test with mixed precision numpy array
        np_mixed = np.array([1, 2.5, 3])
        mixed_tensor = to_tensor(np_mixed)
        assert isinstance(mixed_tensor, torch.Tensor)
        assert mixed_tensor.dtype == torch.float64  # NumPy defaults to float64 for mixed arrays
        assert torch.allclose(mixed_tensor, torch.tensor([1.0, 2.5, 3.0], dtype=torch.float64))

    def test_to_tensor_error_cases(self):
        """Test to_tensor function error handling."""
        # Test with unsupported types
        with pytest.raises(TypeError):
            to_tensor({"key": "value"})

        with pytest.raises(TypeError):
            to_tensor(None)

        with pytest.raises(TypeError):
            to_tensor((1, 2, 3))  # Tuple is not directly supported

        with pytest.raises(TypeError):
            to_tensor("string")

    def test_to_tensor_device_handling(self):
        """Test that to_tensor correctly handles device specification."""
        # Test with CPU
        cpu_tensor = to_tensor([1, 2, 3], device="cpu")
        assert cpu_tensor.device.type == "cpu"

        # Test with existing tensor and device change
        existing = torch.tensor([1, 2, 3])
        assert existing.device.type == "cpu"  # Default is CPU

        # Only run GPU test if CUDA is available
        if torch.cuda.is_available():
            # Move existing tensor to GPU
            gpu_tensor = to_tensor(existing, device="cuda")
            assert gpu_tensor.device.type == "cuda"

            # Direct creation on GPU
            direct_gpu = to_tensor([4, 5, 6], device="cuda")
            assert direct_gpu.device.type == "cuda"

            # Test with device object instead of string
            device_obj = torch.device("cuda:0")
            device_obj_tensor = to_tensor([7, 8, 9], device=device_obj)
            assert device_obj_tensor.device.type == "cuda"

    @pytest.mark.parametrize("num_strided_layers, bw_ratio", [(1, 1.0), (2, 2.0)])
    def test_calculate_num_filters_factor_image(self, num_strided_layers, bw_ratio):
        """Test calculate_num_filters_factor_image function."""
        num_filters = calculate_num_filters_factor_image(num_strided_layers, bw_ratio, is_complex_transmission=True)
        assert isinstance(num_filters, int)
        expected_filters = 2 * 3 * (2 ** (2 * num_strided_layers)) * bw_ratio
        assert np.isclose(num_filters, expected_filters)

    def test_calculate_num_filters_with_params(self):
        """Test calculating number of filters for image processing with various parameters."""
        # Test with various bandwidth ratios
        assert calculate_num_filters_factor_image(1, 1.0) == 3 * 2**2  # RGB channels, no compression
        assert calculate_num_filters_factor_image(1, 0.5) == 6  # Half the base filters (12/2 = 6)
        assert calculate_num_filters_factor_image(1, 2.0) == 24  # Expansion by 2x

        # Test with grayscale (1 channel)
        assert calculate_num_filters_factor_image(1, 1.0, 1) == 2**2  # 1 channel, no compression

        # Test with different channel counts
        assert calculate_num_filters_factor_image(1, 1.0, 4) == 4 * 2**2  # 4 channels (RGBA)
        assert calculate_num_filters_factor_image(1, 0.25, 3) == 3  # RGB compressed to 1/4


# ===== SNR Utility Tests =====


class TestSNRUtils:
    """Tests for SNR utility functions."""

    @pytest.mark.parametrize("snr_linear", [1.0, 10.0, 100.0])
    def test_snr_linear_to_db(self, snr_linear):
        """Test snr_linear_to_db function."""
        snr_db = snr_linear_to_db(snr_linear)
        assert isinstance(snr_db, torch.Tensor)
        expected = torch.tensor(10 * np.log10(snr_linear), dtype=snr_db.dtype)
        assert torch.isclose(snr_db, expected)

    @pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
    def test_snr_db_to_linear(self, snr_db):
        """Test snr_db_to_linear function."""
        snr_linear = snr_db_to_linear(snr_db)
        assert isinstance(snr_linear, torch.Tensor)
        expected = 10 ** (snr_db / 10)
        assert torch.isclose(snr_linear, torch.tensor(expected, dtype=snr_linear.dtype))

    def test_snr_conversion(self):
        """Test SNR conversion between dB and linear scales."""
        # Test dB to linear
        assert snr_db_to_linear(0.0) == 1.0
        assert snr_db_to_linear(10.0) == pytest.approx(10.0)
        assert snr_db_to_linear(20.0) == pytest.approx(100.0)
        assert snr_db_to_linear(30.0) == pytest.approx(1000.0)

        # Test linear to dB
        assert snr_linear_to_db(1.0) == 0.0
        assert snr_linear_to_db(10.0) == pytest.approx(10.0)
        assert snr_linear_to_db(100.0) == pytest.approx(20.0)
        assert snr_linear_to_db(1000.0) == pytest.approx(30.0)

        # Test zero and negative cases
        assert snr_linear_to_db(0.0) == float("-inf")
        with pytest.raises(ValueError):
            snr_linear_to_db(-1.0)

    def test_snr_to_noise_power(self):
        """Test conversion from SNR to noise power."""
        # Test with various signal powers and SNRs
        signal_power = 2.0

        # At 0dB, noise power equals signal power
        assert snr_to_noise_power(signal_power, 0.0) == pytest.approx(2.0)

        # At 10dB, noise power is 10x less than signal
        assert snr_to_noise_power(signal_power, 10.0) == pytest.approx(0.2)

        # At -10dB, noise power is 10x more than signal
        assert snr_to_noise_power(signal_power, -10.0) == pytest.approx(20.0)

        # With complex signal
        complex_power = 4.0  # |2+2j|^2
        assert snr_to_noise_power(complex_power, 3.0) == pytest.approx(2.0, abs=0.01)  # Use absolute tolerance for better precision

    def test_noise_power_to_snr_float_inputs(self):
        """Test noise_power_to_snr with float inputs."""
        signal_power = 10.0
        noise_power = 1.0

        snr_db = noise_power_to_snr(signal_power, noise_power)

        assert isinstance(snr_db, torch.Tensor)
        assert torch.isclose(snr_db, torch.tensor(10.0), rtol=1e-5)

    def test_noise_power_to_snr_tensor_inputs(self):
        """Test noise_power_to_snr with tensor inputs."""
        signal_power = torch.tensor([10.0, 20.0, 30.0])
        noise_power = torch.tensor([1.0, 2.0, 3.0])

        snr_db = noise_power_to_snr(signal_power, noise_power)

        assert isinstance(snr_db, torch.Tensor)
        assert torch.allclose(snr_db, torch.tensor([10.0, 10.0, 10.0]), rtol=1e-5)

    def test_noise_power_to_snr_mixed_inputs(self):
        """Test noise_power_to_snr with mixed float and tensor inputs."""
        signal_power = 10.0
        noise_power = torch.tensor([1.0, 2.0, 5.0])

        snr_db = noise_power_to_snr(signal_power, noise_power)

        assert isinstance(snr_db, torch.Tensor)
        # Increase tolerance for floating-point comparison
        expected_values = torch.tensor([10.0, 7.0, 3.0])
        assert torch.allclose(snr_db, expected_values, rtol=1e-2)

    def test_noise_power_to_snr_zero_noise(self):
        """Test noise_power_to_snr raises error with zero noise power."""
        signal_power = 10.0
        noise_power = 0.0

        with pytest.raises(ValueError, match="Noise power cannot be zero"):
            noise_power_to_snr(signal_power, noise_power)

        # Also test with tensor containing zeros
        noise_power_tensor = torch.tensor([1.0, 0.0, 2.0])
        with pytest.raises(ValueError, match="Noise power cannot be zero"):
            noise_power_to_snr(signal_power, noise_power_tensor)

    def test_add_noise_for_snr_real_signal(self, random_signal):
        """Test add_noise_for_snr with real signal."""
        torch.manual_seed(42)  # For reproducibility
        signal = random_signal
        target_snr_db = 10.0  # 10 dB SNR

        # Add noise to achieve target SNR
        noisy_signal, noise = add_noise_for_snr(signal, target_snr_db)

        # Check shapes
        assert noisy_signal.shape == signal.shape
        assert noise.shape == signal.shape

        # Calculate achieved SNR
        signal_power = torch.mean(signal**2)
        noise_power = torch.mean(noise**2)
        achieved_snr_db = 10 * torch.log10(signal_power / noise_power)

        # Check that we're close to target SNR (allow some variance due to random noise)
        assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)

    def test_add_noise_for_snr_complex_signal(self, complex_signal):
        """Test add_noise_for_snr with complex signal."""
        torch.manual_seed(42)  # For reproducibility
        signal = complex_signal
        target_snr_db = 15.0  # 15 dB SNR

        # Add noise to achieve target SNR
        noisy_signal, noise = add_noise_for_snr(signal, target_snr_db)

        # Check shapes
        assert noisy_signal.shape == signal.shape
        assert noise.shape == signal.shape

        # Check that result is complex
        assert torch.is_complex(noisy_signal)
        assert torch.is_complex(noise)

        # Calculate achieved SNR
        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = torch.mean(torch.abs(noise) ** 2)
        achieved_snr_db = 10 * torch.log10(signal_power / noise_power)

        # Check that we're close to target SNR (allow some variance due to random noise)
        assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)

    def test_add_noise_for_snr_with_dimension(self):
        """Test add_noise_for_snr with specific dimension reduction."""
        torch.manual_seed(42)  # For reproducibility

        # Create a 2D test signal with different powers per channel
        signal = torch.zeros(3, 1000)
        signal[0, :] = 1.0  # Power 1.0
        signal[1, :] = 2.0  # Power 4.0
        signal[2, :] = 3.0  # Power 9.0

        target_snr_db = 20.0  # 20 dB SNR

        # Add noise to achieve target SNR, reducing along dimension 1
        noisy_signal, noise = add_noise_for_snr(signal, target_snr_db, dim=1)

        # Check that different noise powers were applied to each channel
        noise_power_per_channel = torch.mean(noise**2, dim=1)

        # Each channel should have a different noise power based on its signal power
        assert noise_power_per_channel[0] < noise_power_per_channel[1]
        assert noise_power_per_channel[1] < noise_power_per_channel[2]

        # Calculate achieved SNR per channel
        for i in range(3):
            signal_power = torch.mean(signal[i] ** 2)
            noise_power = torch.mean(noise[i] ** 2)
            achieved_snr_db = 10 * torch.log10(signal_power / noise_power)

            # Each channel should be close to target SNR
            assert torch.isclose(achieved_snr_db, torch.tensor(target_snr_db), rtol=0.1)

    def test_calculate_snr_real_signal(self):
        """Test calculate_snr with real signals."""
        # Create original signal
        original = torch.ones(100) * 2.0  # Power = 4.0

        # Create noise with known power
        noise = torch.ones(100) * 1.0  # Power = 1.0

        # Create noisy signal
        noisy = original + noise

        # Calculate SNR (should be 10*log10(4/1) = 6.02 dB)
        snr_db = calculate_snr(original, noisy)

        assert torch.isclose(snr_db, torch.tensor(6.02), rtol=1e-2)

    def test_calculate_snr_complex_signal(self):
        """Test calculate_snr with complex signals."""
        # Create original signal (1+1j throughout, power = 2)
        original = torch.complex(torch.ones(100), torch.ones(100))

        # Create noise with known power (0.5+0.5j throughout, power = 0.5)
        noise = torch.complex(torch.ones(100) * 0.5, torch.ones(100) * 0.5)

        # Create noisy signal
        noisy = original + noise

        # Calculate SNR (should be 10*log10(2/0.5) = 6.02 dB)
        snr_db = calculate_snr(original, noisy)

        assert torch.isclose(snr_db, torch.tensor(6.02), rtol=1e-2)

    def test_calculate_snr_with_dimensions(self):
        """Test calculate_snr with different dimension reductions."""
        # Create 2D signal with different powers per channel
        original = torch.zeros(3, 100)
        original[0, :] = 1.0  # Power = 1.0
        original[1, :] = 2.0  # Power = 4.0
        original[2, :] = 3.0  # Power = 9.0

        # Create noise with known power
        noise = torch.ones_like(original) * 0.5  # Power = 0.25

        # Create noisy signal
        noisy = original + noise

        # Calculate SNR for each channel
        snr_db = calculate_snr(original, noisy, dim=1)

        # Expected SNRs: 10*log10(1/0.25) = 6.02, 10*log10(4/0.25) = 12.04, 10*log10(9/0.25) = 15.56
        expected_snrs = torch.tensor([6.02, 12.04, 15.56])

        assert torch.allclose(snr_db, expected_snrs, rtol=1e-2)

        # Test with keepdim=True
        snr_db_keepdim = calculate_snr(original, noisy, dim=1, keepdim=True)
        assert snr_db_keepdim.shape == torch.Size([3, 1])

    def test_calculate_snr_error_case(self):
        """Test calculate_snr with different shaped inputs."""
        signal1 = torch.ones(100)
        signal2 = torch.ones(50)

        with pytest.raises(ValueError, match="Original and noisy signals must have the same shape"):
            calculate_snr(signal1, signal2)

    def test_estimate_signal_power_real_signal(self):
        """Test estimate_signal_power with real signal."""
        # Create a signal with known power
        signal = torch.ones(100) * 2.0  # Power should be 4.0

        # Calculate power
        power = estimate_signal_power(signal)

        assert isinstance(power, torch.Tensor)
        assert torch.isclose(power, torch.tensor(4.0))

        # Test with dimension reduction
        signal_2d = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
        power_per_dim = estimate_signal_power(signal_2d, dim=1)

        assert power_per_dim.shape == torch.Size([3])
        assert torch.allclose(power_per_dim, torch.tensor([1.0, 4.0, 9.0]))

        # Test with keepdim=True
        power_keepdim = estimate_signal_power(signal_2d, dim=1, keepdim=True)
        assert power_keepdim.shape == torch.Size([3, 1])

    def test_estimate_signal_power_complex_signal(self):
        """Test estimate_signal_power with complex signal."""
        # Create a complex signal with known power
        real_part = torch.ones(100)
        imag_part = torch.ones(100)
        signal = torch.complex(real_part, imag_part)  # |1+1j|^2 = 2, so power should be 2.0

        # Calculate power
        power = estimate_signal_power(signal)

        assert isinstance(power, torch.Tensor)
        assert torch.isclose(power, torch.tensor(2.0))

        # Test with varying amplitude complex signal
        real_part = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
        imag_part = torch.ones(3, 100) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
        complex_signal = torch.complex(real_part, imag_part)

        # Calculate power per first dimension with keepdim=True
        power_per_dim = estimate_signal_power(complex_signal, dim=1, keepdim=True)

        # For complex numbers |a+bj|^2 = a^2 + b^2
        # So for [1+1j, 2+2j, 3+3j] we expect powers [2, 8, 18]
        assert power_per_dim.shape == torch.Size([3, 1])
        assert torch.allclose(power_per_dim, torch.tensor([[2.0], [8.0], [18.0]]))

    def test_snr_db_to_linear_edge_cases(self):
        """Test edge cases for snr_db_to_linear."""
        # Test with extremely high dB value
        high_db = 100.0  # 10^10 in linear scale
        high_linear = snr_db_to_linear(high_db)
        assert torch.isclose(high_linear, torch.tensor(1e10), rtol=1e-5)

        # Test with negative dB value
        negative_db = -10.0  # 0.1 in linear scale
        negative_linear = snr_db_to_linear(negative_db)
        assert torch.isclose(negative_linear, torch.tensor(0.1), rtol=1e-5)

        # Test with zero dB value
        zero_db = 0.0  # 1.0 in linear scale
        zero_linear = snr_db_to_linear(zero_db)
        assert torch.isclose(zero_linear, torch.tensor(1.0), rtol=1e-5)

        # Test with tensor input
        tensor_db = torch.tensor([0.0, 10.0, 20.0])
        tensor_linear = snr_db_to_linear(tensor_db)
        assert torch.allclose(tensor_linear, torch.tensor([1.0, 10.0, 100.0]), rtol=1e-5)

    def test_snr_linear_to_db_edge_cases(self):
        """Test edge cases for snr_linear_to_db."""
        # Test with extremely high linear value
        high_linear = 1e10
        high_db = snr_linear_to_db(high_linear)
        assert torch.isclose(high_db, torch.tensor(100.0), rtol=1e-5)

        # Test with small linear value
        small_linear = 0.1
        small_db = snr_linear_to_db(small_linear)
        assert torch.isclose(small_db, torch.tensor(-10.0), rtol=1e-5)

        # Test with tensor input
        tensor_linear = torch.tensor([1.0, 10.0, 100.0])
        tensor_db = snr_linear_to_db(tensor_linear)
        assert torch.allclose(tensor_db, torch.tensor([0.0, 10.0, 20.0]), rtol=1e-5)

        # Test with negative value (should raise error)
        with pytest.raises(ValueError, match="SNR in linear scale must be positive"):
            snr_linear_to_db(-1.0)

        # Test with zero (should result in -inf)
        zero_db = snr_linear_to_db(0.0)
        assert torch.isinf(zero_db)
        assert zero_db < 0  # Negative infinity

    def test_complete_snr_pipeline(self):
        """Test a complete pipeline of SNR-related functions."""
        torch.manual_seed(42)  # For reproducibility

        # Start with a clean signal
        original_signal = torch.randn(5, 1000)

        # Target SNR
        target_snr_db = 15.0

        # Convert to linear
        target_snr_linear = snr_db_to_linear(target_snr_db)
        assert torch.isclose(snr_linear_to_db(target_snr_linear), torch.tensor(target_snr_db))

        # Add noise to achieve target SNR
        noisy_signal, _ = add_noise_for_snr(original_signal, target_snr_db)

        # Calculate achieved SNR
        measured_snr_db = calculate_snr(original_signal, noisy_signal)

        # Should be close to target (within reasonable tolerance due to randomness)
        assert torch.isclose(measured_snr_db, torch.tensor(target_snr_db), rtol=0.1)

        # Estimate signal power
        estimated_power = estimate_signal_power(original_signal)
        direct_power = torch.mean(original_signal**2)
        assert torch.isclose(estimated_power, direct_power)

        # Calculate required noise power for target SNR
        required_noise_power = snr_to_noise_power(estimated_power, target_snr_db)

        # Verify the relationship
        computed_snr = noise_power_to_snr(estimated_power, required_noise_power)
        assert torch.isclose(computed_snr, torch.tensor(target_snr_db))

    def test_noise_power_to_snr_scalar_expansion(self):
        """Test noise_power_to_snr properly expands scalar noise_power to match signal_power
        tensor."""
        # Create a multi-element signal power tensor
        signal_power = torch.tensor([2.0, 4.0, 8.0, 16.0])
        # Create a scalar noise power
        noise_power = 2.0

        # This should invoke the expansion logic when signal_power.numel() > 1
        snr_db = noise_power_to_snr(signal_power, noise_power)

        # Since noise_power is expanded to [2.0, 2.0, 2.0, 2.0],
        # the expected SNR values should be [0.0, 3.01, 6.02, 9.03] dB
        expected_snr = torch.tensor([0.0, 3.01, 6.02, 9.03])

        assert torch.allclose(snr_db, expected_snr, rtol=1e-2)

        # Also test with a scalar tensor
        noise_power_tensor = torch.tensor(2.0)
        snr_db_2 = noise_power_to_snr(signal_power, noise_power_tensor)

        # Results should be the same as with float scalar
        assert torch.allclose(snr_db_2, expected_snr, rtol=1e-2)

    def test_snr_to_noise_power_tensor_snr_db(self):
        """Test snr_to_noise_power with tensor snr_db input (tests type conversion branch)."""
        # Test with snr_db as a tensor (triggering the else branch for type conversion)
        signal_power = 10.0
        # Create a tensor for snr_db instead of a scalar
        snr_db_tensor = torch.tensor([0.0, 10.0, 20.0])

        noise_power = snr_to_noise_power(signal_power, snr_db_tensor)

        # Check that it's converted properly and calculations are accurate
        assert noise_power.dtype == torch.float32  # Final result should be float32
        expected_noise = torch.tensor([10.0, 1.0, 0.1])  # Signal power / 10^(snr_db/10)
        assert torch.allclose(noise_power, expected_noise, rtol=1e-5)

        # Verify the mixed precision handling worked correctly
        # by comparing with manual calculation using float64
        signal_power_64 = torch.tensor(10.0, dtype=torch.float64)
        snr_linear_64 = 10 ** (snr_db_tensor.to(torch.float64) / 10.0)
        expected_64 = (signal_power_64 / snr_linear_64).to(torch.float32)
        assert torch.allclose(noise_power, expected_64, rtol=1e-6)

    def test_snr_linear_to_db_tensor_with_zeros(self):
        """Test snr_linear_to_db with tensor containing zeros (tests zero-handling branch)."""
        # Create a tensor with multiple elements including zeros
        snr_linear = torch.tensor([0.0, 1.0, 10.0, 0.0, 100.0])

        # This should trigger the branch for handling tensors with zero elements
        snr_db = snr_linear_to_db(snr_linear)

        # Check results - zeros should become -inf, rest should be converted normally
        expected_result = torch.tensor([float("-inf"), 0.0, 10.0, float("-inf"), 20.0])

        # Separately check non-inf values
        non_inf_mask = ~torch.isinf(expected_result)
        assert torch.allclose(snr_db[non_inf_mask], expected_result[non_inf_mask])

        # Check that zeros were properly converted to -inf
        zero_mask = snr_linear == 0
        assert torch.all(torch.isinf(snr_db[zero_mask]))
        assert torch.all(snr_db[zero_mask] < 0)  # Confirm it's negative infinity

        # Also verify that the high-precision calculation path works correctly
        # by checking against manual computation
        manual_result = torch.empty_like(snr_linear)
        manual_result[zero_mask] = float("-inf")
        manual_result[~zero_mask] = 10 * torch.log10(snr_linear[~zero_mask])
        assert torch.all(torch.eq(snr_db, manual_result) | (torch.isnan(snr_db) & torch.isnan(manual_result)))


# ===== Seeding Utility Tests =====


class TestSeedUtils:
    """Tests for seeding utility functions."""

    def test_seed_everything_reproducibility(self):
        """Test that seed_everything makes randomization reproducible."""
        # Set a specific seed
        seed_value = 42
        seed_everything(seed_value)

        # Get random numbers from different generators
        random_py = random.random()  # nosec B311
        random_np = np.random.rand()
        random_torch = torch.rand(1).item()

        # Reset and seed again with the same value
        seed_everything(seed_value)

        # Check that we get the same values after re-seeding
        assert random_py == random.random()  # nosec B311
        assert random_np == np.random.rand()
        assert random_torch == torch.rand(1).item()

    def test_seed_everything_different_values(self):
        """Test that different seeds produce different random values."""
        # Seed with one value
        seed_everything(42)
        random_val_1 = torch.rand(10)

        # Seed with a different value
        seed_everything(123)
        random_val_2 = torch.rand(10)

        # Values should be different
        assert not torch.allclose(random_val_1, random_val_2)

    def test_seed_everything_os_environ(self):
        """Test that seed_everything sets PYTHONHASHSEED environment variable."""
        seed_value = 42
        seed_everything(seed_value)

        assert os.environ["PYTHONHASHSEED"] == str(seed_value)

    def test_seed_everything_cudnn_settings(self):
        """Test that seed_everything correctly sets CUDNN parameters."""
        # Test with default values
        seed_everything(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

        # Test with custom values
        seed_everything(42, cudnn_benchmark=True, cudnn_deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_seed_everything_global_seed_state(self):
        """Test that seed_everything affects the global seed state."""
        # Set a specific seed
        seed_value = 42
        seed_everything(seed_value)

        # Generate some random values
        rand1 = torch.rand(5)
        rand2 = np.random.rand(5)
        rand3 = [random.random() for _ in range(5)]  # nosec B311

        # Reset with the same seed
        seed_everything(seed_value)

        # Generate new random values - they should match the previous ones
        rand1_new = torch.rand(5)
        rand2_new = np.random.rand(5)
        rand3_new = [random.random() for _ in range(5)]  # nosec B311

        # Check if the random values are identical
        assert torch.all(torch.eq(rand1, rand1_new))
        assert np.array_equal(rand2, rand2_new)
        assert rand3 == rand3_new
