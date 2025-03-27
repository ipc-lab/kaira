import numpy as np
import pytest
import torch

from kaira.utils import (
    calculate_num_filters_image,
    snr_db_to_linear,
    snr_linear_to_db,
    snr_to_noise_power,
    to_tensor,
)


def test_snr_conversion():
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


def test_snr_to_noise_power():
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
    assert snr_to_noise_power(complex_power, 3.0) == pytest.approx(2.0)  # 4/(10^(3/10))


def test_calculate_num_filters_image():
    """Test calculating number of filters for image processing."""
    # Test with various bandwidth ratios
    assert calculate_num_filters_image(1, 1.0) == 3  # RGB channels, no compression
    assert calculate_num_filters_image(1, 0.5) == 1  # Compression by half
    assert calculate_num_filters_image(1, 2.0) == 6  # Expansion by 2x

    # Test with grayscale (1 channel)
    assert calculate_num_filters_image(1, 1.0, 1) == 1  # 1 channel, no compression

    # Test with different channel counts
    assert calculate_num_filters_image(1, 1.0, 4) == 4  # 4 channels (RGBA)
    assert calculate_num_filters_image(1, 0.25, 3) == 1  # RGB compressed to 1/4

    # Test with complex bandwidth ratios
    assert calculate_num_filters_image(1, 1.33, 3) == 4  # Ceiling of 3*1.33


def test_to_tensor():
    """Test conversion of various types to tensors."""
    # Test with existing tensors (should return unchanged)
    tensor = torch.tensor([1, 2, 3])
    assert torch.equal(to_tensor(tensor), tensor)

    # Test with Python list
    assert torch.equal(to_tensor([1, 2, 3]), torch.tensor([1, 2, 3]))

    # Test with NumPy array
    np_array = np.array([1, 2, 3])
    assert torch.equal(to_tensor(np_array), torch.tensor([1, 2, 3]))

    # Test with scalar values
    assert torch.equal(to_tensor(5), torch.tensor(5))
    assert torch.equal(to_tensor(5.0), torch.tensor(5.0))

    # Test with specific device
    if torch.cuda.is_available():
        cuda_tensor = to_tensor([1, 2, 3], device="cuda")
        assert cuda_tensor.device.type == "cuda"

    # Test with unsupported type
    with pytest.raises(TypeError):
        to_tensor({"key": "value"})


def test_to_tensor_with_mixed_dtype():
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


def test_to_tensor_error_cases():
    """Test to_tensor function error handling."""
    # Test with unsupported types
    with pytest.raises(TypeError):
        to_tensor({"key": "value"})
    
    with pytest.raises(TypeError):
        to_tensor(None)
    
    with pytest.raises(TypeError):
        to_tensor((1, 2, 3))  # Tuple is not directly supported


def test_to_tensor_device_handling():
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
