import torch

from kaira.data import create_binary_tensor, create_uniform_tensor


def test_create_binary_tensor_shape():
    """Test binary tensor creation with various shapes."""
    # Test 1D shape
    shape1 = 10
    tensor1 = create_binary_tensor(shape1)
    assert tensor1.shape == (10,)

    # Test 2D shape
    shape2 = (5, 8)
    tensor2 = create_binary_tensor(shape2)
    assert tensor2.shape == (5, 8)

    # Test 3D shape
    shape3 = (2, 3, 4)
    tensor3 = create_binary_tensor(shape3)
    assert tensor3.shape == (2, 3, 4)


def test_create_binary_tensor_probability():
    """Test binary tensor with different probabilities."""
    # High probability (most 1s)
    high_prob = create_binary_tensor((1000,), prob=0.9)
    high_mean = high_prob.float().mean().item()
    assert 0.85 <= high_mean <= 0.95  # Allow for randomness

    # Low probability (most 0s)
    low_prob = create_binary_tensor((1000,), prob=0.1)
    low_mean = low_prob.float().mean().item()
    assert 0.05 <= low_mean <= 0.15  # Allow for randomness

    # Edge case - all 1s
    all_ones = create_binary_tensor((100,), prob=1.0)
    assert all_ones.all().item()

    # Edge case - all 0s
    all_zeros = create_binary_tensor((100,), prob=0.0)
    assert not all_zeros.any().item()


def test_create_binary_tensor_dtype():
    """Test binary tensor with different dtypes."""
    # Float by default
    tensor_float = create_binary_tensor(10, dtype=torch.float)
    assert tensor_float.dtype == torch.float

    # Integer
    tensor_int = create_binary_tensor(10, dtype=torch.int)
    assert tensor_int.dtype == torch.int

    # Bool
    tensor_bool = create_binary_tensor(10, dtype=torch.bool)
    assert tensor_bool.dtype == torch.bool


def test_create_uniform_tensor():
    """Test uniform tensor creation."""
    # Test basic creation
    shape = (5, 10)
    tensor = create_uniform_tensor(shape)
    assert tensor.shape == shape

    # Test with custom range
    low, high = -2.0, 5.0
    tensor_range = create_uniform_tensor(shape, low=low, high=high)
    assert tensor_range.min() >= low
    assert tensor_range.max() <= high

    # Test with specific dtype
    tensor_double = create_uniform_tensor(shape, dtype=torch.double)
    assert tensor_double.dtype == torch.double

    # Test with specific device
    if torch.cuda.is_available():
        tensor_gpu = create_uniform_tensor(shape, device="cuda")
        assert tensor_gpu.device.type == "cuda"
