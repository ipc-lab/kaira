import torch

from kaira.data.generation import (
    BinaryTensorDataset,
    UniformTensorDataset,
    create_binary_tensor,
    create_uniform_tensor,
)


def test_binary_tensor_dataset_length():
    dataset = BinaryTensorDataset(size=(100, 10), prob=0.5)
    assert len(dataset) == 100


def test_binary_tensor_dataset_item_shape():
    dataset = BinaryTensorDataset(size=(100, 10), prob=0.5)
    item = dataset[0]
    assert item.shape == torch.Size([10])


def test_binary_tensor_dataset_item_values():
    dataset = BinaryTensorDataset(size=(100, 10), prob=0.5)
    item = dataset[0]
    assert torch.all((item == 0) | (item == 1))


def test_binary_tensor_dataset_slice_shape():
    dataset = BinaryTensorDataset(size=(100, 10), prob=0.5)
    batch = dataset[10:20]
    assert batch.shape == torch.Size([10, 10])


def test_binary_tensor_dataset_prob():
    dataset = BinaryTensorDataset(size=(1000, 10), prob=0.7)
    data = dataset.data
    mean = data.float().mean().item()
    assert abs(mean - 0.7) < 0.05


def test_binary_tensor_dataset_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BinaryTensorDataset(size=(100, 10), prob=0.5, device=device)
    assert dataset.data.device.type == device.type
    assert (dataset.data.device.index or 0) == (device.index or 0)


def test_create_binary_tensor_shape():
    """Test binary tensor creation with various shapes."""
    # Test with list shape
    shape = [10, 20]
    tensor = create_binary_tensor(shape)
    assert tensor.shape == torch.Size(shape)

    # Test with torch.Size
    tensor = create_binary_tensor(torch.Size([5, 15]))
    assert tensor.shape == torch.Size([5, 15])

    # Test with single integer
    tensor = create_binary_tensor(7)
    assert tensor.shape == torch.Size([7])


def test_create_binary_tensor_probability():
    """Test binary tensor with different probabilities."""
    # Test with high probability (mostly 1s)
    tensor = create_binary_tensor((1000,), prob=0.9)
    assert 0.85 <= tensor.mean().item() <= 0.95  # Statistical approximation

    # Test with low probability (mostly 0s)
    tensor = create_binary_tensor((1000,), prob=0.1)
    assert 0.05 <= tensor.mean().item() <= 0.15  # Statistical approximation


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
        
    # Test with single integer size parameter
    tensor_single_int = create_uniform_tensor(7)
    assert tensor_single_int.shape == torch.Size([7])
    assert 0.0 <= tensor_single_int.min() < tensor_single_int.max() <= 1.0


def test_binary_tensor_dataset():
    """Test BinaryTensorDataset functionality."""
    # Test dataset initialization
    size = [100, 20]
    dataset = BinaryTensorDataset(size, prob=0.3)

    # Test dataset length
    assert len(dataset) == 100

    # Test getitem for a single item
    item = dataset[0]
    assert item.shape == torch.Size([20])

    # Test getitem for a slice
    items = dataset[10:20]
    assert items.shape == torch.Size([10, 20])

    # Test getitem with a list of indices
    indices = [5, 10, 15]
    items = dataset[indices]
    assert items.shape == torch.Size([3, 20])


def test_uniform_tensor_dataset():
    """Test UniformTensorDataset functionality."""
    # Test dataset initialization
    size = [50, 10]
    dataset = UniformTensorDataset(size, low=-1.0, high=1.0)

    # Test dataset length
    assert len(dataset) == 50

    # Test getitem
    item = dataset[0]
    assert item.shape == torch.Size([10])
    assert (item >= -1.0).all() and (item <= 1.0).all()

    # Test slicing
    items = dataset[5:15]
    assert items.shape == torch.Size([10, 10])
