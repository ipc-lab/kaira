import pytest
import torch
from kaira.data.generation import BinaryTensorDataset

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
    assert dataset.data.device == device
