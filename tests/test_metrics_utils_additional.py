import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # Use non-interactive backend for testing
from kaira.metrics import BaseMetric
from kaira.metrics.utils import benchmark_metrics


@pytest.fixture
def sample_data():
    """Fixture for creating sample data tensors."""
    return torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)


def test_benchmark_metrics_with_empty_metrics(sample_data):
    """Test benchmark_metrics with empty metrics dictionary."""
    preds, target = sample_data
    
    # Should raise AttributeError when passed an empty dictionary
    # because it tries to access preds.is_cuda in an empty loop
    with pytest.raises(ValueError):
        benchmark_metrics({}, preds, target)


class MockMetric(BaseMetric):
    """Simple mock metric that returns a constant value."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.tensor(0.5)


def test_benchmark_metrics_with_cuda_check(sample_data, monkeypatch):
    """Test benchmark_metrics with CUDA tensor check."""
    preds, target = sample_data
    
    # Mock the CUDA check
    is_cuda_called = False
    
    def mock_is_cuda():
        nonlocal is_cuda_called
        is_cuda_called = True  # Fixed: assign True to the variable
        return False
    
    # Apply the mock to the tensor
    preds.is_cuda = mock_is_cuda
    
    # Run benchmark
    metrics = {"test_metric": MockMetric()}
    results = benchmark_metrics(metrics, preds, target, repeat=1)
    
    # Verify results
    assert "test_metric" in results
    assert is_cuda_called
