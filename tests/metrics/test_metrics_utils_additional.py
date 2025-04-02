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
    
    # In the actual implementation, an empty metrics dictionary will just
    # return an empty dictionary of results, not raise an error
    results = benchmark_metrics({}, preds, target)
    
    # Verify we get an empty dictionary back
    assert isinstance(results, dict)
    assert len(results) == 0


class MockMetric(BaseMetric):
    """Simple mock metric that returns a constant value."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.tensor(0.5)


def test_benchmark_metrics_without_mocking(sample_data, monkeypatch):
    """Test benchmark_metrics without trying to mock tensor attributes."""
    preds, target = sample_data
    
    # Mock time.time instead
    import time
    time_calls = []
    original_time = time.time
    
    def mock_time():
        # Record that time was called and return incrementing values
        time_calls.append(True)
        return len(time_calls) * 0.1
    
    monkeypatch.setattr(time, "time", mock_time)
    
    # Run benchmark
    metrics = {"test_metric": MockMetric()}
    results = benchmark_metrics(metrics, preds, target, repeat=1)
    
    # Restore original time function
    monkeypatch.setattr(time, "time", original_time)
    
    # Verify results structure
    assert "test_metric" in results
    assert "mean_time" in results["test_metric"]
    assert "std_time" in results["test_metric"]
    assert "min_time" in results["test_metric"]
    assert "max_time" in results["test_metric"]
    
    # Verify that time.time was called multiple times
    assert len(time_calls) > 2
