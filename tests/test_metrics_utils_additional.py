import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # Use non-interactive backend for testing

from kaira.metrics.utils import benchmark_metrics


@pytest.fixture
def sample_data():
    """Fixture for creating sample data tensors."""
    return torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)


def test_benchmark_metrics_with_empty_metrics():
    """Test benchmark_metrics with empty metrics list."""
    preds = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    # Should raise ValueError with empty metrics list
    with pytest.raises(ValueError):
        benchmark_metrics([], preds, target)
