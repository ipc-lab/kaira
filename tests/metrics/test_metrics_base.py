import pytest
import torch

from kaira.metrics.base import BaseMetric


class ConcreteMetric(BaseMetric):
    """Concrete implementation of BaseMetric for testing."""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.total = 0.0
        self.count = 0

    def update(self, preds, targets):
        """Update metric state."""
        error = torch.abs(preds - targets).mean()
        self.total += error.item()
        self.count += 1

    def compute(self):
        """Compute final metric value."""
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.total / self.count)

    def forward(self, preds, targets):
        """Forward pass for metric computation."""
        error = torch.abs(preds - targets)
        if self.reduction == "none":
            return error
        elif self.reduction == "sum":
            return error.sum()
        else:  # default: 'mean'
            return error.mean()


class DummyMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.value = 0

    def update(self, preds, target):
        self.value += 1

    def compute(self):
        return self.value

    def forward(self, preds, target):
        self.update(preds, target)
        return self.compute()


def test_base_metric_stateful_computation():
    """Test stateful computation of metrics (update-compute pattern)."""
    metric = ConcreteMetric()

    # Create test data
    batch1 = (torch.tensor([1.0, 2.0]), torch.tensor([1.2, 1.8]))
    batch2 = (torch.tensor([3.0, 4.0]), torch.tensor([2.8, 4.2]))

    # First batch
    metric.update(*batch1)
    assert metric.count == 1

    # Second batch
    metric.update(*batch2)
    assert metric.count == 2

    # Compute final result
    result = metric.compute()
    assert isinstance(result, torch.Tensor)

    # Reset state
    metric.reset()
    assert metric.count == 0
    assert metric.total == 0.0


def test_base_metric_direct_computation():
    """Test direct computation of metrics (forward pass)."""
    metric = ConcreteMetric()

    # Create test data
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.2, 1.8, 3.1])

    # Compute directly
    result = metric(preds, targets)
    assert isinstance(result, torch.Tensor)

    # Test different reductions
    metric_none = ConcreteMetric(reduction="none")
    result_none = metric_none(preds, targets)
    assert result_none.shape == preds.shape

    metric_sum = ConcreteMetric(reduction="sum")
    result_sum = metric_sum(preds, targets)
    assert result_sum.shape == torch.Size([])


def test_base_metric_initialization():
    """Test BaseMetric initialization."""
    metric = DummyMetric()
    assert isinstance(metric, BaseMetric)


def test_base_metric_forward_calls_update_compute():
    """Test that BaseMetric forward method calls update and compute."""
    metric = DummyMetric()

    # Setup mock inputs
    preds = torch.randn(1, 3, 10, 10)
    target = torch.randn(1, 3, 10, 10)

    # Call forward
    result = metric(preds, target)

    # Check that update was called (since value should be 1)
    assert result == 1


def test_compute_with_stats():
    """Test computing metric with statistics."""
    metric = ConcreteMetric(reduction="none")

    # Create test data with predictable variance
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.5, 2.5, 3.5, 4.5])

    # Call compute_with_stats
    mean, std = metric.compute_with_stats(preds, targets)

    # Check return types
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

    # Check values are correct
    # Forward returns absolute difference which is 0.5 for all values
    assert mean.item() == 0.5
    assert std.item() == 0.0

    # Test with data that has non-zero standard deviation
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 2.4, 2.7, 4.2])

    mean, std = metric.compute_with_stats(preds, targets)
    # Absolute differences should be [0.1, 0.4, 0.3, 0.2]
    assert pytest.approx(mean.item()) == 0.25  # Using pytest.approx to handle floating-point precision
    assert 0.11 < std.item() < 0.13  # Approximately 0.12247...


def test_string_representation():
    """Test string representation of BaseMetric."""
    # Test with default name (class name)
    metric = DummyMetric()
    assert str(metric) == "DummyMetric Metric"

    # Test with custom name
    metric_custom = ConcreteMetric()
    metric_custom.name = "CustomName"
    assert str(metric_custom) == "CustomName Metric"
