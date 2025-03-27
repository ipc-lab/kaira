import pytest
import torch

from kaira.metrics import BaseMetric
from kaira.metrics.composite import CompositeMetric


class MockMetric(BaseMetric):
    """Mock metric that returns a fixed value."""

    def __init__(self, return_value=0.5):
        super().__init__()
        self.return_value = return_value

    def forward(self, preds, targets):
        return torch.tensor(self.return_value)

    def reset(self):
        pass


def test_composite_metric_creation():
    """Test creating CompositeMetric with various configurations."""
    # Simple case with two metrics
    metric1 = MockMetric(0.3)
    metric2 = MockMetric(0.7)

    metrics = {"metric1": metric1, "metric2": metric2}
    composite = CompositeMetric(metrics)

    # Default weights should be equal
    assert composite.weights["metric1"] == 0.5
    assert composite.weights["metric2"] == 0.5

    # With custom weights
    weights = {"metric1": 0.3, "metric2": 0.7}
    composite = CompositeMetric(metrics, weights)

    assert composite.weights["metric1"] == 0.3
    assert composite.weights["metric2"] == 0.7

    # Invalid weights (should raise exception)
    with pytest.raises(ValueError):
        CompositeMetric(metrics, {"metric1": 0.3, "invalid_key": 0.7})


def test_composite_metric_forward():
    """Test forward pass of CompositeMetric."""
    metric1 = MockMetric(0.3)
    metric2 = MockMetric(0.7)

    # Equal weights
    metrics = {"metric1": metric1, "metric2": metric2}
    composite = CompositeMetric(metrics)

    # Forward with dummy inputs
    result = composite(torch.zeros(1), torch.zeros(1))

    # Should be weighted average: 0.3 * 0.5 + 0.7 * 0.5 = 0.5
    assert result.item() == 0.5

    # Custom weights
    weights = {"metric1": 0.8, "metric2": 0.2}
    composite = CompositeMetric(metrics, weights)

    result = composite(torch.zeros(1), torch.zeros(1))

    # Should be weighted average: 0.3 * 0.8 + 0.7 * 0.2 = 0.38
    assert abs(result.item() - 0.38) < 1e-6


def test_composite_metric_individual_metrics():
    """Test getting individual metric values."""
    metric1 = MockMetric(0.3)
    metric2 = MockMetric(0.7)

    metrics = {"metric1": metric1, "metric2": metric2}
    composite = CompositeMetric(metrics)

    # Get individual metrics
    individual = composite.compute_individual(torch.zeros(1), torch.zeros(1))

    assert "metric1" in individual
    assert "metric2" in individual
    assert pytest.approx(individual["metric1"].item(), abs=1e-5) == 0.3
    assert pytest.approx(individual["metric2"].item(), abs=1e-5) == 0.7


def test_add_metric():
    """Test adding a new metric to an existing CompositeMetric."""
    metric1 = MockMetric(0.3)

    # Start with single metric
    metrics = {"metric1": metric1}
    composite = CompositeMetric(metrics)

    # Add another metric
    metric2 = MockMetric(0.7)
    composite.add_metric("metric2", metric2, weight=0.3)

    # Check weights (should be normalized)
    assert composite.weights["metric1"] == 0.7
    assert composite.weights["metric2"] == 0.3

    # Adding duplicate should raise error
    with pytest.raises(ValueError):
        composite.add_metric("metric1", MockMetric(0.5))


def test_composite_metric_get_individual_metrics():
    """Test getting individual metric values from CompositeMetric."""
    metric1 = MockMetric(return_value=torch.tensor(1.0))
    metric2 = MockMetric(return_value=torch.tensor(2.0))
    
    # Create with metrics dictionary instead of empty constructor
    metrics = {"metric1": metric1, "metric2": metric2}
    composite = CompositeMetric(metrics)
    
    # Setup mock inputs
    preds = torch.randn(1, 3, 10, 10)
    target = torch.randn(1, 3, 10, 10)
    
    # Get individual metrics using compute_individual method
    individual_metrics = composite.compute_individual(preds, target)
    
    assert isinstance(individual_metrics, dict)
    assert "metric1" in individual_metrics
    assert "metric2" in individual_metrics
    assert torch.equal(individual_metrics["metric1"], torch.tensor(1.0))
    assert torch.equal(individual_metrics["metric2"], torch.tensor(2.0))


def test_composite_metric_string_representation():
    """Test string representation of CompositeMetric."""
    metric1 = MockMetric(return_value=torch.tensor(1.0))
    metric2 = MockMetric(return_value=torch.tensor(2.0))

    # Initialize with dictionary of metrics
    metrics = {"metric1": metric1, "metric2": metric2}
    composite = CompositeMetric(metrics)

    # Check string representation
    string_repr = str(composite)

    assert "CompositeMetric" in string_repr
    assert "metric1" in string_repr
    assert "metric2" in string_repr


def test_composite_metric_get_metrics():
    """Test get_metrics method in CompositeMetric."""
    metric1 = MockMetric(return_value=torch.tensor(1.0))
    metric2 = MockMetric(return_value=torch.tensor(2.0))

    composite = CompositeMetric()
    composite.add_metric("metric1", metric1)
    composite.add_metric("metric2", metric2)

    # Get the metrics dictionary
    metrics_dict = composite.get_metrics()

    # Check keys and values
    assert isinstance(metrics_dict, dict)
    assert "metric1" in metrics_dict
    assert "metric2" in metrics_dict
    assert metrics_dict["metric1"] is metric1
    assert metrics_dict["metric2"] is metric2
