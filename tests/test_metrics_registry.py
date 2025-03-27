import pytest
import torch

from kaira.metrics import BaseMetric, MetricRegistry
from kaira.metrics.signal import BitErrorRate


class DummyMetric(BaseMetric):
    def __init__(self, value=0.5):
        super().__init__()
        self.value = value

    def forward(self, preds, targets):
        return torch.tensor(self.value)

    def reset(self):
        pass


def test_metric_registry_register():
    """Test registering a metric with the MetricRegistry."""
    # Clear existing registrations for this test
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()
    
    try:
        # Register a new metric
        MetricRegistry.register("dummy", DummyMetric)
        assert "dummy" in MetricRegistry._metrics
        assert MetricRegistry._metrics["dummy"] == DummyMetric
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_register_decorator():
    """Test using register_metric decorator."""
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()
    
    try:
        # Define and register a metric using decorator
        @MetricRegistry.register_metric("decorator_test")
        class TestMetric(BaseMetric):
            def forward(self, preds, targets):
                return torch.tensor(0.0)
            
            def reset(self):
                pass
        
        # Check registration
        assert "decorator_test" in MetricRegistry._metrics
        assert MetricRegistry._metrics["decorator_test"] == TestMetric
        
        # Test with custom name
        @MetricRegistry.register_metric()
        class ImplicitNameMetric(BaseMetric):
            def forward(self, preds, targets):
                return torch.tensor(0.0)
            
            def reset(self):
                pass
        
        # Should use class name (lowercase)
        assert "implicitnamemetric" in MetricRegistry._metrics
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_create():
    """Test creating a metric instance from the registry."""
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()
    
    try:
        # Register a metric and create an instance
        MetricRegistry.register("test_param", DummyMetric)
        metric = MetricRegistry.create("test_param", value=0.75)
        
        # Verify the instance
        assert isinstance(metric, DummyMetric)
        assert metric.value == 0.75
        
        # Test with non-existent metric
        with pytest.raises(ValueError):
            MetricRegistry.create("nonexistent_metric")
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_list_metrics():
    """Test listing registered metrics."""
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()
    
    try:
        MetricRegistry.register("metric1", DummyMetric)
        MetricRegistry.register("metric2", DummyMetric)
        
        metrics = MetricRegistry.list_metrics()
        assert "metric1" in metrics
        assert "metric2" in metrics
        assert len(metrics) == 2
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_get_metric_info():
    """Test getting metric info."""
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()
    
    try:
        MetricRegistry.register("info_test", DummyMetric)
        
        # Get info
        info = MetricRegistry.get_metric_info("info_test")
        assert info["name"] == "info_test"
        assert info["class"] == DummyMetric.__name__
        assert "signature" in info
        
        # Test with non-existent metric
        with pytest.raises(ValueError):
            MetricRegistry.get_metric_info("nonexistent")
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_create_image_quality_metrics():
    """Test creating image quality metrics."""
    metrics = MetricRegistry.create_image_quality_metrics(data_range=2.0)
    
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "ms_ssim" in metrics
    assert "lpips" in metrics


def test_create_composite_metric():
    """Test creating composite metrics with weights."""
    # Create component metrics
    metrics = {
        "metric1": DummyMetric(0.3),
        "metric2": DummyMetric(0.7)
    }
    
    # Test with default equal weights
    composite = MetricRegistry.create_composite_metric(metrics)
    result = composite(torch.zeros(1), torch.zeros(1))
    assert result.item() == pytest.approx(0.5)  # Average of 0.3 and 0.7
    
    # Test with custom weights
    weights = {"metric1": 0.8, "metric2": 0.2}
    composite = MetricRegistry.create_composite_metric(metrics, weights)
    result = composite(torch.zeros(1), torch.zeros(1))
    assert result.item() == pytest.approx(0.3*0.8 + 0.7*0.2)
