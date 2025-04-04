import pytest
import torch
import torch.nn as nn

from kaira.metrics import BaseMetric  # Add this import for the mock classes
from kaira.metrics.registry import MetricRegistry


class DummyMetric(nn.Module):
    def __init__(self, value=0.0, other_param=None):
        super().__init__()
        self.value = value
        self.other_param = other_param

    def forward(self, preds, target):
        return torch.tensor(self.value)


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
        class TestMetric(nn.Module):
            def forward(self, preds, targets):
                return torch.tensor(0.0)

            def reset(self):
                pass

        # Check registration
        assert "decorator_test" in MetricRegistry._metrics
        assert MetricRegistry._metrics["decorator_test"] == TestMetric

        # Test with custom name
        @MetricRegistry.register_metric()
        class ImplicitNameMetric(nn.Module):
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
        with pytest.raises(KeyError):  # The actual implementation raises KeyError not ValueError
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
        # The actual implementation doesn't have a 'signature' key but has 'parameters' instead
        assert "parameters" in info

        # Test with non-existent metric - implementation raises KeyError
        with pytest.raises(KeyError):
            MetricRegistry.get_metric_info("nonexistent")
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_create_image_quality_metrics():
    """Test creating actual image quality metrics without mocking."""
    # Create metrics with default parameters
    metrics = MetricRegistry.create_image_quality_metrics()
    
    # Verify all expected metrics are created
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "ms_ssim" in metrics
    assert "lpips" in metrics
    
    # Check that metrics are the correct type
    from kaira.metrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleSSIM, LearnedPerceptualImagePatchSimilarity
    
    assert isinstance(metrics["psnr"], PeakSignalNoiseRatio)
    assert isinstance(metrics["ssim"], StructuralSimilarityIndexMeasure)
    assert isinstance(metrics["ms_ssim"], MultiScaleSSIM)
    assert isinstance(metrics["lpips"], LearnedPerceptualImagePatchSimilarity)
    
    # Test with custom data_range
    custom_metrics = MetricRegistry.create_image_quality_metrics(data_range=255.0)
    assert custom_metrics["psnr"].psnr.data_range == 255.0
    assert custom_metrics["ssim"].ssim.data_range == 255.0
    assert custom_metrics["ms_ssim"].data_range == 255.0
    
    # Test with custom lpips_net_type
    lpips_metrics = MetricRegistry.create_image_quality_metrics(lpips_net_type="vgg")
    assert lpips_metrics["lpips"].net_type == "vgg"
    
    # Test device placement
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_metrics = MetricRegistry.create_image_quality_metrics(device=device)
        # Check at least one parameter is on the target device
        psnr_params = list(gpu_metrics["psnr"].parameters())
        if psnr_params:  # Ensure there are parameters to check
            assert next(iter(psnr_params)).device == device
    
    # Test with sample data - use 192x192 images to satisfy MS-SSIM requirements (needs > 160)
    sample_preds = torch.rand(2, 3, 192, 192)
    sample_targets = torch.rand(2, 3, 192, 192)
    
    # Ensure all metrics can process the sample data without errors
    for metric_name, metric in metrics.items():
        result = metric(sample_preds, sample_targets)
        assert isinstance(result, torch.Tensor)


def test_create_composite_metric():
    """Test creating composite metrics with weights."""
    # Create component metrics
    metrics = {"metric1": DummyMetric(0.3), "metric2": DummyMetric(0.7)}

    # Test with default equal weights
    composite = MetricRegistry.create_composite_metric(metrics)
    result = composite(torch.zeros(1), torch.zeros(1))
    assert result.item() == pytest.approx(0.5)  # Average of 0.3 and 0.7

    # Test with custom weights
    weights = {"metric1": 0.8, "metric2": 0.2}
    composite = MetricRegistry.create_composite_metric(metrics, weights)
    result = composite(torch.zeros(1), torch.zeros(1))
    assert result.item() == pytest.approx(0.3 * 0.8 + 0.7 * 0.2)


def test_metric_registry_clear():
    """Test clearing the MetricRegistry."""
    original_metrics = MetricRegistry._metrics.copy()

    try:
        # Register a test metric
        MetricRegistry.register("test_clear", DummyMetric)

        # Clear registry
        MetricRegistry.clear()

        # Check that registry is empty
        assert len(MetricRegistry._metrics) == 0

        # Test re-registering after clear
        MetricRegistry.register("test_after_clear", DummyMetric)
        assert "test_after_clear" in MetricRegistry._metrics
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_create_with_args_kwargs():
    """Test creating a metric with args and kwargs."""
    original_metrics = MetricRegistry._metrics.copy()

    try:
        # Register a test metric
        MetricRegistry.register("test_args_kwargs", DummyMetric)

        # Create with positional arg
        metric1 = MetricRegistry.create("test_args_kwargs", 0.5)
        assert metric1.value == 0.5

        # Create with keyword arg
        metric2 = MetricRegistry.create("test_args_kwargs", value=0.7)
        assert metric2.value == 0.7

        # Create with both positional and keyword args (but the first positional goes to other_param)
        metric3 = MetricRegistry.create("test_args_kwargs", 0.5, other_param=0.9)
        assert metric3.value == 0.5
        assert metric3.other_param == 0.9
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_available_metrics():
    """Test getting available metrics."""
    original_metrics = MetricRegistry._metrics.copy()

    try:
        # Clear and add known metrics
        MetricRegistry._metrics.clear()
        MetricRegistry.register("metric1", DummyMetric)
        MetricRegistry.register("metric2", DummyMetric)

        # Get available metrics
        available = MetricRegistry.available_metrics()

        # Check result
        assert isinstance(available, list)
        assert "metric1" in available
        assert "metric2" in available
        assert len(available) == 2
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_register_duplicate():
    """Test that registering a metric with a duplicate name raises ValueError."""
    original_metrics = MetricRegistry._metrics.copy()
    MetricRegistry._metrics.clear()

    try:
        # Register a metric
        MetricRegistry.register("duplicate_test", DummyMetric)
        
        # Try to register another metric with the same name
        with pytest.raises(ValueError, match="Metric with name 'duplicate_test' already registered"):
            MetricRegistry.register("duplicate_test", BaseMetric)
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics


def test_metric_registry_available_metrics_alias():
    """Test that available_metrics is an alias for list_metrics."""
    original_metrics = MetricRegistry._metrics.copy()
    
    try:
        # Register some test metrics
        MetricRegistry._metrics.clear()
        MetricRegistry.register("alias_test1", DummyMetric)
        MetricRegistry.register("alias_test2", DummyMetric)
        
        # Get metrics using both methods
        list_result = MetricRegistry.list_metrics()
        available_result = MetricRegistry.available_metrics()
        
        # Verify they return the same result
        assert sorted(list_result) == sorted(available_result)
        assert "alias_test1" in available_result
        assert "alias_test2" in available_result
    finally:
        # Restore original metrics
        MetricRegistry._metrics = original_metrics
