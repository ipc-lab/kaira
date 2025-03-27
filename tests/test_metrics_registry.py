import pytest
from kaira.metrics.registry import MetricRegistry

def test_metric_registry_create():
    metric = MetricRegistry.create("psnr", data_range=255.0)
    assert metric.name == "PSNR"

def test_metric_registry_list_metrics():
    metrics = MetricRegistry.list_metrics()
    assert "psnr" in metrics
    assert "ssim" in metrics

def test_metric_registry_get_metric_info():
    info = MetricRegistry.get_metric_info("psnr")
    assert info["name"] == "psnr"
    assert info["class"] == "PeakSignalNoiseRatio"
    assert "data_range" in info["parameters"]

def test_metric_registry_create_image_quality_metrics():
    metrics = MetricRegistry.create_image_quality_metrics(data_range=1.0)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "lpips" in metrics

def test_metric_registry_create_composite_metric():
    metrics = MetricRegistry.create_image_quality_metrics(data_range=1.0)
    weights = {"psnr": 0.5, "ssim": 0.5}
    composite = MetricRegistry.create_composite_metric(metrics, weights)
    assert composite.name == "CompositeMetric"
