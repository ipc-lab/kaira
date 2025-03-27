# tests/test_metrics_base.py

import pytest
import torch
from kaira.metrics.base import BaseMetric

class DummyMetric(BaseMetric):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(x - y)

def test_base_metric_initialization():
    metric = DummyMetric()
    assert isinstance(metric, BaseMetric)
    assert metric.name == "DummyMetric"

def test_base_metric_forward():
    metric = DummyMetric()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.0])
    result = metric(x, y)
    assert torch.isclose(result, torch.tensor(0.3333), atol=1e-4)

def test_base_metric_compute_with_stats():
    metric = DummyMetric()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.0])
    mean, std = metric.compute_with_stats(x, y)
    assert torch.isclose(mean, torch.tensor(0.3333), atol=1e-4)
    assert torch.isclose(std, torch.tensor(0.4714), atol=1e-4)

def test_base_metric_str():
    metric = DummyMetric()
    assert str(metric) == "DummyMetric Metric"
