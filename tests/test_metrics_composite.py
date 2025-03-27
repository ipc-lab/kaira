import pytest
import torch
from kaira.metrics import CompositeMetric, PSNR, SSIM, LPIPS

@pytest.fixture
def sample_preds():
    return torch.randn(1, 3, 256, 256)

@pytest.fixture
def sample_targets():
    return torch.randn(1, 3, 256, 256)

def test_composite_metric_initialization():
    metrics = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}
    weights = {"psnr": 0.3, "ssim": 0.3, "lpips": -0.4}
    composite = CompositeMetric(metrics=metrics, weights=weights)
    assert isinstance(composite, CompositeMetric)

def test_composite_metric_forward(sample_preds, sample_targets):
    metrics = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}
    weights = {"psnr": 0.3, "ssim": 0.3, "lpips": -0.4}
    composite = CompositeMetric(metrics=metrics, weights=weights)
    result = composite(sample_preds, sample_targets)
    assert isinstance(result, torch.Tensor)

def test_composite_metric_compute_individual(sample_preds, sample_targets):
    metrics = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}
    weights = {"psnr": 0.3, "ssim": 0.3, "lpips": -0.4}
    composite = CompositeMetric(metrics=metrics, weights=weights)
    individual_scores = composite.compute_individual(sample_preds, sample_targets)
    assert isinstance(individual_scores, dict)
    assert "psnr" in individual_scores
    assert "ssim" in individual_scores
    assert "lpips" in individual_scores

def test_composite_metric_normalized_weights():
    metrics = {"psnr": PSNR(), "ssim": SSIM(), "lpips": LPIPS()}
    weights = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}
    composite = CompositeMetric(metrics=metrics, weights=weights)
    total_weight = sum(composite.weights.values())
    assert torch.isclose(torch.tensor(total_weight), torch.tensor(1.0))

def test_composite_metric_forward_with_tuple_return(sample_preds, sample_targets):
    class DummyMetricWithTupleReturn(PSNR):
        def forward(self, x, y):
            return super().forward(x, y), torch.tensor(0.0)

    metrics = {"psnr": DummyMetricWithTupleReturn(), "ssim": SSIM(), "lpips": LPIPS()}
    weights = {"psnr": 0.3, "ssim": 0.3, "lpips": -0.4}
    composite = CompositeMetric(metrics=metrics, weights=weights)
    result = composite(sample_preds, sample_targets)
    assert isinstance(result, torch.Tensor)
