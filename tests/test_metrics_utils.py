import pytest
import torch
from kaira.metrics.utils import (
    compute_multiple_metrics,
    format_metric_results,
    visualize_metrics_comparison,
    benchmark_metrics,
    batch_metrics_to_table,
    print_metric_table,
    summarize_metrics_over_batches,
)
from kaira.metrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

@pytest.fixture
def sample_preds():
    return torch.randn(1, 3, 256, 256)

@pytest.fixture
def sample_targets():
    return torch.randn(1, 3, 256, 256)

def test_compute_multiple_metrics(sample_preds, sample_targets):
    metrics = {
        "PSNR": PeakSignalNoiseRatio(),
        "SSIM": StructuralSimilarityIndexMeasure(),
    }
    results = compute_multiple_metrics(metrics, sample_preds, sample_targets)
    assert "PSNR" in results
    assert "SSIM" in results

def test_format_metric_results():
    results = {
        "PSNR": (30.0, 0.5),
        "SSIM": 0.85,
    }
    formatted = format_metric_results(results)
    assert "PSNR: 30.0000 ± 0.5000" in formatted
    assert "SSIM: 0.8500" in formatted

def test_visualize_metrics_comparison(sample_preds, sample_targets):
    metrics = {
        "PSNR": PeakSignalNoiseRatio(),
        "SSIM": StructuralSimilarityIndexMeasure(),
    }
    results1 = compute_multiple_metrics(metrics, sample_preds, sample_targets)
    results2 = compute_multiple_metrics(metrics, sample_preds * 0.9, sample_targets)
    visualize_metrics_comparison([results1, results2], ["Run 1", "Run 2"])

def test_benchmark_metrics(sample_preds, sample_targets):
    metrics = {
        "PSNR": PeakSignalNoiseRatio(),
        "SSIM": StructuralSimilarityIndexMeasure(),
    }
    results = benchmark_metrics(metrics, sample_preds, sample_targets)
    assert "PSNR" in results
    assert "SSIM" in results
    assert "mean_time" in results["PSNR"]
    assert "mean_time" in results["SSIM"]

def test_batch_metrics_to_table():
    metrics_dict = {
        "PSNR": [30.0, 31.0, 29.5],
        "SSIM": [0.85, 0.86, 0.84],
    }
    table = batch_metrics_to_table(metrics_dict)
    assert table[0] == ["Metric", "Mean", "Std"]
    assert table[1][0] == "PSNR"
    assert table[2][0] == "SSIM"

def test_print_metric_table():
    table = [
        ["Metric", "Mean", "Std"],
        ["PSNR", "30.0000", "0.5000"],
        ["SSIM", "0.8500", "0.0100"],
    ]
    print_metric_table(table)

def test_summarize_metrics_over_batches():
    metrics_history = [
        {"PSNR": 30.0, "SSIM": 0.85},
        {"PSNR": 31.0, "SSIM": 0.86},
        {"PSNR": 29.5, "SSIM": 0.84},
    ]
    summary = summarize_metrics_over_batches(metrics_history)
    assert "PSNR" in summary
    assert "SSIM" in summary
    assert "mean" in summary["PSNR"]
    assert "mean" in summary["SSIM"]
