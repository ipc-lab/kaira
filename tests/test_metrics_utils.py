import pytest
import torch
import numpy as np
from typing import Dict, Any, List
import tempfile
import os

from kaira.metrics import BaseMetric
from kaira.metrics.utils import (
    compute_multiple_metrics,
    format_metric_results,
    visualize_metrics_comparison,
    benchmark_metrics,
    batch_metrics_to_table,
    summarize_metrics_over_batches
)


class MockMetric(BaseMetric):
    """Mock metric that returns a predefined value."""
    def __init__(self, return_value=0.5, has_std=False):
        super().__init__()
        self.return_value = return_value
        self.has_std = has_std
        
    def forward(self, preds, targets):
        if self.has_std:
            return (torch.tensor(self.return_value), torch.tensor(0.1))
        return torch.tensor(self.return_value)


def test_compute_multiple_metrics():
    """Test computing multiple metrics at once."""
    # Create test metrics
    metrics = {
        "metric1": MockMetric(0.8),
        "metric2": MockMetric(0.6, has_std=True)
    }
    
    # Create dummy tensors
    preds = torch.randn(8, 3, 32, 32)
    targets = torch.randn(8, 3, 32, 32)
    
    # Compute metrics
    results = compute_multiple_metrics(metrics, preds, targets)
    
    # Check results
    assert "metric1" in results
    assert "metric2" in results
    assert isinstance(results["metric1"], torch.Tensor)
    assert isinstance(results["metric2"], tuple)
    assert len(results["metric2"]) == 2
    assert results["metric1"].item() == 0.8
    assert results["metric2"][0].item() == 0.6
    assert results["metric2"][1].item() == 0.1


def test_format_metric_results():
    """Test formatting metric results as a string."""
    # Create test results
    results = {
        "metric1": torch.tensor(0.8),
        "metric2": (torch.tensor(0.6), torch.tensor(0.1)),
        "metric3": 0.7  # Plain number
    }
    
    # Format results
    formatted = format_metric_results(results)
    
    # Check output
    assert "metric1: 0.8000" in formatted
    assert "metric2: 0.6000 (±0.1000)" in formatted
    assert "metric3: 0.7000" in formatted


def test_visualize_metrics_comparison(monkeypatch):
    """Test visualization of metric comparisons."""
    # Mock matplotlib to avoid actual plotting
    import matplotlib.pyplot as plt
    
    # Track whether functions are called
    show_called = False
    save_called = False
    
    # Mock functions
    def mock_show():
        nonlocal show_called
        show_called = True
    
    def mock_savefig(path):
        nonlocal save_called
        save_called = True
        assert path == "test_path.png"
    
    monkeypatch.setattr(plt, "show", mock_show)
    monkeypatch.setattr(plt, "savefig", mock_savefig)
    
    # Create test data
    results_list = [
        {"metric1": 0.8, "metric2": 0.6},
        {"metric1": 0.7, "metric2": 0.5},
    ]
    labels = ["Model A", "Model B"]
    
    # Test without saving
    visualize_metrics_comparison(results_list, labels)
    assert show_called
    
    # Reset flags
    show_called = False
    
    # Test with saving
    visualize_metrics_comparison(results_list, labels, save_path="test_path.png")
    assert show_called
    assert save_called
    
    # Test with empty results
    with pytest.raises(ValueError):
        visualize_metrics_comparison([], ["Empty"])


def test_benchmark_metrics(monkeypatch):
    """Test benchmarking of metrics."""
    # Mock time module
    import time
    
    time_sequence = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    call_count = 0
    
    def mock_time():
        nonlocal call_count
        result = time_sequence[call_count % len(time_sequence)]
        call_count += 1
        return result
    
    monkeypatch.setattr(time, "time", mock_time)
    
    # Create test data
    metrics = {
        "fast_metric": MockMetric(0.8),
        "slow_metric": MockMetric(0.6)
    }
    
    preds = torch.randn(8, 3, 32, 32)
    targets = torch.randn(8, 3, 32, 32)
    
    # Run benchmark with fewer repeats to speed up test
    results = benchmark_metrics(metrics, preds, targets, repeat=2)
    
    # Check results structure
    assert "fast_metric" in results
    assert "slow_metric" in results
    for metric_name in results:
        assert "mean_time" in results[metric_name]
        assert "std_time" in results[metric_name]
        assert "min_time" in results[metric_name]
        assert "max_time" in results[metric_name]


def test_batch_metrics_to_table():
    """Test conversion of batch metrics to table format."""
    # Create test data
    metrics_dict = {
        "metric1": [0.8, 0.7, 0.9],
        "metric2": [0.6, 0.5, 0.7]
    }
    
    # Generate table
    table = batch_metrics_to_table(metrics_dict)
    
    # Check format
    assert len(table) > 0
    assert isinstance(table, list)
    assert isinstance(table[0], list)
    
    # Check headers
    assert "Metric" in table[0]
    assert "Mean" in table[0]
    assert "Std" in table[0]
    
    # Check with different precision
    table_precise = batch_metrics_to_table(metrics_dict, precision=6)
    assert len(table_precise[1][1]) > len(table[1][1])  # More digits
    
    # Check without std
    table_no_std = batch_metrics_to_table(metrics_dict, include_std=False)
    assert "Std" not in table_no_std[0]


def test_summarize_metrics_over_batches():
    """Test summarizing metrics over multiple batches."""
    # Create test data
    metrics_history = [
        {"metric1": 0.8, "metric2": (0.6, 0.1)},
        {"metric1": 0.7, "metric2": (0.5, 0.1)},
        {"metric1": 0.9, "metric2": (0.7, 0.1)}
    ]
    
    # Summarize metrics
    summary = summarize_metrics_over_batches(metrics_history)
    
    # Check structure
    assert "metric1" in summary
    assert "metric2" in summary
    
    # Check statistics
    for metric in summary.values():
        assert "mean" in metric
        assert "std" in metric
        assert "min" in metric
        assert "max" in metric
        assert "median" in metric
        assert "n_samples" in metric
        
    # Check values
    assert summary["metric1"]["mean"] == pytest.approx(0.8)
    assert summary["metric1"]["n_samples"] == 3
    assert summary["metric2"]["mean"] == pytest.approx(0.6)
