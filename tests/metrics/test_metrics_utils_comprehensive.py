"""Enhanced tests for the metrics/utils.py module to increase coverage."""
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from kaira.metrics.utils import (
    compute_multiple_metrics,
    format_metric_results,
    visualize_metrics_comparison,
    benchmark_metrics,
    batch_metrics_to_table,
    print_metric_table,
    summarize_metrics_over_batches
)
from kaira.metrics.base import BaseMetric

class MockMetric(BaseMetric):
    """Mock metric class for testing."""
    
    def __init__(self, return_value=0.5, return_with_stats=False):
        super().__init__()
        self.return_value = return_value
        self.return_with_stats = return_with_stats
    
    def forward(self, preds, targets):
        """Return a mock metric value."""
        return torch.tensor(self.return_value)
    
    def compute_with_stats(self, preds, targets):
        """Return a mock metric value with statistics."""
        return torch.tensor(self.return_value), torch.tensor(0.1)

class TestMetricsUtils:
    """Test suite for metrics utility functions."""
    
    def test_compute_multiple_metrics(self):
        """Test computing multiple metrics at once."""
        # Create mock metrics
        metrics = {
            "metric1": MockMetric(return_value=0.7),
            "metric2": MockMetric(return_value=0.8),
            "metric3": MockMetric(return_value=0.9, return_with_stats=True)
        }
        
        # Create dummy tensors
        preds = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        # Compute metrics
        results = compute_multiple_metrics(metrics, preds, targets)
        
        # Verify results
        assert "metric1" in results
        assert "metric2" in results
        assert "metric3" in results
        assert torch.isclose(results["metric1"], torch.tensor(0.7))
        assert torch.isclose(results["metric2"], torch.tensor(0.8))
        assert isinstance(results["metric3"], tuple)
        assert torch.isclose(results["metric3"][0], torch.tensor(0.9))
        assert torch.isclose(results["metric3"][1], torch.tensor(0.1))
    
    def test_format_metric_results(self):
        """Test formatting metric results as a string."""
        # Test with scalar values
        results = {
            "metric1": torch.tensor(0.7532),
            "metric2": 0.8456
        }
        formatted = format_metric_results(results)
        assert "metric1: 0.7532" in formatted
        assert "metric2: 0.8456" in formatted
        
        # Test with (mean, std) tuples
        results = {
            "metric1": (torch.tensor(0.7532), torch.tensor(0.0123)),
            "metric2": (0.8456, 0.0234)
        }
        formatted = format_metric_results(results)
        assert "metric1: 0.7532 ± 0.0123" in formatted
        assert "metric2: 0.8456 ± 0.0234" in formatted
        
        # Test with mixed values
        results = {
            "metric1": torch.tensor(0.7532),
            "metric2": (0.8456, 0.0234)
        }
        formatted = format_metric_results(results)
        assert "metric1: 0.7532" in formatted
        assert "metric2: 0.8456 ± 0.0234" in formatted
    
    def test_batch_metrics_to_table(self):
        """Test converting batch metrics to a table format."""
        metrics_dict = {
            "metric1": [0.75, 0.76, 0.74, 0.77],
            "metric2": [0.85, 0.84, 0.86, 0.83]
        }
        
        # Test with default parameters
        table = batch_metrics_to_table(metrics_dict)
        assert len(table) == 3  # Header + 2 metrics
        assert table[0] == ["Metric", "Mean", "Std"]
        assert table[1][0] == "metric1"
        assert table[2][0] == "metric2"
        
        # Test with custom precision
        table = batch_metrics_to_table(metrics_dict, precision=2)
        assert table[1][1] == "0.76"  # Rounded to 2 decimal places
        
        # Test without std
        table = batch_metrics_to_table(metrics_dict, include_std=False)
        assert table[0] == ["Metric", "Mean"]
        assert len(table[1]) == 2
    
    def test_print_metric_table(self, capsys):
        """Test printing a formatted table of metrics."""
        table = [
            ["Metric", "Mean", "Std"],
            ["metric1", "0.7532", "0.0123"],
            ["metric2", "0.8456", "0.0234"]
        ]
        
        # Test with default column widths
        print_metric_table(table)
        captured = capsys.readouterr()
        assert "Metric  | Mean   | Std" in captured.out
        assert "metric1 | 0.7532 | 0.0123" in captured.out
        assert "metric2 | 0.8456 | 0.0234" in captured.out
        
        # Test with custom column widths
        print_metric_table(table, column_widths=[10, 8, 8])
        captured = capsys.readouterr()
        assert "Metric     | Mean     | Std" in captured.out
    
    def test_summarize_metrics_over_batches(self):
        """Test summarizing metrics collected over multiple batches."""
        metrics_history = [
            {"metric1": 0.75, "metric2": (0.85, 0.02)},
            {"metric1": 0.76, "metric2": (0.84, 0.03)},
            {"metric1": 0.74, "metric2": (0.86, 0.01)},
            {"metric1": 0.77, "metric2": (0.83, 0.02)}
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        
        assert "metric1" in summary
        assert "metric2" in summary
        assert "mean" in summary["metric1"]
        assert "std" in summary["metric1"]
        assert "min" in summary["metric1"]
        assert "max" in summary["metric1"]
        assert "median" in summary["metric1"]
        assert "n_samples" in summary["metric1"]
        
        assert summary["metric1"]["n_samples"] == 4
        assert 0.755 - 0.001 <= summary["metric1"]["mean"] <= 0.755 + 0.001  # Approx 0.755
        
        # Test with tensor values
        metrics_history = [
            {"metric1": torch.tensor(0.75), "metric2": (torch.tensor(0.85), torch.tensor(0.02))},
            {"metric1": torch.tensor(0.76), "metric2": (torch.tensor(0.84), torch.tensor(0.03))},
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        assert "metric1" in summary
        assert "metric2" in summary
        assert 0.755 - 0.001 <= summary["metric1"]["mean"] <= 0.755 + 0.001  # Approx 0.755
    
    def test_visualize_metrics_comparison(self):
        """Test visualization of metrics comparison across multiple experiments."""
        results_list = [
            {"metric1": 0.75, "metric2": 0.85, "metric3": (0.90, 0.05)},
            {"metric1": 0.78, "metric2": 0.82, "metric3": (0.92, 0.04)}
        ]
        labels = ["Model A", "Model B"]
        
        # Test basic functionality (should not raise errors)
        # Temporarily replace plt.show with a no-op to avoid displaying
        original_show = plt.show
        plt.show = lambda: None
        
        try:
            visualize_metrics_comparison(results_list, labels)
            # Test with tensor values
            results_list = [
                {"metric1": torch.tensor(0.75), "metric2": torch.tensor(0.85)},
                {"metric1": torch.tensor(0.78), "metric2": torch.tensor(0.82)}
            ]
            visualize_metrics_comparison(results_list, labels)
            
            # Test with save path (but don't actually save)
            with pytest.raises(OSError):  # Assuming the directory doesn't exist
                visualize_metrics_comparison(results_list, labels, save_path="/nonexistent/dir/plot.png")
                
            # Test with empty results list
            with pytest.raises(ValueError):
                visualize_metrics_comparison([], labels)
        finally:
            # Restore original plt.show
            plt.show = original_show
            plt.close("all")  # Close any open plots
    
    def test_benchmark_metrics(self):
        """Test benchmarking execution time of metrics."""
        # Create mock metrics
        metrics = {
            "fast_metric": MockMetric(return_value=0.7),
            "slow_metric": MockMetric(return_value=0.8)
        }
        
        # Create dummy tensors
        preds = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        # Benchmark with fewer repetitions to speed up test
        results = benchmark_metrics(metrics, preds, targets, repeat=2)
        
        # Verify results structure
        assert "fast_metric" in results
        assert "slow_metric" in results
        
        for metric_name in results:
            assert "mean_time" in results[metric_name]
            assert "std_time" in results[metric_name]
            assert "min_time" in results[metric_name]
            assert "max_time" in results[metric_name]
            
            assert results[metric_name]["mean_time"] >= 0
            assert results[metric_name]["min_time"] <= results[metric_name]["max_time"]