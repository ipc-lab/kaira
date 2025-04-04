"""Tests for metrics utility functions."""
import pytest
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from kaira.metrics import BaseMetric
from kaira.metrics.utils import (
    batch_metrics_to_table,
    benchmark_metrics,
    compute_multiple_metrics,
    format_metric_results,
    print_metric_table,
    summarize_metrics_over_batches,
    visualize_metrics_comparison,
)


class MockMetric(BaseMetric):
    """Mock metric that returns a predefined value."""

    def __init__(self, return_value=0.5, has_std=False, return_with_stats=False):
        super().__init__()
        self.return_value = return_value
        self.has_std = has_std
        self.return_with_stats = return_with_stats

    def forward(self, preds, targets):
        if self.has_std:
            return (torch.tensor(self.return_value), torch.tensor(0.1))
        return torch.tensor(self.return_value)
        
    def compute_with_stats(self, preds, targets):
        """Return a mock metric value with statistics."""
        return torch.tensor(self.return_value), torch.tensor(0.1)


class TestMetricsUtils:
    """Test suite for metrics utility functions."""
    
    def test_compute_multiple_metrics(self):
        """Test computing multiple metrics at once."""
        # Create test metrics
        metrics = {
            "metric1": MockMetric(0.8),
            "metric2": MockMetric(0.6),
            "metric3": MockMetric(0.9, return_with_stats=True)
        }
        
        # Create dummy tensors
        preds = torch.randn(8, 3, 32, 32)
        targets = torch.randn(8, 3, 32, 32)
        
        # Compute metrics
        results = compute_multiple_metrics(metrics, preds, targets)
        
        # Check results
        assert "metric1" in results
        assert "metric2" in results
        assert "metric3" in results
        
        # When a metric doesn't have compute_with_stats, the forward method is called
        # But the implementation of compute_multiple_metrics will always call compute_with_stats
        # which returns a tuple (mean, std) for any metric
        assert isinstance(results["metric1"], tuple)
        assert isinstance(results["metric2"], tuple)
        assert isinstance(results["metric3"], tuple)
        
        # Check tuple values - both should be (value, std)
        assert len(results["metric1"]) == 2
        assert len(results["metric2"]) == 2
        
        # Check the actual values with appropriate tolerance
        assert pytest.approx(results["metric1"][0].item(), abs=1e-5) == 0.8
        assert pytest.approx(results["metric2"][0].item(), abs=1e-5) == 0.6
        assert pytest.approx(results["metric3"][0].item(), abs=1e-5) == 0.9
        assert pytest.approx(results["metric3"][1].item(), abs=1e-5) == 0.1

    def test_format_metric_results(self):
        """Test formatting metric results as a string."""
        # Test with scalar values
        results = {
            "metric1": torch.tensor(0.7532),
            "metric2": 0.8456,
            "metric3": (torch.tensor(0.6), torch.tensor(0.1))  # With std
        }
        formatted = format_metric_results(results)
        
        # Check output with the correct formatting that matches the implementation
        assert "metric1: 0.7532" in formatted
        assert "metric2: 0.8456" in formatted
        assert "metric3: 0.6000 ± 0.1000" in formatted  # The implementation uses ± not (±)

        # Test with more (mean, std) tuples
        results = {
            "metric1": (torch.tensor(0.7532), torch.tensor(0.0123)),
            "metric2": (0.8456, 0.0234),
            "metric3": 0.7  # Plain number
        }
        formatted = format_metric_results(results)
        assert "metric1: 0.7532 ± 0.0123" in formatted
        assert "metric2: 0.8456 ± 0.0234" in formatted
        assert "metric3: 0.7000" in formatted

    def test_visualize_metrics_comparison(self, monkeypatch):
        """Test visualization of metric comparisons."""
        # Mock matplotlib to avoid actual plotting
        
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
            
        # Test with tensor values
        results_list = [
            {"metric1": torch.tensor(0.75), "metric2": torch.tensor(0.85)},
            {"metric1": torch.tensor(0.78), "metric2": torch.tensor(0.82)}
        ]
        visualize_metrics_comparison(results_list, labels)

    def test_visualize_metrics_comparison_with_tensor_tuples(self, monkeypatch):
        """Test visualization of metrics comparison with tensor tuples."""
        # Track whether functions are called
        show_called = False
        
        def mock_show():
            nonlocal show_called
            show_called = True
            
        def mock_bar(*args, **kwargs):
            # Capture the arguments to verify tensor conversion
            if 'yerr' in kwargs:
                # Ensure errors are converted to floats
                assert isinstance(kwargs['yerr'][0], float)
            return None
            
        monkeypatch.setattr(plt, "show", mock_show)
        monkeypatch.setattr(plt, "bar", mock_bar)
        
        # Create test data with tensor tuples of (mean, std)
        results_list = [
            {
                "metric1": (torch.tensor(0.8), torch.tensor(0.05)),
                "metric2": (torch.tensor(0.6), torch.tensor(0.03))
            },
            {
                "metric1": (torch.tensor(0.7), torch.tensor(0.04)),
                "metric2": (torch.tensor(0.5), torch.tensor(0.02))
            }
        ]
        
        labels = ["Model A", "Model B"]
        
        # Call the function
        visualize_metrics_comparison(results_list, labels)
        
        # Verify show was called
        assert show_called

    def test_benchmark_metrics(self, monkeypatch):
        """Test benchmarking of metrics."""
        # Mock time module
        time_sequence = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        call_count = 0

        def mock_time():
            nonlocal call_count
            result = time_sequence[call_count % len(time_sequence)]
            call_count += 1
            return result

        monkeypatch.setattr(time, "time", mock_time)

        # Create test data
        metrics = {"fast_metric": MockMetric(0.8), "slow_metric": MockMetric(0.6)}

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

    def test_benchmark_metrics_with_empty_metrics(self, sample_preds, sample_targets):
        """Test benchmark_metrics with empty metrics dictionary."""
        preds, targets = sample_preds, sample_targets
        
        # In the actual implementation, an empty metrics dictionary will just
        # return an empty dictionary of results, not raise an error
        results = benchmark_metrics({}, preds, targets)
        
        # Verify we get an empty dictionary back
        assert isinstance(results, dict)
        assert len(results) == 0

    def test_benchmark_metrics_without_mocking(self, sample_preds, sample_targets, monkeypatch):
        """Test benchmark_metrics without trying to mock tensor attributes."""
        preds, targets = sample_preds, sample_targets
        
        # Mock time.time instead
        time_calls = []
        original_time = time.time
        
        def mock_time():
            # Record that time was called and return incrementing values
            time_calls.append(True)
            return len(time_calls) * 0.1
        
        monkeypatch.setattr(time, "time", mock_time)
        
        # Run benchmark
        metrics = {"test_metric": MockMetric()}
        results = benchmark_metrics(metrics, preds, targets, repeat=1)
        
        # Restore original time function
        monkeypatch.setattr(time, "time", original_time)
        
        # Verify results structure
        assert "test_metric" in results
        assert "mean_time" in results["test_metric"]
        assert "std_time" in results["test_metric"]
        assert "min_time" in results["test_metric"]
        assert "max_time" in results["test_metric"]
        
        # Verify that time.time was called multiple times
        assert len(time_calls) > 2

    def test_batch_metrics_to_table(self):
        """Test conversion of batch metrics to table format."""
        # Create test data
        metrics_dict = {"metric1": [0.8, 0.7, 0.9], "metric2": [0.6, 0.5, 0.7]}

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
        """Test summarizing metrics over multiple batches."""
        # Create test data with mix of direct values and (mean, std) tuples
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
        
        # Test with tensor values
        metrics_history = [
            {"metric1": torch.tensor(0.75), "metric2": (torch.tensor(0.85), torch.tensor(0.02))},
            {"metric1": torch.tensor(0.76), "metric2": (torch.tensor(0.84), torch.tensor(0.03))},
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        assert "metric1" in summary
        assert "metric2" in summary
        assert 0.755 - 0.001 <= summary["metric1"]["mean"] <= 0.755 + 0.001  # Approx 0.755

    def test_compute_multiple_metrics_direct_call(self):
        """Test computing metrics without compute_with_stats method."""
        # Create a custom mock metric without compute_with_stats
        class DirectMockMetric(BaseMetric):
            def __init__(self, return_value=0.5):
                super().__init__()
                self.return_value = return_value
                
            def forward(self, preds, targets):
                return torch.tensor(self.return_value)
                
            # No compute_with_stats method
        
        metrics = {
            "direct_metric": DirectMockMetric(0.7)
        }
        
        # Create dummy tensors
        preds = torch.randn(8, 3, 32, 32)
        targets = torch.randn(8, 3, 32, 32)
        
        # Compute metrics
        results = compute_multiple_metrics(metrics, preds, targets)
        
        # Check results
        assert "direct_metric" in results
        # Just a regular tensor, not a tuple
        assert isinstance(results["direct_metric"], torch.Tensor)
        assert pytest.approx(results["direct_metric"].item(), abs=1e-5) == 0.7

    def test_visualize_metrics_comparison_single_value(self, monkeypatch):
        """Test visualization of metric comparisons with single values (not tuples)."""
        # Track function calls with arguments
        bar_calls = []
        
        def mock_bar(*args, **kwargs):
            nonlocal bar_calls
            bar_calls.append((args, kwargs))
            return None
            
        # Mock other matplotlib functions to avoid actual plotting
        def mock_show(): pass
        def mock_figure(*args, **kwargs): pass
        def mock_xlabel(*args, **kwargs): pass
        def mock_ylabel(*args, **kwargs): pass
        def mock_title(*args, **kwargs): pass
        def mock_xticks(*args, **kwargs): pass
        def mock_legend(*args, **kwargs): pass
        def mock_tight_layout(*args, **kwargs): pass
        
        # Apply all mocks
        monkeypatch.setattr(plt, "bar", mock_bar)
        monkeypatch.setattr(plt, "show", mock_show)
        monkeypatch.setattr(plt, "figure", mock_figure)
        monkeypatch.setattr(plt, "xlabel", mock_xlabel)
        monkeypatch.setattr(plt, "ylabel", mock_ylabel)
        monkeypatch.setattr(plt, "title", mock_title)
        monkeypatch.setattr(plt, "xticks", mock_xticks)
        monkeypatch.setattr(plt, "legend", mock_legend)
        monkeypatch.setattr(plt, "tight_layout", mock_tight_layout)
        
        # Create test data with a tensor value (not a tuple)
        results_list = [
            {
                "single_value_metric": torch.tensor(0.85)
            }
        ]
        labels = ["Single Value Model"]
        
        # Call the function
        visualize_metrics_comparison(results_list, labels)
        
        # Verify bar was called for the single value
        assert len(bar_calls) == 1
        args, kwargs = bar_calls[0]
        
        # Check that value was properly converted
        assert len(args) >= 2
        assert args[1][0] == pytest.approx(0.85, abs=1e-5)
        assert 'yerr' not in kwargs  # No error bars for single values

    def test_print_metric_table_with_empty_rows(self, capsys):
        """Test printing a table with empty rows."""
        table = [
            ["Metric", "Mean", "Std"],
            ["metric1", "0.7532", "0.0123"],
            [],  # Empty row
            ["metric2", "0.8456", "0.0234"],
            []   # Another empty row
        ]
        
        # Print table
        print_metric_table(table)
        captured = capsys.readouterr()
        
        # Check output
        assert "Metric  | Mean   | Std" in captured.out
        assert "metric1 | 0.7532 | 0.0123" in captured.out
        assert "metric2 | 0.8456 | 0.0234" in captured.out
        # Empty rows should be skipped, so we shouldn't see empty lines
        assert "| |" not in captured.out

    def test_print_metric_table_with_missing_columns(self, capsys):
        """Test printing a table with rows that have missing columns."""
        table = [
            ["Metric", "Mean", "Std"],
            ["metric1", "0.7532", "0.0123"],
            ["metric2", "0.8456"],  # Missing Std column
            ["metric3"]  # Missing Mean and Std columns
        ]
        
        # Print table
        print_metric_table(table)
        captured = capsys.readouterr()
        
        # Check output
        assert "Metric  | Mean   | Std" in captured.out
        assert "metric1 | 0.7532 | 0.0123" in captured.out
        assert "metric2 | 0.8456 |" in captured.out  # Empty cell for missing Std
        assert "metric3 |        |" in captured.out  # Empty cells for missing columns

    def test_print_metric_table_empty_cases(self, capsys):
        """Test print_metric_table function with various empty table scenarios."""
        from kaira.metrics.utils import print_metric_table
        
        # Case 1: Completely empty table
        print_metric_table([])
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Case 2: Table with empty first row
        print_metric_table([[]])
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Case 3: Table with all empty rows
        print_metric_table([[], [], []])
        captured = capsys.readouterr()
        assert captured.out == ""
        
        # Case 4: Table with non-empty header but empty data rows
        print_metric_table([["Metric", "Value"], [], []])
        captured = capsys.readouterr()
        assert "Metric" in captured.out
        assert "Value" in captured.out
        assert "----------" in captured.out
        
        # Case 5: Table with mixed empty and non-empty rows
        print_metric_table([
            ["Metric", "Value"],
            [],
            ["metric1", "0.123"],
            [],
            ["metric2", "0.456"]
        ])
        captured = capsys.readouterr()
        assert "Metric" in captured.out
        assert "Value" in captured.out
        assert "metric1" in captured.out
        assert "0.123" in captured.out
        assert "metric2" in captured.out
        assert "0.456" in captured.out

    def test_print_metric_table_non_empty_rows_check(self, capsys):
        """Test the specific condition 'if not non_empty_rows: return' in print_metric_table."""
        # Create a special row class that has non-zero length but evaluates to False
        class FalsyRow(list):
            def __init__(self, *args):
                super().__init__(*args)
            
            def __bool__(self):
                return False
        
        # This row will pass the len(table[0]) > 0 check but will be filtered out by `if row`
        special_row = FalsyRow(["Column1", "Column2"])
        assert len(special_row) > 0  # Has length
        assert not special_row       # But evaluates to False in boolean context
        
        # Create a table where all rows will be filtered out by the `if row` condition
        table = [special_row, special_row, special_row]
        
        # Call print_metric_table
        print_metric_table(table)
        captured = capsys.readouterr()
        
        # Function should return early without printing anything
        assert captured.out == ""
