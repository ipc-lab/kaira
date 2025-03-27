"""Comprehensive tests for metrics utility functions."""
import pytest
import torch
import numpy as np
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


class SimpleMetric(BaseMetric):
    """A simple metric for testing."""
    
    def __init__(self, scalar_output=True):
        super().__init__()
        self.scalar_output = scalar_output
    
    def forward(self, preds, targets):
        """Calculate mean absolute error."""
        return torch.mean(torch.abs(preds - targets))
    
    def compute_with_stats(self, preds, targets):
        """Calculate mean and std of absolute error."""
        abs_errors = torch.abs(preds - targets)
        return torch.mean(abs_errors), torch.std(abs_errors)


class TestComputeMultipleMetrics:
    """Tests for compute_multiple_metrics function."""
    
    def test_with_single_metric(self):
        """Test computing a single metric."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.2])
        
        metrics = {"mae": SimpleMetric()}
        results = compute_multiple_metrics(metrics, preds, targets)
        
        assert "mae" in results
        assert isinstance(results["mae"], torch.Tensor)
        assert results["mae"].ndim == 0  # Scalar tensor
        
    def test_with_multiple_metrics(self):
        """Test computing multiple metrics."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.2])
        
        metrics = {
            "mae1": SimpleMetric(),
            "mae2": SimpleMetric(),
            "mae3": SimpleMetric()
        }
        results = compute_multiple_metrics(metrics, preds, targets)
        
        assert len(results) == 3
        assert "mae1" in results
        assert "mae2" in results
        assert "mae3" in results
        
        # All metrics should give the same result in this case
        assert torch.allclose(results["mae1"], results["mae2"])
        assert torch.allclose(results["mae2"], results["mae3"])
    
    def test_with_compute_with_stats(self):
        """Test computing metrics that implement compute_with_stats."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.2])
        
        metrics = {"mae_with_stats": SimpleMetric()}
        results = compute_multiple_metrics(metrics, preds, targets)
        
        assert "mae_with_stats" in results
        assert isinstance(results["mae_with_stats"], tuple)
        assert len(results["mae_with_stats"]) == 2
        mean, std = results["mae_with_stats"]
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
    
    def test_with_mixed_metrics(self):
        """Test computing a mix of metrics with and without stats."""
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.2])
        
        # Create a metric without compute_with_stats method
        class SimpleMetricNoStats(BaseMetric):
            def forward(self, preds, targets):
                return torch.mean(torch.abs(preds - targets))
        
        metrics = {
            "with_stats": SimpleMetric(),
            "no_stats": SimpleMetricNoStats()
        }
        
        results = compute_multiple_metrics(metrics, preds, targets)
        
        assert "with_stats" in results
        assert "no_stats" in results
        assert isinstance(results["with_stats"], tuple)
        assert isinstance(results["no_stats"], torch.Tensor)


class TestFormatMetricResults:
    """Tests for format_metric_results function."""
    
    def test_with_scalar_values(self):
        """Test formatting scalar metric results."""
        results = {
            "metric1": torch.tensor(0.123),
            "metric2": torch.tensor(0.456),
            "metric3": 0.789  # Python float
        }
        
        formatted = format_metric_results(results)
        
        assert isinstance(formatted, str)
        assert "metric1: 0.1230" in formatted
        assert "metric2: 0.4560" in formatted
        assert "metric3: 0.7890" in formatted
    
    def test_with_mean_std_tuples(self):
        """Test formatting (mean, std) tuple results."""
        results = {
            "metric1": (torch.tensor(0.123), torch.tensor(0.011)),
            "metric2": (0.456, 0.022)  # Python floats
        }
        
        formatted = format_metric_results(results)
        
        assert isinstance(formatted, str)
        assert "metric1: 0.1230 ± 0.0110" in formatted
        assert "metric2: 0.4560 ± 0.0220" in formatted
    
    def test_with_mixed_result_types(self):
        """Test formatting a mix of scalar and (mean, std) tuple results."""
        results = {
            "metric1": torch.tensor(0.123),
            "metric2": (torch.tensor(0.456), torch.tensor(0.022))
        }
        
        formatted = format_metric_results(results)
        
        assert isinstance(formatted, str)
        assert "metric1: 0.1230" in formatted
        assert "metric2: 0.4560 ± 0.0220" in formatted


class TestVisualizeMetricsComparison:
    """Tests for visualize_metrics_comparison function."""
    
    def test_basic_visualization(self, monkeypatch):
        """Test basic visualization functionality with mocked plt functions."""
        # Mock matplotlib functions to avoid actual plotting
        mock_calls = []
        
        class MockFigure:
            def __init__(self, figsize):
                mock_calls.append(("figure", figsize))
        
        def mock_bar(x, height, width, yerr, label, capsize):
            mock_calls.append(("bar", x, height, width, yerr, label, capsize))
            return []
        
        def mock_xlabel(label):
            mock_calls.append(("xlabel", label))
        
        def mock_ylabel(label):
            mock_calls.append(("ylabel", label))
        
        def mock_title(title):
            mock_calls.append(("title", title))
        
        def mock_xticks(ticks, labels, rotation):
            mock_calls.append(("xticks", ticks, labels, rotation))
        
        def mock_legend():
            mock_calls.append(("legend",))
        
        def mock_tight_layout():
            mock_calls.append(("tight_layout",))
        
        def mock_savefig(path):
            mock_calls.append(("savefig", path))
        
        def mock_show():
            mock_calls.append(("show",))
        
        # Apply mocks
        monkeypatch.setattr(plt, "figure", MockFigure)
        monkeypatch.setattr(plt, "bar", mock_bar)
        monkeypatch.setattr(plt, "xlabel", mock_xlabel)
        monkeypatch.setattr(plt, "ylabel", mock_ylabel)
        monkeypatch.setattr(plt, "title", mock_title)
        monkeypatch.setattr(plt, "xticks", mock_xticks)
        monkeypatch.setattr(plt, "legend", mock_legend)
        monkeypatch.setattr(plt, "tight_layout", mock_tight_layout)
        monkeypatch.setattr(plt, "savefig", mock_savefig)
        monkeypatch.setattr(plt, "show", mock_show)
        
        # Test data
        results_list = [
            {"metric1": 0.1, "metric2": 0.2, "metric3": 0.3},
            {"metric1": 0.4, "metric2": 0.5, "metric3": 0.6}
        ]
        labels = ["Method A", "Method B"]
        
        # Call function
        visualize_metrics_comparison(results_list, labels)
        
        # Verify calls
        assert ("figure", (12, 6)) in mock_calls
        assert ("title", "Metrics Comparison") in mock_calls
        assert ("xlabel", "Metrics") in mock_calls
        assert ("ylabel", "Value") in mock_calls
        assert ("legend",) in mock_calls
        assert ("tight_layout",) in mock_calls
        assert ("show",) in mock_calls
    
    def test_with_mean_std_tuples(self, monkeypatch):
        """Test visualization with (mean, std) tuple results."""
        # Mock plt.bar to capture arguments
        bar_args = []
        
        def mock_bar(x, height, width, yerr, label, capsize):
            bar_args.append((x, height, yerr, label))
            return []
        
        # Mock other plt functions to do nothing
        def mock_noop(*args, **kwargs):
            pass
        
        # Apply mocks
        monkeypatch.setattr(plt, "figure", mock_noop)
        monkeypatch.setattr(plt, "bar", mock_bar)
        monkeypatch.setattr(plt, "xlabel", mock_noop)
        monkeypatch.setattr(plt, "ylabel", mock_noop)
        monkeypatch.setattr(plt, "title", mock_noop)
        monkeypatch.setattr(plt, "xticks", mock_noop)
        monkeypatch.setattr(plt, "legend", mock_noop)
        monkeypatch.setattr(plt, "tight_layout", mock_noop)
        monkeypatch.setattr(plt, "show", mock_noop)
        
        # Test data with (mean, std) tuples
        results_list = [
            {"metric1": (0.1, 0.01), "metric2": (0.2, 0.02)},
            {"metric1": (0.3, 0.03), "metric2": (0.4, 0.04)}
        ]
        labels = ["Method A", "Method B"]
        
        # Call function
        visualize_metrics_comparison(results_list, labels)
        
        # Verify bar arguments
        assert len(bar_args) == 2  # Two methods
        
        # Check that error bars (yerr) are passed correctly for Method A
        _, heights1, yerrs1, label1 = bar_args[0]
        assert label1 == "Method A"
        assert heights1 == [0.1, 0.2]  # Mean values
        assert yerrs1 == [0.01, 0.02]  # Std values
        
        # Check Method B
        _, heights2, yerrs2, label2 = bar_args[1]
        assert label2 == "Method B"
        assert heights2 == [0.3, 0.4]
        assert yerrs2 == [0.03, 0.04]
    
    def test_with_save_path(self, monkeypatch):
        """Test saving visualization to a file."""
        # Mock plt functions
        savefig_called = [False]
        
        def mock_noop(*args, **kwargs):
            pass
        
        def mock_savefig(path):
            savefig_called[0] = True
            assert path == "test_metrics.png"
        
        # Apply mocks
        monkeypatch.setattr(plt, "figure", mock_noop)
        monkeypatch.setattr(plt, "bar", lambda *args, **kwargs: [])
        monkeypatch.setattr(plt, "xlabel", mock_noop)
        monkeypatch.setattr(plt, "ylabel", mock_noop)
        monkeypatch.setattr(plt, "title", mock_noop)
        monkeypatch.setattr(plt, "xticks", mock_noop)
        monkeypatch.setattr(plt, "legend", mock_noop)
        monkeypatch.setattr(plt, "tight_layout", mock_noop)
        monkeypatch.setattr(plt, "savefig", mock_savefig)
        monkeypatch.setattr(plt, "show", mock_noop)
        
        # Test data
        results_list = [{"metric1": 0.1}]
        labels = ["Method A"]
        
        # Call function with save_path
        visualize_metrics_comparison(results_list, labels, save_path="test_metrics.png")
        
        # Verify savefig was called
        assert savefig_called[0]
    
    def test_empty_results_list(self):
        """Test that error is raised with empty results list."""
        with pytest.raises(ValueError):
            visualize_metrics_comparison([], ["Method A"])
    
    def test_tensor_conversion(self, monkeypatch):
        """Test proper handling of torch.Tensor values."""
        # Mock plt.bar to capture arguments
        bar_args = []
        
        def mock_bar(x, height, width, yerr, label, capsize):
            bar_args.append((height, yerr))
            return []
        
        # Mock other plt functions to do nothing
        def mock_noop(*args, **kwargs):
            pass
        
        # Apply mocks
        monkeypatch.setattr(plt, "figure", mock_noop)
        monkeypatch.setattr(plt, "bar", mock_bar)
        monkeypatch.setattr(plt, "xlabel", mock_noop)
        monkeypatch.setattr(plt, "ylabel", mock_noop)
        monkeypatch.setattr(plt, "title", mock_noop)
        monkeypatch.setattr(plt, "xticks", mock_noop)
        monkeypatch.setattr(plt, "legend", mock_noop)
        monkeypatch.setattr(plt, "tight_layout", mock_noop)
        monkeypatch.setattr(plt, "show", mock_noop)
        
        # Test data with torch.Tensor values
        results_list = [
            {"metric1": torch.tensor(0.1), "metric2": (torch.tensor(0.2), torch.tensor(0.02))},
        ]
        labels = ["Method A"]
        
        # Call function
        visualize_metrics_comparison(results_list, labels)
        
        # Check that tensors were converted to Python floats
        heights, yerrs = bar_args[0]
        assert isinstance(heights[0], float)
        assert isinstance(heights[1], float)
        assert isinstance(yerrs[1], float)


class TestBenchmarkMetrics:
    """Tests for benchmark_metrics function."""
    
    def test_basic_benchmarking(self, monkeypatch):
        """Test basic benchmarking functionality with mocked time function."""
        # Mock time.time to return deterministic values
        time_counter = [0.0]
        
        def mock_time():
            time_counter[0] += 0.1  # Increment by 0.1 seconds each call
            return time_counter[0]
        
        # Mock cuda synchronize to do nothing
        def mock_synchronize():
            pass
        
        # Apply mocks
        monkeypatch.setattr("time.time", mock_time)
        monkeypatch.setattr(torch.cuda, "synchronize", mock_synchronize)
        
        # Test data
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = torch.tensor([1.1, 2.2, 2.9, 4.2])
        
        metrics = {
            "mae": SimpleMetric(),
            "another_metric": SimpleMetric()
        }
        
        # Call function with fewer repeats for testing
        results = benchmark_metrics(metrics, preds, targets, repeat=3)
        
        # Check results structure
        assert "mae" in results
        assert "another_metric" in results
        
        for metric_name in results:
            assert "mean_time" in results[metric_name]
            assert "std_time" in results[metric_name]
            assert "min_time" in results[metric_name]
            assert "max_time" in results[metric_name]
            
            # With our mock, each measurement should be 0.1 seconds
            assert results[metric_name]["mean_time"] == 0.1
            assert results[metric_name]["min_time"] == 0.1
            assert results[metric_name]["max_time"] == 0.1


class TestBatchMetricsToTable:
    """Tests for batch_metrics_to_table function."""
    
    def test_basic_table_conversion(self):
        """Test basic conversion of metrics to table format."""
        metrics_dict = {
            "metric1": [0.1, 0.2, 0.3],
            "metric2": [0.4, 0.5, 0.6]
        }
        
        table = batch_metrics_to_table(metrics_dict)
        
        # Check table structure
        assert len(table) == 3  # Header + 2 rows
        assert table[0] == ["Metric", "Mean", "Std"]  # Header row
        
        # Check metric1 row
        assert table[1][0] == "metric1"
        assert table[1][1] == "0.2000"  # Mean
        assert table[1][2] == "0.0816"  # Std (approximately)
        
        # Check metric2 row
        assert table[2][0] == "metric2"
        assert table[2][1] == "0.5000"  # Mean
        assert table[2][2] == "0.0816"  # Std (approximately)
    
    def test_custom_precision(self):
        """Test table conversion with custom precision."""
        metrics_dict = {
            "metric1": [0.12345, 0.23456]
        }
        
        table = batch_metrics_to_table(metrics_dict, precision=2)
        
        # Check formatting with precision=2
        assert table[1][1] == "0.18"  # Mean with 2 decimal places
        
        # Test with higher precision
        table = batch_metrics_to_table(metrics_dict, precision=6)
        assert table[1][1] == "0.179005"  # Mean with 6 decimal places
    
    def test_without_std(self):
        """Test table conversion without standard deviation."""
        metrics_dict = {
            "metric1": [0.1, 0.2, 0.3]
        }
        
        table = batch_metrics_to_table(metrics_dict, include_std=False)
        
        # Check table structure
        assert len(table) == 2  # Header + 1 row
        assert table[0] == ["Metric", "Mean"]  # Header without Std
        assert len(table[1]) == 2  # Only metric name and mean
        assert table[1][0] == "metric1"
        assert table[1][1] == "0.2000"


class TestPrintMetricTable:
    """Tests for print_metric_table function."""
    
    def test_basic_printing(self, capsys):
        """Test basic table printing functionality."""
        table = [
            ["Metric", "Mean", "Std"],
            ["metric1", "0.1000", "0.0100"],
            ["metric2", "0.2000", "0.0200"]
        ]
        
        print_metric_table(table)
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Check header and separator
        assert "Metric | Mean   | Std" in captured.out
        assert "-------------------" in captured.out
        
        # Check data rows
        assert "metric1 | 0.1000 | 0.0100" in captured.out
        assert "metric2 | 0.2000 | 0.0200" in captured.out
    
    def test_with_custom_column_widths(self, capsys):
        """Test table printing with custom column widths."""
        table = [
            ["Metric", "Mean", "Std"],
            ["metric1", "0.1000", "0.0100"]
        ]
        
        # Custom widths (wider than needed)
        column_widths = [10, 8, 6]
        
        print_metric_table(table, column_widths)
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Check formatting with custom widths
        assert "Metric     | Mean     | Std   " in captured.out
        assert "metric1    | 0.1000   | 0.0100" in captured.out


class TestSummarizeMetricsOverBatches:
    """Tests for summarize_metrics_over_batches function."""
    
    def test_basic_summarization(self):
        """Test basic metrics summarization over batches."""
        # Test data - metrics from 3 batches
        metrics_history = [
            {"metric1": 0.1, "metric2": 0.4},
            {"metric1": 0.2, "metric2": 0.5},
            {"metric1": 0.3, "metric2": 0.6}
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        
        # Check summary structure
        assert "metric1" in summary
        assert "metric2" in summary
        
        # Check statistics for metric1
        assert abs(summary["metric1"]["mean"] - 0.2) < 1e-6
        assert abs(summary["metric1"]["std"] - 0.0816) < 1e-3
        assert summary["metric1"]["min"] == 0.1
        assert summary["metric1"]["max"] == 0.3
        assert abs(summary["metric1"]["median"] - 0.2) < 1e-6
        assert summary["metric1"]["n_samples"] == 3
        
        # Check statistics for metric2
        assert abs(summary["metric2"]["mean"] - 0.5) < 1e-6
        assert abs(summary["metric2"]["std"] - 0.0816) < 1e-3
        assert summary["metric2"]["min"] == 0.4
        assert summary["metric2"]["max"] == 0.6
        assert abs(summary["metric2"]["median"] - 0.5) < 1e-6
        assert summary["metric2"]["n_samples"] == 3
    
    def test_with_tensor_values(self):
        """Test summarization with torch.Tensor values."""
        # Metrics with tensor values
        metrics_history = [
            {"metric1": torch.tensor(0.1), "metric2": torch.tensor(0.4)},
            {"metric1": torch.tensor(0.2), "metric2": torch.tensor(0.5)},
            {"metric1": torch.tensor(0.3), "metric2": torch.tensor(0.6)}
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        
        # Check that tensor values were properly converted
        assert isinstance(summary["metric1"]["mean"], float)
        assert isinstance(summary["metric1"]["std"], float)
        
        # Check actual values (same as previous test)
        assert abs(summary["metric1"]["mean"] - 0.2) < 1e-6
        assert abs(summary["metric1"]["std"] - 0.0816) < 1e-3
    
    def test_with_tuple_values(self):
        """Test summarization with (mean, std) tuple values."""
        # Metrics with (mean, std) tuples
        metrics_history = [
            {"metric1": (0.1, 0.01), "metric2": 0.4},
            {"metric1": (0.2, 0.02), "metric2": 0.5},
            {"metric1": (0.3, 0.03), "metric2": 0.6}
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        
        # Should use the mean values from the tuples
        assert abs(summary["metric1"]["mean"] - 0.2) < 1e-6
        assert abs(summary["metric1"]["std"] - 0.0816) < 1e-3
        
        # Regular metric should be the same as before
        assert abs(summary["metric2"]["mean"] - 0.5) < 1e-6
    
    def test_with_tensor_tuple_values(self):
        """Test summarization with tuple values containing tensors."""
        # Metrics with (tensor, tensor) tuples
        metrics_history = [
            {"metric1": (torch.tensor(0.1), torch.tensor(0.01))},
            {"metric1": (torch.tensor(0.2), torch.tensor(0.02))},
            {"metric1": (torch.tensor(0.3), torch.tensor(0.03))}
        ]
        
        summary = summarize_metrics_over_batches(metrics_history)
        
        # Should convert tensor values to Python floats
        assert isinstance(summary["metric1"]["mean"], float)
        assert abs(summary["metric1"]["mean"] - 0.2) < 1e-6