"""Tests for the Kaira benchmark runners."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from kaira.benchmarks.base import BaseBenchmark, BenchmarkResult, BenchmarkSuite
from kaira.benchmarks.runners import (
    ComparisonRunner,
    ParallelRunner,
    ParametricRunner,
    StandardRunner,
)


class MockBenchmark(BaseBenchmark):
    """Mock benchmark for testing."""

    def __init__(self, name: str, description: str = "", sleep_time: float = 0.01, should_fail: bool = False, metrics: Dict[str, Any] = None):
        super().__init__(name, description)
        self.sleep_time = sleep_time
        self.should_fail = should_fail
        self.test_metrics = metrics or {"accuracy": 0.95, "loss": 0.1}
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False

    def setup(self, **kwargs) -> None:
        super().setup(**kwargs)
        self.setup_called = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.run_called = True
        time.sleep(self.sleep_time)

        if self.should_fail:
            raise ValueError("Mock benchmark failure")

        # Add any kwargs to metrics for parameter testing
        metrics = self.test_metrics.copy()
        metrics.update(kwargs)
        return metrics

    def teardown(self) -> None:
        super().teardown()
        self.teardown_called = True


class TestStandardRunner:
    """Test StandardRunner functionality."""

    def test_init(self):
        """Test StandardRunner initialization."""
        runner = StandardRunner()
        assert runner.verbose is True
        assert runner.save_results is True
        assert runner.results == []

        runner = StandardRunner(verbose=False, save_results=False)
        assert runner.verbose is False
        assert runner.save_results is False

    def test_run_benchmark_success(self):
        """Test running a single successful benchmark."""
        runner = StandardRunner(verbose=False)
        benchmark = MockBenchmark("test_benchmark", "A test benchmark")

        result = runner.run_benchmark(benchmark)

        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_benchmark"
        assert result.description == "A test benchmark"
        assert result.metrics["accuracy"] == 0.95
        assert result.metrics["loss"] == 0.1
        assert result.execution_time > 0
        assert len(runner.results) == 1
        assert benchmark.setup_called
        assert benchmark.run_called
        assert benchmark.teardown_called

    def test_run_benchmark_with_kwargs(self):
        """Test running benchmark with additional kwargs."""
        runner = StandardRunner(verbose=False)
        benchmark = MockBenchmark("test_benchmark")

        result = runner.run_benchmark(benchmark, learning_rate=0.01, epochs=10)

        assert result.metrics["learning_rate"] == 0.01
        assert result.metrics["epochs"] == 10
        assert result.metadata["learning_rate"] == 0.01
        assert result.metadata["epochs"] == 10

    def test_run_benchmark_failure(self):
        """Test running a failing benchmark."""
        runner = StandardRunner(verbose=False)
        benchmark = MockBenchmark("failing_benchmark", should_fail=True)

        result = runner.run_benchmark(benchmark)

        assert result.metrics["success"] is False
        assert "error" in result.metrics
        assert "Mock benchmark failure" in result.metrics["error"]
        assert result.execution_time > 0

    def test_run_benchmark_verbose(self, capsys):
        """Test verbose output during benchmark execution."""
        runner = StandardRunner(verbose=True)
        benchmark = MockBenchmark("test_benchmark")

        runner.run_benchmark(benchmark)

        captured = capsys.readouterr()
        assert "Running benchmark: test_benchmark" in captured.out
        assert "✓ Completed in" in captured.out

    def test_run_benchmark_verbose_failure(self, capsys):
        """Test verbose output for failing benchmark."""
        runner = StandardRunner(verbose=True)
        benchmark = MockBenchmark("failing_benchmark", should_fail=True)

        runner.run_benchmark(benchmark)

        captured = capsys.readouterr()
        assert "Running benchmark: failing_benchmark" in captured.out
        assert "✗ Completed in" in captured.out

    def test_run_suite(self):
        """Test running a benchmark suite."""
        runner = StandardRunner(verbose=False)

        suite = BenchmarkSuite("test_suite", "A test suite")
        benchmark1 = MockBenchmark("benchmark1")
        benchmark2 = MockBenchmark("benchmark2")
        suite.add_benchmark(benchmark1)
        suite.add_benchmark(benchmark2)

        results = runner.run_suite(suite)

        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert results[0].name == "benchmark1"
        assert results[1].name == "benchmark2"
        assert len(runner.results) == 2
        assert suite.results == results  # save_results=True by default

    def test_run_suite_no_save(self):
        """Test running suite without saving results."""
        runner = StandardRunner(verbose=False, save_results=False)

        suite = BenchmarkSuite("test_suite")
        benchmark = MockBenchmark("benchmark1")
        suite.add_benchmark(benchmark)

        results = runner.run_suite(suite)

        assert len(results) == 1
        assert len(suite.results) == 0  # Results not saved to suite

    def test_run_suite_verbose(self, capsys):
        """Test verbose output for suite execution."""
        runner = StandardRunner(verbose=True)

        suite = BenchmarkSuite("test_suite", "A test suite")
        benchmark1 = MockBenchmark("benchmark1")
        benchmark2 = MockBenchmark("benchmark2")
        suite.add_benchmark(benchmark1)
        suite.add_benchmark(benchmark2)

        runner.run_suite(suite)

        captured = capsys.readouterr()
        assert "Running benchmark suite: test_suite" in captured.out
        assert "2 benchmarks to run" in captured.out
        assert "[1/2] benchmark1" in captured.out
        assert "[2/2] benchmark2" in captured.out

    def test_save_all_results(self):
        """Test saving all results to directory."""
        runner = StandardRunner(verbose=False)

        # Run some benchmarks
        benchmark1 = MockBenchmark("benchmark1")
        benchmark2 = MockBenchmark("benchmark2")
        runner.run_benchmark(benchmark1)
        runner.run_benchmark(benchmark2)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.save_all_results(tmpdir)

            # Check that files were created
            result_files = list(Path(tmpdir).glob("*.json"))
            assert len(result_files) == 2

            # Check file content
            for file_path in result_files:
                with open(file_path) as f:
                    data = json.load(f)
                assert "benchmark_id" in data
                assert "name" in data
                assert data["name"] in ["benchmark1", "benchmark2"]


class TestParallelRunner:
    """Test ParallelRunner functionality."""

    def test_init(self):
        """Test ParallelRunner initialization."""
        runner = ParallelRunner()
        assert runner.max_workers is None
        assert runner.verbose is True
        assert runner.results == []

        runner = ParallelRunner(max_workers=4, verbose=False)
        assert runner.max_workers == 4
        assert runner.verbose is False

    def test_run_benchmarks_parallel(self):
        """Test running benchmarks in parallel."""
        runner = ParallelRunner(max_workers=2, verbose=False)

        benchmarks = [MockBenchmark("benchmark1", sleep_time=0.05), MockBenchmark("benchmark2", sleep_time=0.05), MockBenchmark("benchmark3", sleep_time=0.05)]

        start_time = time.time()
        results = runner.run_benchmarks(benchmarks)
        execution_time = time.time() - start_time

        assert len(results) == 3
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert len(runner.results) == 3

        # Check that execution was faster than sequential
        # (allowing some overhead for thread management)
        assert execution_time < 0.15  # Should be much less than 3 * 0.05

        # Check all benchmarks were executed
        result_names = {r.name for r in results}
        assert result_names == {"benchmark1", "benchmark2", "benchmark3"}

    def test_run_benchmarks_with_kwargs(self):
        """Test running parallel benchmarks with kwargs."""
        runner = ParallelRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2")]

        results = runner.run_benchmarks(benchmarks, learning_rate=0.01)

        assert len(results) == 2
        for result in results:
            assert result.metrics["learning_rate"] == 0.01
            assert result.metadata["learning_rate"] == 0.01

    def test_run_benchmarks_with_failure(self):
        """Test parallel execution with one failing benchmark."""
        runner = ParallelRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2", should_fail=True), MockBenchmark("benchmark3")]

        results = runner.run_benchmarks(benchmarks)

        assert len(results) == 3

        # Find the failing result
        failing_result = next(r for r in results if r.name == "benchmark2")
        assert failing_result.metrics["success"] is False
        assert "error" in failing_result.metrics

        # Check successful results
        successful_results = [r for r in results if r.name in ["benchmark1", "benchmark3"]]
        assert len(successful_results) == 2
        for result in successful_results:
            assert result.metrics.get("success", True) is True

    def test_run_benchmarks_verbose(self, capsys):
        """Test verbose output for parallel execution."""
        runner = ParallelRunner(max_workers=2, verbose=True)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2")]

        runner.run_benchmarks(benchmarks)

        captured = capsys.readouterr()
        assert "Running 2 benchmarks in parallel" in captured.out
        assert "Using 2 workers" in captured.out
        assert "Starting: benchmark1" in captured.out
        assert "Starting: benchmark2" in captured.out
        assert "✓ benchmark1 completed" in captured.out
        assert "✓ benchmark2 completed" in captured.out


class TestParametricRunner:
    """Test ParametricRunner functionality."""

    def test_init(self):
        """Test ParametricRunner initialization."""
        runner = ParametricRunner()
        assert runner.verbose is True
        assert runner.results == {}

        runner = ParametricRunner(verbose=False)
        assert runner.verbose is False

    def test_run_parameter_sweep(self):
        """Test parameter sweep functionality."""
        runner = ParametricRunner(verbose=False)
        benchmark = MockBenchmark("test_benchmark")

        parameter_grid = {"learning_rate": [0.01, 0.1], "batch_size": [32, 64]}

        results = runner.run_parameter_sweep(benchmark, parameter_grid)

        assert len(results) == 1
        sweep_key = "test_benchmark_sweep"
        assert sweep_key in results

        sweep_results = results[sweep_key]
        assert len(sweep_results) == 4  # 2 * 2 combinations

        # Check that all parameter combinations were tested
        param_combinations = set()
        for result in sweep_results:
            lr = result.metadata["learning_rate"]
            bs = result.metadata["batch_size"]
            param_combinations.add((lr, bs))

        expected_combinations = {(0.01, 32), (0.01, 64), (0.1, 32), (0.1, 64)}
        assert param_combinations == expected_combinations

        # Check that parameters were passed to benchmark
        for result in sweep_results:
            assert result.metrics["learning_rate"] == result.metadata["learning_rate"]
            assert result.metrics["batch_size"] == result.metadata["batch_size"]

    def test_run_parameter_sweep_single_param(self):
        """Test parameter sweep with single parameter."""
        runner = ParametricRunner(verbose=False)
        benchmark = MockBenchmark("test_benchmark")

        parameter_grid = {"epochs": [10, 20, 30]}

        results = runner.run_parameter_sweep(benchmark, parameter_grid)

        sweep_results = results["test_benchmark_sweep"]
        assert len(sweep_results) == 3

        epochs_tested = {r.metadata["epochs"] for r in sweep_results}
        assert epochs_tested == {10, 20, 30}

    def test_run_parameter_sweep_verbose(self, capsys):
        """Test verbose output for parameter sweep."""
        runner = ParametricRunner(verbose=True)
        benchmark = MockBenchmark("test_benchmark")

        parameter_grid = {"learning_rate": [0.01, 0.1], "batch_size": [32, 64]}

        runner.run_parameter_sweep(benchmark, parameter_grid)

        captured = capsys.readouterr()
        assert "Running parameter sweep for: test_benchmark" in captured.out
        assert "4 parameter combinations" in captured.out
        assert "[1/4]" in captured.out
        assert "[4/4]" in captured.out
        assert "learning_rate" in captured.out
        assert "batch_size" in captured.out


class TestComparisonRunner:
    """Test ComparisonRunner functionality."""

    def test_init(self):
        """Test ComparisonRunner initialization."""
        runner = ComparisonRunner()
        assert runner.verbose is True
        assert runner.comparison_results == {}

        runner = ComparisonRunner(verbose=False)
        assert runner.verbose is False

    def test_run_comparison(self):
        """Test benchmark comparison functionality."""
        runner = ComparisonRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1", metrics={"accuracy": 0.9, "speed": 100}), MockBenchmark("benchmark2", metrics={"accuracy": 0.95, "speed": 80}), MockBenchmark("benchmark3", metrics={"accuracy": 0.85, "speed": 120})]

        results = runner.run_comparison(benchmarks, "accuracy_comparison")

        assert len(results) == 3
        assert "benchmark1" in results
        assert "benchmark2" in results
        assert "benchmark3" in results

        assert all(isinstance(r, BenchmarkResult) for r in results.values())

        # Check that results are stored in comparison_results
        assert "accuracy_comparison" in runner.comparison_results
        assert len(runner.comparison_results["accuracy_comparison"]) == 3

    def test_run_comparison_with_kwargs(self):
        """Test comparison with additional kwargs."""
        runner = ComparisonRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2")]

        results = runner.run_comparison(benchmarks, "test_comparison", learning_rate=0.01, epochs=10)

        for result in results.values():
            assert result.metrics["learning_rate"] == 0.01
            assert result.metrics["epochs"] == 10

    def test_run_comparison_with_failure(self):
        """Test comparison with failing benchmark."""
        runner = ComparisonRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2", should_fail=True)]

        results = runner.run_comparison(benchmarks, "test_comparison")

        assert len(results) == 2
        assert results["benchmark1"].metrics.get("success", True) is True
        assert results["benchmark2"].metrics["success"] is False

    def test_run_comparison_verbose(self, capsys):
        """Test verbose output for comparison."""
        runner = ComparisonRunner(verbose=True)

        benchmarks = [MockBenchmark("benchmark1"), MockBenchmark("benchmark2")]

        runner.run_comparison(benchmarks, "test_comparison")

        captured = capsys.readouterr()
        assert "Running comparison: test_comparison" in captured.out
        assert "Comparing 2 benchmarks" in captured.out
        assert "Running: benchmark1" in captured.out
        assert "Running: benchmark2" in captured.out
        assert "✓ Completed in" in captured.out

    def test_get_comparison_summary_empty(self):
        """Test getting summary for non-existent comparison."""
        runner = ComparisonRunner()

        summary = runner.get_comparison_summary("non_existent")

        assert summary == {}

    def test_get_comparison_summary(self):
        """Test getting comparison summary."""
        runner = ComparisonRunner(verbose=False)

        # Use only successful benchmarks to test common metrics
        benchmarks = [MockBenchmark("benchmark1", metrics={"accuracy": 0.9, "speed": 100, "custom_metric": 1.5}), MockBenchmark("benchmark2", metrics={"accuracy": 0.95, "speed": 80, "custom_metric": 2.0})]

        runner.run_comparison(benchmarks, "test_comparison")
        summary = runner.get_comparison_summary("test_comparison")

        assert summary["comparison_name"] == "test_comparison"
        assert set(summary["benchmarks"]) == {"benchmark1", "benchmark2"}

        # Check execution times
        assert "execution_times" in summary
        assert len(summary["execution_times"]) == 2
        assert all(t > 0 for t in summary["execution_times"].values())

        # Check success rates
        assert summary["success_rates"]["benchmark1"] is True
        assert summary["success_rates"]["benchmark2"] is True

        # Check metric comparisons (should include common metrics)
        assert "accuracy_comparison" in summary
        assert summary["accuracy_comparison"]["benchmark1"] == 0.9
        assert summary["accuracy_comparison"]["benchmark2"] == 0.95

        assert "speed_comparison" in summary
        assert summary["speed_comparison"]["benchmark1"] == 100
        assert summary["speed_comparison"]["benchmark2"] == 80

        assert "custom_metric_comparison" in summary
        assert summary["custom_metric_comparison"]["benchmark1"] == 1.5
        assert summary["custom_metric_comparison"]["benchmark2"] == 2.0

        # Should not include success/error in comparisons
        assert "success_comparison" not in summary
        assert "error_comparison" not in summary

    def test_get_comparison_summary_with_failure(self):
        """Test summary when one benchmark fails - should have no common metrics."""
        runner = ComparisonRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1", metrics={"accuracy": 0.9, "speed": 100}), MockBenchmark("benchmark2", should_fail=True)]

        runner.run_comparison(benchmarks, "test_comparison")
        summary = runner.get_comparison_summary("test_comparison")

        assert summary["comparison_name"] == "test_comparison"
        assert set(summary["benchmarks"]) == {"benchmark1", "benchmark2"}

        # Check success rates
        assert summary["success_rates"]["benchmark1"] is True
        assert summary["success_rates"]["benchmark2"] is False

        # Should not have metric comparisons due to failure
        assert "accuracy_comparison" not in summary
        assert "speed_comparison" not in summary

    def test_get_comparison_summary_no_common_metrics(self):
        """Test summary when benchmarks have no common metrics."""
        runner = ComparisonRunner(verbose=False)

        benchmarks = [MockBenchmark("benchmark1", metrics={"metric_a": 1.0}), MockBenchmark("benchmark2", metrics={"metric_b": 2.0})]

        runner.run_comparison(benchmarks, "test_comparison")
        summary = runner.get_comparison_summary("test_comparison")

        # Should still have basic info
        assert summary["comparison_name"] == "test_comparison"
        assert "execution_times" in summary
        assert "success_rates" in summary

        # Should not have metric comparisons
        assert "metric_a_comparison" not in summary
        assert "metric_b_comparison" not in summary


class TestIntegration:
    """Integration tests for runners."""

    def test_standard_to_parallel_consistency(self):
        """Test that StandardRunner and ParallelRunner produce consistent results."""
        # Create identical benchmarks
        benchmarks1 = [MockBenchmark(f"benchmark{i}") for i in range(3)]
        benchmarks2 = [MockBenchmark(f"benchmark{i}") for i in range(3)]

        # Run with StandardRunner
        standard_runner = StandardRunner(verbose=False)
        standard_results = []
        for benchmark in benchmarks1:
            result = standard_runner.run_benchmark(benchmark)
            standard_results.append(result)

        # Run with ParallelRunner
        parallel_runner = ParallelRunner(verbose=False)
        parallel_results = parallel_runner.run_benchmarks(benchmarks2)

        # Sort results by name for comparison
        standard_results.sort(key=lambda x: x.name)
        parallel_results.sort(key=lambda x: x.name)

        # Check that results are equivalent (excluding timing differences)
        assert len(standard_results) == len(parallel_results)
        for s_result, p_result in zip(standard_results, parallel_results):
            assert s_result.name == p_result.name
            assert s_result.metrics == p_result.metrics

    def test_runner_with_benchmark_suite(self):
        """Test that runners work correctly with BenchmarkSuite."""
        suite = BenchmarkSuite("integration_suite")
        suite.add_benchmark(MockBenchmark("benchmark1"))
        suite.add_benchmark(MockBenchmark("benchmark2"))

        # Test StandardRunner with suite
        runner = StandardRunner(verbose=False)
        results = runner.run_suite(suite)

        assert len(results) == 2
        assert suite.results == results
        assert all(r.name in ["benchmark1", "benchmark2"] for r in results)
