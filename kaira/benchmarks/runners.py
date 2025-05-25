"""Benchmark runners for executing benchmarks in different modes."""

import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseBenchmark, BenchmarkResult, BenchmarkSuite
from .results_manager import BenchmarkResultsManager


class StandardRunner:
    """Standard sequential benchmark runner."""

    def __init__(self, verbose: bool = True, save_results: bool = True, results_manager: Optional[BenchmarkResultsManager] = None):
        """Initialize standard benchmark runner.

        Args:
            verbose: Whether to print verbose output
            save_results: Whether to save results automatically
            results_manager: Custom results manager (creates default if None)
        """
        self.verbose = verbose
        self.save_results = save_results
        self.results: List[BenchmarkResult] = []
        self.results_manager = results_manager or BenchmarkResultsManager()

    def run_benchmark(self, benchmark: BaseBenchmark, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if self.verbose:
            print(f"Running benchmark: {benchmark.name}")

        result = benchmark.execute(**kwargs)

        if self.verbose:
            success = result.metrics.get("success", True)
            status = "✓" if success else "✗"
            print(f"  {status} Completed in {result.execution_time:.2f}s")

        self.results.append(result)
        return result

    def run_suite(self, suite: BenchmarkSuite, **kwargs) -> List[BenchmarkResult]:
        """Run a benchmark suite."""
        if self.verbose:
            print(f"Running benchmark suite: {suite.name}")
            print(f"  {len(suite.benchmarks)} benchmarks to run")

        results = []
        for i, benchmark in enumerate(suite.benchmarks, 1):
            if self.verbose:
                print(f"  [{i}/{len(suite.benchmarks)}] {benchmark.name}")

            result = self.run_benchmark(benchmark, **kwargs)
            results.append(result)

        if self.save_results:
            suite.results = results
            # Save suite results using the new results manager
            self.results_manager.save_suite_results(results, suite.name, experiment_name=kwargs.get("experiment_name"))

        return results

    def save_all_results(self, experiment_name: Optional[str] = None) -> Dict[str, Path]:
        """Save all results using the results manager.

        Args:
            experiment_name: Optional experiment name for grouping results

        Returns:
            Dictionary mapping result names to saved file paths
        """
        saved_files = {}
        for result in self.results:
            filepath = self.results_manager.save_benchmark_result(result, category="benchmarks", experiment_name=experiment_name)
            saved_files[result.name] = filepath

        return saved_files


class ParallelRunner:
    """Parallel benchmark runner using thread pool."""

    def __init__(self, max_workers: Optional[int] = None, verbose: bool = True):
        """Initialize parallel benchmark runner.

        Args:
            max_workers: Maximum number of worker threads (None for default)
            verbose: Whether to print verbose output
        """
        self.max_workers = max_workers
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def run_benchmarks(self, benchmarks: List[BaseBenchmark], **kwargs) -> List[BenchmarkResult]:
        """Run multiple benchmarks in parallel."""
        if self.verbose:
            print(f"Running {len(benchmarks)} benchmarks in parallel")
            print(f"Using {self.max_workers or 'default'} workers")

        def run_single(benchmark):
            """Run a single benchmark and return result."""
            if self.verbose:
                print(f"Starting: {benchmark.name}")
            result = benchmark.execute(**kwargs)
            if self.verbose:
                success = result.metrics.get("success", True)
                status = "✓" if success else "✗"
                print(f"  {status} {benchmark.name} completed in {result.execution_time:.2f}s")
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_benchmark = {executor.submit(run_single, benchmark): benchmark for benchmark in benchmarks}

            results = []
            for future in concurrent.futures.as_completed(future_to_benchmark):
                result = future.result()
                results.append(result)

        self.results.extend(results)
        return results


class ParametricRunner:
    """Runner for sweeping parameters across benchmarks."""

    def __init__(self, verbose: bool = True):
        """Initialize parametric runner.

        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.results: Dict[str, List[BenchmarkResult]] = {}

    def run_parameter_sweep(self, benchmark: BaseBenchmark, parameter_grid: Dict[str, List[Any]]) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark with parameter sweep."""
        if self.verbose:
            print(f"Running parameter sweep for: {benchmark.name}")

        # Generate all parameter combinations
        import itertools

        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(itertools.product(*param_values))

        if self.verbose:
            print(f"  {len(param_combinations)} parameter combinations")

        results = []
        for i, combination in enumerate(param_combinations, 1):
            params = dict(zip(param_names, combination))

            if self.verbose:
                print(f"  [{i}/{len(param_combinations)}] {params}")

            result = benchmark.execute(**params)
            result.metadata.update(params)
            results.append(result)

        sweep_key = f"{benchmark.name}_sweep"
        self.results[sweep_key] = results
        return {sweep_key: results}


class ComparisonRunner:
    """Runner for comparing multiple benchmarks on the same task."""

    def __init__(self, verbose: bool = True):
        """Initialize comparison runner.

        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.comparison_results: Dict[str, Dict[str, BenchmarkResult]] = {}

    def run_comparison(self, benchmarks: List[BaseBenchmark], comparison_name: str, **kwargs) -> Dict[str, BenchmarkResult]:
        """Run comparison between multiple benchmarks."""
        if self.verbose:
            print(f"Running comparison: {comparison_name}")
            print(f"  Comparing {len(benchmarks)} benchmarks")

        results = {}
        for benchmark in benchmarks:
            if self.verbose:
                print(f"  Running: {benchmark.name}")

            result = benchmark.execute(**kwargs)
            results[benchmark.name] = result

            if self.verbose:
                success = result.metrics.get("success", True)
                status = "✓" if success else "✗"
                print(f"    {status} Completed in {result.execution_time:.2f}s")

        self.comparison_results[comparison_name] = results
        return results

    def get_comparison_summary(self, comparison_name: str) -> Dict[str, Any]:
        """Get summary of comparison results."""
        if comparison_name not in self.comparison_results:
            return {}

        results = self.comparison_results[comparison_name]
        summary = {"comparison_name": comparison_name, "benchmarks": list(results.keys()), "execution_times": {name: result.execution_time for name, result in results.items()}, "success_rates": {name: result.metrics.get("success", True) for name, result in results.items()}}

        # Add metric comparisons if available
        common_metrics: set[str] = set()
        for result in results.values():
            if common_metrics:
                common_metrics &= set(result.metrics.keys())
            else:
                common_metrics = set(result.metrics.keys())

        for metric in common_metrics:
            if metric in ["success", "error"]:
                continue
            summary[f"{metric}_comparison"] = {name: result.metrics.get(metric) for name, result in results.items()}

        return summary
