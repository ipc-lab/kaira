#!/usr/bin/env python3
"""
===========================================
New Results Management System Demo
===========================================

This example demonstrates the new organized results management system in Kaira,
showcasing automatic directory structuring, experiment naming, suite management,
result comparison, and maintenance features.
"""
import time

from kaira.benchmarks.base import BaseBenchmark, BenchmarkSuite
from kaira.benchmarks.results_manager import BenchmarkResultsManager
from kaira.benchmarks.runners import StandardRunner


class ExampleBenchmark(BaseBenchmark):
    """Example benchmark for demonstration purposes."""

    def __init__(self, name: str, description: str = "", delay: float = 0.1):
        super().__init__(name, description)
        self.delay = delay

    def setup(self, **kwargs) -> None:
        """Setup benchmark environment."""
        super().setup(**kwargs)

    def run(self, **kwargs) -> dict:
        """Run the benchmark and return metrics."""
        # Simulate benchmark execution
        time.sleep(self.delay)

        # Return some example metrics
        return {"throughput": 1000 / self.delay, "latency": self.delay, "success": True, "memory_usage": 100 + self.delay * 50, "accuracy": 0.95 + (0.05 * (1 - self.delay))}  # Operations per second  # Seconds  # MB  # Percentage


def demonstrate_basic_usage():
    """Demonstrate basic usage of the new results system."""
    print("=" * 60)
    print("1. Basic Benchmark Results Management")
    print("=" * 60)

    # Create a results manager
    results_manager = BenchmarkResultsManager("example_results")

    # Create and run a simple benchmark
    benchmark = ExampleBenchmark("Performance Test", "Example benchmark for testing", delay=0.2)
    result = benchmark.execute()

    # Save the result
    saved_path = results_manager.save_benchmark_result(result, category="benchmarks", experiment_name="demo_experiment")

    print("Saved benchmark result to:", saved_path)

    # List available results
    results = results_manager.list_results(category="benchmarks")
    print(f"Found {len(results)} benchmark results")

    return results_manager


def demonstrate_suite_management(results_manager):
    """Demonstrate benchmark suite management."""
    print("\n" + "=" * 60)
    print("2. Benchmark Suite Management")
    print("=" * 60)

    # Create a benchmark suite
    suite = BenchmarkSuite("Performance Suite", "Collection of performance benchmarks")

    # Add multiple benchmarks to the suite
    benchmarks = [ExampleBenchmark("Fast Benchmark", "Quick test", delay=0.1), ExampleBenchmark("Medium Benchmark", "Medium test", delay=0.2), ExampleBenchmark("Slow Benchmark", "Thorough test", delay=0.3)]

    for benchmark in benchmarks:
        suite.benchmarks.append(benchmark)

    # Run the suite using the StandardRunner
    runner = StandardRunner(verbose=True, results_manager=results_manager)
    suite_results = runner.run_suite(suite, experiment_name="demo_experiment")

    print(f"\nSuite completed with {len(suite_results)} results")

    # The results are automatically saved by the runner
    suite_files = results_manager.list_results(category="suites")
    print(f"Found {len(suite_files)} suite-related files")


def demonstrate_comparison_and_analysis(results_manager):
    """Demonstrate result comparison and analysis features."""
    print("\n" + "=" * 60)
    print("3. Result Comparison and Analysis")
    print("=" * 60)

    # Get all available results
    all_results = results_manager.list_results()

    if len(all_results) >= 2:
        # Create a comparison report
        comparison_path = results_manager.create_comparison_report(all_results[:3], "demo_comparison")  # Compare first 3 results
        print("Created comparison report:", comparison_path)

        # Load and display a result
        sample_result = results_manager.load_benchmark_result(all_results[0])
        print("\nSample result:", sample_result.name)
        print(f"  Execution time: {sample_result.execution_time:.3f}s")
        print(f"  Key metrics: {sample_result.metrics}")


def demonstrate_maintenance_features(results_manager):
    """Demonstrate maintenance and cleanup features."""
    print("\n" + "=" * 60)
    print("4. Maintenance and Cleanup")
    print("=" * 60)

    # Archive old results (in a real scenario, you'd set a meaningful days_old value)
    print("Archiving old results...")
    results_manager.archive_old_results(days_old=0)  # Archive everything for demo

    # Clean up empty directories
    print("Cleaning up empty directories...")
    results_manager.cleanup_empty_directories()

    # Show directory structure
    print(f"\nFinal directory structure in {results_manager.base_dir}:")
    for item in sorted(results_manager.base_dir.rglob("*")):
        if item.is_dir():
            print(f"  📁 {item.relative_to(results_manager.base_dir)}/")
        else:
            print(f"  📄 {item.relative_to(results_manager.base_dir)}")


def main():
    """Main demonstration function."""
    print("Kaira Benchmark Results Management Demo")
    print("This script demonstrates the new organized benchmark results system.")

    try:
        # 1. Basic usage
        results_manager = demonstrate_basic_usage()

        # 2. Suite management
        demonstrate_suite_management(results_manager)

        # 3. Comparison and analysis
        demonstrate_comparison_and_analysis(results_manager)

        # 4. Maintenance features
        demonstrate_maintenance_features(results_manager)

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nKey benefits of the new system:")
        print("• Organized directory structure")
        print("• Automatic file naming and timestamping")
        print("• Suite-level result management")
        print("• Built-in comparison and analysis tools")
        print("• Maintenance and archiving features")

        print("\nCheck the 'example_results' directory to see the organized structure.")

    except Exception as e:
        print("Error during demonstration:", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
