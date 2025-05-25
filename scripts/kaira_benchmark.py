#!/usr/bin/env python3
"""Kaira Benchmark CLI.

Command-line interface for running Kaira benchmarks.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kaira.benchmarks import (
    BenchmarkConfig,
    BenchmarkSuite,
    ParallelRunner,
    StandardRunner,
    get_benchmark,
    get_config,
    list_benchmarks,
    list_configs,
)
from kaira.benchmarks.results_manager import BenchmarkResultsManager


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Kaira Benchmark CLI - Run standardized communication system benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available benchmarks
  kaira-benchmark --list

  # Run a single benchmark
  kaira-benchmark --benchmark ber_simulation --config fast

  # Run multiple benchmarks
  kaira-benchmark --benchmark ber_simulation throughput_test --parallel

  # Run with custom configuration
  kaira-benchmark --benchmark ber_simulation --snr-range -5 10 --num-bits 50000

  # Run benchmark suite and save results
  kaira-benchmark --suite --output ./results --config comprehensive
        """,
    )

    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list", action="store_true", help="List available benchmarks and configurations")
    action_group.add_argument("--benchmark", nargs="+", metavar="NAME", help="Run specific benchmark(s)")
    action_group.add_argument("--suite", action="store_true", help="Run a predefined benchmark suite")

    # Configuration arguments
    parser.add_argument("--config", type=str, choices=list_configs(), default="fast", help="Use predefined configuration (default: fast)")
    parser.add_argument("--config-file", type=Path, help="Load configuration from JSON file")

    # Execution options
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--output", type=Path, default="./benchmark_results", help="Output directory for results (default: ./benchmark_results)")

    # Benchmark-specific options
    parser.add_argument("--snr-range", nargs=2, type=int, metavar=("MIN", "MAX"), help="SNR range for communication benchmarks")
    parser.add_argument("--num-bits", type=int, help="Number of bits for simulation")
    parser.add_argument("--num-trials", type=int, help="Number of trial runs")
    parser.add_argument("--modulation", type=str, help="Modulation scheme for BER simulation")

    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except errors")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device (default: auto)")

    return parser


def list_available_items():
    """List available benchmarks and configurations."""
    print("Available Benchmarks:")
    benchmarks = list_benchmarks()
    if benchmarks:
        for benchmark in sorted(benchmarks):
            print(f"  - {benchmark}")
    else:
        print("  No benchmarks available")

    print("\nAvailable Configurations:")
    configs = list_configs()
    for config_name in sorted(configs):
        config = get_config(config_name)
        print(f"  - {config_name}: {config.description}")


def create_config_from_args(args) -> BenchmarkConfig:
    """Create benchmark configuration from command-line arguments."""
    if args.config_file:
        config = BenchmarkConfig.load(args.config_file)
    else:
        config = get_config(args.config)

    # Override with command-line arguments
    overrides: Dict[str, Any] = {}

    if args.snr_range:
        overrides["snr_range"] = [float(x) for x in range(args.snr_range[0], args.snr_range[1] + 1)]
    if args.num_bits:
        overrides["num_bits"] = args.num_bits
    if args.num_trials:
        overrides["num_trials"] = args.num_trials
    if args.device:
        overrides["device"] = args.device
    if args.verbose:
        overrides["verbose"] = args.verbose
    if args.quiet:
        overrides["verbose"] = not args.quiet

    config.update(**overrides)
    return config


def run_single_benchmarks(benchmark_names: List[str], config: BenchmarkConfig, parallel: bool = False, workers: Optional[int] = None) -> List[Any]:
    """Run individual benchmarks."""
    benchmarks = []

    for name in benchmark_names:
        benchmark_class = get_benchmark(name)
        if benchmark_class is None:
            print(f"Error: Unknown benchmark '{name}'", file=sys.stderr)
            print(f"Available benchmarks: {', '.join(list_benchmarks())}", file=sys.stderr)
            sys.exit(1)

        # Create benchmark instance with appropriate parameters
        kwargs = {}
        if name == "ber_simulation" and hasattr(config, "modulation"):
            kwargs["modulation"] = config.get("modulation", "bpsk")

        benchmark = benchmark_class(**kwargs)
        benchmarks.append(benchmark)

    # Run benchmarks
    if parallel:
        parallel_runner = ParallelRunner(max_workers=workers, verbose=config.verbose)
        results = parallel_runner.run_benchmarks(benchmarks, **config.to_dict())
    else:
        standard_runner = StandardRunner(verbose=config.verbose)
        results = []
        for benchmark in benchmarks:
            result = standard_runner.run_benchmark(benchmark, **config.to_dict())
            results.append(result)

    return results


def run_benchmark_suite(config: BenchmarkConfig) -> Tuple[List[Any], BenchmarkSuite]:
    """Run a comprehensive benchmark suite."""
    suite = BenchmarkSuite(name="Kaira Standard Benchmark Suite", description="Comprehensive evaluation of communication system performance")

    # Add available benchmarks to suite
    available_benchmarks = list_benchmarks()

    if "channel_capacity" in available_benchmarks:
        benchmark_class = get_benchmark("channel_capacity")
        if benchmark_class:
            suite.add_benchmark(benchmark_class(name="Channel Capacity Benchmark"))

    if "ber_simulation" in available_benchmarks:
        benchmark_class = get_benchmark("ber_simulation")
        if benchmark_class:
            suite.add_benchmark(benchmark_class(name="BER Simulation Benchmark"))

    if "throughput_test" in available_benchmarks:
        benchmark_class = get_benchmark("throughput_test")
        if benchmark_class:
            suite.add_benchmark(benchmark_class(name="Throughput Test Benchmark"))

    if "latency_test" in available_benchmarks:
        benchmark_class = get_benchmark("latency_test")
        if benchmark_class:
            suite.add_benchmark(benchmark_class(name="Latency Test Benchmark"))

    if "model_complexity" in available_benchmarks:
        benchmark_class = get_benchmark("model_complexity")
        if benchmark_class:
            suite.add_benchmark(benchmark_class(name="Model Complexity Benchmark"))

    if not suite.benchmarks:
        print("Error: No benchmarks available for suite", file=sys.stderr)
        sys.exit(1)

    # Run suite
    runner = StandardRunner(verbose=config.verbose)
    results = runner.run_suite(suite, **config.to_dict())

    # Print summary
    summary = suite.get_summary()
    if not config.get("quiet", False):
        print("\nBenchmark Suite Summary:")
        print(f"  Total benchmarks: {summary['total_benchmarks']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Total execution time: {summary['total_execution_time']:.2f}s")

    return results, suite


def save_results(results, output_dir: Path, suite=None, experiment_name: str = "cli_run"):
    """Save benchmark results using the improved results management system."""
    # Create results manager with the specified output directory
    results_manager = BenchmarkResultsManager(output_dir)

    if suite:
        # Save suite results using the new results manager
        saved_files = results_manager.save_suite_results(suite.results, suite_name=suite.name, experiment_name=experiment_name)
        print(f"Suite results saved to: {output_dir}")
        for name, path in saved_files.items():
            print(f"  {name}: {path.relative_to(output_dir)}")
    else:
        # Save individual results using the new results manager
        saved_files = {}
        for result in results:
            filepath = results_manager.save_benchmark_result(result, category="benchmarks", experiment_name=experiment_name)
            saved_files[result.name] = filepath

        # Create overall summary using the results manager
        if results:
            comparison_path = results_manager.create_comparison_report(list(saved_files.values()), f"{experiment_name}_summary")
            print(f"Individual results saved to: {output_dir}")
            for name, path in saved_files.items():
                print(f"  {name}: {path.relative_to(output_dir)}")
            print(f"Summary report: {comparison_path.relative_to(output_dir)}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_available_items()
        return

    # Create configuration
    config = create_config_from_args(args)

    if not args.quiet:
        print(f"Using configuration: {config.name}")
        if args.verbose:
            print(f"Configuration details: {config.to_json()}")

    try:
        # Run benchmarks
        if args.benchmark:
            results = run_single_benchmarks(args.benchmark, config, parallel=args.parallel, workers=args.workers)
            suite = None
        elif args.suite:
            results, suite = run_benchmark_suite(config)

        # Create a unique experiment name for this CLI run
        from datetime import datetime

        experiment_name = f"cli_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save results
        save_results(results, args.output, suite, experiment_name)

        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
            print("Benchmarks completed successfully!")

    except KeyboardInterrupt:
        print("\nBenchmark execution interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
