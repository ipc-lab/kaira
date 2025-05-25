Kaira Benchmarking System
=========================

The Kaira benchmarking system provides standardized benchmarks for evaluating communication system components and deep learning models. This system enables fair comparison of different approaches and reproducible performance evaluation.

Overview
--------

The benchmarking system consists of:

- **Base classes** for creating custom benchmarks
- **Standard benchmarks** for common communication tasks
- **Metrics** for evaluating performance
- **Runners** for executing benchmarks in different modes
- **Configuration management** for reproducible experiments
- **CLI tool** for command-line usage

Quick Start
-----------

Basic usage with the new organized results system::

    from kaira.benchmarks import get_benchmark, StandardRunner, BenchmarkConfig

    # Create a benchmark
    ber_benchmark = get_benchmark("ber_simulation")(modulation="bpsk")

    # Configure the benchmark
    config = BenchmarkConfig(
        snr_range=list(range(-5, 11)),
        num_bits=100000
    )

    # Run the benchmark with automatic result organization
    runner = StandardRunner()
    result = runner.run_benchmark(ber_benchmark, **config.to_dict())

    # Results are automatically saved to organized directory structure
    print(f"BER results: {result.metrics['ber_simulated']}")

    # Access saved results using the results manager
    saved_files = runner.save_all_results(experiment_name="ber_evaluation")
    print(f"Results saved to: {saved_files}")

Traditional usage (still supported)::

    # Manual result saving
    result.save("benchmark_result.json")

Available Benchmarks
-------------------

Standard Communication Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ber_simulation**: Bit Error Rate simulation for various modulation schemes
- **channel_capacity**: Shannon channel capacity calculations
- **throughput_test**: System throughput evaluation
- **latency_test**: System latency measurement
- **model_complexity**: Model computational complexity analysis

Custom Benchmarks
~~~~~~~~~~~~~~~~~

You can create custom benchmarks by inheriting from ``BaseBenchmark``::

    from kaira.benchmarks import BaseBenchmark, register_benchmark

    @register_benchmark("my_benchmark")
    class MyBenchmark(BaseBenchmark):
        def setup(self, **kwargs):
            super().setup(**kwargs)
            # Initialize benchmark

        def run(self, **kwargs):
            # Run benchmark and return metrics
            return {"success": True, "metric_value": 42}

Configuration
-------------

Predefined Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **fast**: Quick testing configuration
- **accurate**: High-accuracy configuration for publication results
- **comprehensive**: Full evaluation with all metrics
- **gpu**: GPU-optimized configuration
- **minimal**: Minimal configuration for CI/CD

Custom Configuration::

    config = BenchmarkConfig(
        name="my_config",
        num_trials=10,
        snr_range=list(range(-10, 16)),
        device="cuda",
        verbose=True
    )

Benchmark Execution
------------------

Sequential Execution::

    runner = StandardRunner(verbose=True)
    result = runner.run_benchmark(benchmark, **config.to_dict())

Parallel Execution::

    runner = ParallelRunner(max_workers=4)
    results = runner.run_benchmarks(benchmarks, **config.to_dict())

Benchmark Suites::

    suite = BenchmarkSuite("My Suite")
    suite.add_benchmark(benchmark1)
    suite.add_benchmark(benchmark2)

    results = runner.run_suite(suite, **config.to_dict())

Comparison and Analysis::

    runner = ComparisonRunner()
    results = runner.run_comparison(
        [benchmark1, benchmark2],
        "Algorithm Comparison",
        **config.to_dict()
    )

Metrics and Analysis
-------------------

Standard Metrics
~~~~~~~~~~~~~~~

The ``StandardMetrics`` class provides common communication system metrics:

- Bit Error Rate (BER)
- Block Error Rate (BLER)
- Signal-to-Noise Ratio (SNR)
- Mutual Information
- Throughput
- Latency statistics
- Channel capacity
- Confidence intervals

Example::

    from kaira.benchmarks import StandardMetrics

    ber = StandardMetrics.bit_error_rate(transmitted, received)
    snr = StandardMetrics.signal_to_noise_ratio(signal, noise)
    capacity = StandardMetrics.channel_capacity(snr_db=10.0)

Results Management
-----------------

Kaira provides an organized results management system that automatically structures benchmark results in a clean directory hierarchy.

Results Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark system creates the following directory structure::

    results/
    ├── benchmarks/          # Individual benchmark results
    │   ├── experiment_name/
    │   └── benchmark_files.json
    ├── suites/             # Benchmark suite results
    │   ├── suite_name/
    │   └── summary.json
    ├── experiments/        # Experimental runs
    ├── comparisons/        # Comparative studies
    ├── archives/          # Archived old results
    ├── configs/           # Configuration files
    ├── logs/              # Execution logs
    └── summaries/         # Summary reports

Using the Results Manager
~~~~~~~~~~~~~~~~~~~~~~~~

The new results management system provides automated organization::

    from kaira.benchmarks import StandardRunner, BenchmarkResultsManager

    # Create a results manager (uses 'results/' directory by default)
    results_manager = BenchmarkResultsManager("my_results")

    # Create a runner with the results manager
    runner = StandardRunner(results_manager=results_manager)

    # Run benchmarks - results are automatically saved and organized
    result = runner.run_benchmark(benchmark, experiment_name="my_experiment")

    # Results are automatically saved to:
    # my_results/benchmarks/my_experiment/benchmark_name_timestamp_id.json

Manual Results Management
~~~~~~~~~~~~~~~~~~~~~~~~

You can also manage results manually::

    # Save individual result with automatic organization
    results_manager = BenchmarkResultsManager()
    saved_path = results_manager.save_benchmark_result(
        result,
        category="benchmarks",
        experiment_name="my_experiment"
    )

    # Save suite results
    saved_files = results_manager.save_suite_results(
        results_list,
        suite_name="performance_suite",
        experiment_name="my_experiment"
    )

    # List available results
    all_results = results_manager.list_results()
    experiment_results = results_manager.list_results(
        category="benchmarks",
        experiment_name="my_experiment"
    )

    # Load results
    result = results_manager.load_benchmark_result(result_path)

Loading and Analysis
~~~~~~~~~~~~~~~~~~~

    # Load results using the results manager
    results_manager = BenchmarkResultsManager()
    result_paths = results_manager.list_results(category="benchmarks")

    for path in result_paths:
        result = results_manager.load_benchmark_result(path)
        print(f"Result: {result.name}, Time: {result.execution_time:.2f}s")

    # Create comparison reports
    comparison_path = results_manager.create_comparison_report(
        result_paths[:3],
        "algorithm_comparison"
    )

Results Maintenance
~~~~~~~~~~~~~~~~~~

The system includes maintenance features for long-term management::

    # Archive old results (older than 30 days)
    results_manager.archive_old_results(days_old=30)

    # Clean up empty directories
    results_manager.cleanup_empty_directories()

Command Line Interface
---------------------

The ``kaira-benchmark`` CLI tool provides easy access to benchmarks::

    # List available benchmarks
    kaira-benchmark --list

    # Run a single benchmark
    kaira-benchmark --benchmark ber_simulation --config fast

    # Run multiple benchmarks in parallel
    kaira-benchmark --benchmark ber_simulation throughput_test --parallel

    # Run benchmark suite
    kaira-benchmark --suite --config comprehensive --output ./results

    # Custom parameters
    kaira-benchmark --benchmark ber_simulation --snr-range -5 10 --num-bits 50000

Best Practices
--------------

1. **Use appropriate configurations** for your use case (fast for development, accurate for publications)

2. **Set random seeds** for reproducible results::

    config = BenchmarkConfig(seed=42)

3. **Save raw data** for important experiments::

    config = BenchmarkConfig(save_raw_data=True)

4. **Use confidence intervals** for statistical analysis::

    config = BenchmarkConfig(
        calculate_confidence_intervals=True,
        confidence_level=0.95
    )

5. **Monitor memory usage** for large experiments::

    config = BenchmarkConfig(memory_limit_mb=8192)

Examples
--------

See the ``examples/benchmarks/`` directory for comprehensive examples:

- ``basic_usage.py``: Basic benchmark usage
- ``comparison_example.py``: Comparing different approaches
- ``custom_benchmark.py``: Creating custom benchmarks
- ``demo_new_results_system.py``: New results management system demonstration

Results Management Example
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``demo_new_results_system.py`` example demonstrates the complete workflow::

    # Create and configure results manager
    results_manager = BenchmarkResultsManager("example_results")

    # Run benchmarks with automatic result organization
    runner = StandardRunner(results_manager=results_manager)

    # Create and run benchmark suites
    suite = BenchmarkSuite("Performance Suite")
    # ... add benchmarks to suite
    results = runner.run_suite(suite, experiment_name="demo_experiment")

    # Results are automatically organized in structured directories

API Reference
-------------

.. automodule:: kaira.benchmarks
   :members:
   :undoc-members:
   :show-inheritance:
