Kaira Benchmarking Examples
===========================

This directory contains examples demonstrating how to use the Kaira benchmarking system to evaluate communication system performance.

.. note::
   These examples are also included in the main documentation gallery for integrated browsing and reference.

Examples
--------

plot_basic_usage.py
~~~~~~~~~~~~~~~~~~~

Demonstrates basic benchmark usage including:

- Running individual benchmarks (BER simulation, throughput test)
- Creating and running benchmark suites
- Saving and analyzing results

plot_comparison_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Shows how to compare different approaches:

- Comparing modulation schemes
- Parameter sweep functionality
- Visualization of comparison results

plot_demo_new_results_system.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates the new organized results management system:

- Automatic directory structure creation
- Organized result saving with experiment names
- Benchmark suite management
- Result comparison and analysis tools
- Maintenance and archiving features
- Integration with existing benchmark runners

kaira_benchmark.py (CLI Script)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located in ``scripts/kaira_benchmark.py``, this provides a complete command-line interface:

- Command-line argument parsing and configuration
- Parallel and sequential benchmark execution
- Suite management and result organization
- Integration with the results management system

plot_visualization_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates comprehensive benchmark result visualization:

- BER curve plotting and analysis
- Throughput performance visualization
- Modulation scheme comparisons
- Performance summary generation

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

    # Results are automatically organized in the structured directory system
    saved_files = runner.save_all_results(experiment_name="my_experiment")
    print(f"Results saved to: {saved_files}")

Using the New Results Management System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from kaira.benchmarks import StandardRunner
    from kaira.benchmarks.results_manager import BenchmarkResultsManager

    # Create a custom results manager
    results_manager = BenchmarkResultsManager("my_results")

    # Use it with a runner
    runner = StandardRunner(results_manager=results_manager)

    # Run benchmarks - results are automatically organized
    result = runner.run_benchmark(ber_benchmark, **config.to_dict())

    # Results are saved in organized directory structure:
    # my_results/benchmarks/experiment_name/benchmark_files.json

Available Benchmarks
--------------------

- ``ber_simulation``: Bit Error Rate simulation for various modulation schemes
- ``channel_capacity``: Channel capacity calculations
- ``throughput_test``: System throughput evaluation
- ``latency_test``: System latency measurement
- ``model_complexity``: Model computational complexity analysis

Configuration Options
---------------------

The ``BenchmarkConfig`` class provides various configuration options:

- ``snr_range``: Range of SNR values to test
- ``num_bits``: Number of bits for simulation
- ``num_trials``: Number of trial runs
- ``device``: Computation device ("auto", "cpu", "cuda")
- ``verbose``: Enable verbose output
- ``save_results``: Save benchmark results

Running Examples
----------------

.. code-block:: bash

    cd examples/benchmarks
    python plot_basic_usage.py
    python plot_comparison_example.py
    python plot_demo_new_results_system.py
    python plot_visualization_example.py

Results will be saved in the ``./benchmark_results`` directory.

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

You can also run benchmarks using the CLI tool:

.. code-block:: bash

    # List available benchmarks
    python scripts/kaira_benchmark.py --list

    # Run a single benchmark
    python scripts/kaira_benchmark.py --benchmark ber_simulation --config fast

    # Run multiple benchmarks in parallel
    python scripts/kaira_benchmark.py --benchmark ber_simulation throughput_test --parallel

    # Run a comprehensive benchmark suite
    python scripts/kaira_benchmark.py --suite --output ./my_results
