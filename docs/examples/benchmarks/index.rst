:orphan:

.. _sphx_glr_examples_benchmarks:

Benchmarking Examples
=========================================

This section contains examples demonstrating how to use the Kaira benchmarking system to evaluate communication system performance and machine learning models.

The benchmarking system provides:

* Standardized benchmarks for BER simulation, throughput testing, latency measurement, and model complexity analysis
* Organized results management with automatic directory structure
* Comparison and visualization tools
* Suite management for running multiple benchmarks
* Integration with existing Kaira components

.. raw:: html

    <div class="benchmark-examples-intro">
        <p>These examples show you how to:</p>
        <ul>
            <li><strong>Run individual benchmarks</strong> with custom configurations</li>
            <li><strong>Create benchmark suites</strong> for comprehensive evaluation</li>
            <li><strong>Compare different approaches</strong> using parameter sweeps</li>
            <li><strong>Organize and manage results</strong> using the new results management system</li>
            <li><strong>Visualize performance metrics</strong> and generate reports</li>
        </ul>
    </div>

Examples
--------------

.. raw:: html

    <div class="example-grid">
        <div class="example-card">
            <h3><a href="../../../examples/benchmarks/plot_basic_usage.py">Basic Usage</a></h3>
            <p>Introduction to running benchmarks and saving results</p>
            <ul>
                <li>Running individual benchmarks (BER simulation, throughput test)</li>
                <li>Creating and running benchmark suites</li>
                <li>Saving and analyzing results</li>
            </ul>
        </div>

        <div class="example-card">
            <h3><a href="../../../examples/benchmarks/plot_comparison_example.py">Comparison Example</a></h3>
            <p>Comparing different approaches and configurations</p>
            <ul>
                <li>Comparing modulation schemes</li>
                <li>Parameter sweep functionality</li>
                <li>Visualization of comparison results</li>
            </ul>
        </div>

        <div class="example-card">
            <h3><a href="../../../examples/benchmarks/plot_demo_new_results_system.py">Results Management System</a></h3>
            <p>Comprehensive demonstration of the organized results management</p>
            <ul>
                <li>Automatic directory structure creation</li>
                <li>Organized result saving with experiment names</li>
                <li>Benchmark suite management</li>
                <li>Result comparison and analysis tools</li>
                <li>Maintenance and archiving features</li>
            </ul>
        </div>

        <div class="example-card">
            <h3><a href="../../../scripts/kaira_benchmark.py">Command Line Interface</a></h3>
            <p>Complete CLI tool for running benchmarks from the command line</p>
            <ul>
                <li>Command-line argument parsing and configuration</li>
                <li>Parallel and sequential benchmark execution</li>
                <li>Suite management and result organization</li>
                <li>Integration with the results management system</li>
            </ul>
        </div>

        <div class="example-card">
            <h3><a href="../../../examples/benchmarks/plot_visualization_example.py">Visualization Example</a></h3>
            <p>Comprehensive demonstration of benchmark result visualization</p>
            <ul>
                <li>BER curve plotting and analysis</li>
                <li>Throughput performance visualization</li>
                <li>Modulation scheme comparisons</li>
                <li>Performance summary generation</li>
            </ul>
        </div>
    </div>

Quick Start
-----------------

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

    # Save results
    saved_files = runner.save_all_results(experiment_name="my_experiment")

Available Benchmarks
--------------------------------------

* **ber_simulation**: Bit Error Rate simulation for various modulation schemes
* **channel_capacity**: Channel capacity calculations
* **throughput_test**: System throughput evaluation
* **latency_test**: System latency measurement
* **model_complexity**: Model computational complexity analysis

Results Organization
--------------------------------------

The benchmarking system automatically organizes results in a structured directory layout:

.. code-block:: text

    results/
    ├── benchmarks/          # Individual benchmark results
    ├── suites/             # Benchmark suite results
    ├── experiments/        # Experimental runs
    ├── comparisons/        # Comparative studies
    ├── archives/           # Archived old results
    ├── configs/            # Configuration files
    ├── logs/               # Execution logs
    └── summaries/          # Summary reports

For more details, see the :doc:`../../benchmarks` documentation.

.. raw:: html

    <div class="example-footer">
        <p><strong>Running the Examples:</strong></p>
        <pre><code>cd examples/benchmarks
python plot_basic_usage.py
python plot_comparison_example.py
python plot_demo_new_results_system.py
python plot_visualization_example.py

# Or use the CLI tool
cd ../../scripts
python kaira_benchmark.py --list
python kaira_benchmark.py --benchmark ber_simulation --config fast</code></pre>
        <p>Results will be saved in the <code>./benchmark_results</code> directory.</p>
    </div>
