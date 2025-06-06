"""
Example: Basic benchmark usage with Kaira

This example demonstrates how to use the Kaira benchmarking system
to evaluate communication system performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Import Kaira benchmarking components
from kaira.benchmarks import BenchmarkConfig, BenchmarkSuite, StandardRunner, create_benchmark


def run_ber_benchmark():
    """Run a BER simulation benchmark."""
    print("Running BER Simulation Benchmark...")

    # Create benchmark instance
    ber_benchmark = create_benchmark("ber_simulation", modulation="bpsk")

    # Configure benchmark
    config = BenchmarkConfig(name="ber_example", snr_range=list(range(-5, 11)), block_length=100000, verbose=True)

    # Run benchmark
    runner = StandardRunner(verbose=True)
    result = runner.run_benchmark(ber_benchmark, **config.to_dict())

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(result.metrics["snr_range"], result.metrics["ber_simulated"], "bo-", label="Simulated")
    plt.semilogy(result.metrics["snr_range"], result.metrics["ber_theoretical"], "r--", label="Theoretical")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("BPSK BER Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Benchmark completed in {result.execution_time:.2f} seconds")
    print("RMSE between simulated and theoretical: {:.6f}".format(result.metrics["rmse"]))


def run_throughput_benchmark():
    """Run a throughput benchmark."""
    print("\nRunning Throughput Benchmark...")

    # Create benchmark instance
    throughput_benchmark = create_benchmark("throughput_test")

    # Configure benchmark - pass payload_sizes as runtime kwargs instead of config
    config = BenchmarkConfig(name="throughput_example", num_trials=5)

    # Run benchmark with payload_sizes as kwargs
    runner = StandardRunner(verbose=True)
    result = runner.run_benchmark(throughput_benchmark, payload_sizes=[1000, 10000, 100000], **config.to_dict())

    # Display results
    print("\nThroughput Results:")
    for size, stats in result.metrics["throughput_results"].items():
        print("  Payload size {}: {:.2f} ± {:.2f} bits/s".format(size, stats["mean"], stats["std"]))

    print("Peak throughput: {:.2f} bits/s".format(result.metrics["peak_throughput"]))


def run_benchmark_suite():
    """Run a complete benchmark suite."""
    print("\nRunning Benchmark Suite...")

    # Create benchmark suite
    suite = BenchmarkSuite(name="Communication System Benchmarks", description="Comprehensive evaluation of communication system performance")

    # Add benchmarks to suite
    suite.add_benchmark(create_benchmark("channel_capacity", channel_type="awgn"))
    suite.add_benchmark(create_benchmark("ber_simulation", modulation="bpsk"))
    suite.add_benchmark(create_benchmark("throughput_test"))
    suite.add_benchmark(create_benchmark("latency_test"))

    # Configure and run suite - use block_length instead of num_bits
    config = BenchmarkConfig(name="suite_example", snr_range=[-5, 0, 5, 10], block_length=10000, verbose=True)

    runner = StandardRunner(verbose=True)
    runner.run_suite(suite, num_bits=10000, **config.to_dict())

    # Get summary
    summary = suite.get_summary()
    print("\nSuite Summary:")
    print("  Total benchmarks: {}".format(summary["total_benchmarks"]))
    print("  Successful: {}".format(summary["successful"]))
    print("  Failed: {}".format(summary["failed"]))
    print("  Total execution time: {:.2f}s".format(summary["total_execution_time"]))

    # Save results
    output_dir = Path("./benchmark_results")
    suite.save_results(output_dir)
    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    # Run individual benchmarks
    run_ber_benchmark()
    run_throughput_benchmark()

    # Run benchmark suite
    run_benchmark_suite()

    print("\nBenchmarking examples completed!")
