"""
Example: Comparing different modulation schemes

This example demonstrates how to use the benchmarking system to compare
the performance of different modulation schemes.
"""

import matplotlib.pyplot as plt

from kaira.benchmarks import BenchmarkConfig, ComparisonRunner, create_benchmark


def compare_modulation_schemes():
    """Compare BER performance of different modulation schemes."""
    print("Comparing Modulation Schemes...")

    # Create benchmarks for different modulation schemes
    bpsk_benchmark = create_benchmark("ber_simulation", modulation="bpsk")

    # For this example, we'll just use BPSK, but in a real implementation
    # you would have multiple modulation schemes
    benchmarks = [bpsk_benchmark]

    # Configure comparison - use block_length instead of num_bits
    config = BenchmarkConfig(name="modulation_comparison", snr_range=list(range(-10, 11)), block_length=50000, verbose=True)

    # Run comparison with num_bits as runtime parameter
    runner = ComparisonRunner(verbose=True)
    results = runner.run_comparison(benchmarks, "Modulation Scheme Comparison", num_bits=50000, **config.to_dict())

    # Get comparison summary
    summary = runner.get_comparison_summary("Modulation Scheme Comparison")

    print("\nComparison Summary:")
    print(f"Benchmarks compared: {', '.join(summary['benchmarks'])}")
    for name, time in summary["execution_times"].items():
        print(f"  {name}: {time:.2f}s")

    # Plot comparison results
    plt.figure(figsize=(12, 8))

    for name, result in results.items():
        plt.semilogy(result.metrics["snr_range"], result.metrics["ber_simulated"], "o-", label=f"{name} (Simulated)")
        plt.semilogy(result.metrics["snr_range"], result.metrics["ber_theoretical"], "--", label=f"{name} (Theoretical)")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("Modulation Scheme Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def parameter_sweep_example():
    """Demonstrate parameter sweep functionality."""
    print("\nRunning Parameter Sweep Example...")

    from kaira.benchmarks.runners import ParametricRunner

    # Create benchmark
    ber_benchmark = create_benchmark("ber_simulation", modulation="bpsk")

    # Define parameter grid
    parameter_grid = {"num_bits": [10000, 50000, 100000], "snr_range": [list(range(-5, 6)), list(range(-10, 11)), list(range(-15, 16))]}

    # Run parameter sweep
    runner = ParametricRunner(verbose=True)
    sweep_results = runner.run_parameter_sweep(ber_benchmark, parameter_grid)

    print("\nParameter Sweep Completed!")
    print(f"Total configurations tested: {len(list(sweep_results.values())[0])}")


if __name__ == "__main__":
    compare_modulation_schemes()
    parameter_sweep_example()

    print("\nComparison examples completed!")
