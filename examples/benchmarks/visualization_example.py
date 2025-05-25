#!/usr/bin/env python3
"""Visualization Example for Kaira Benchmarks.

This example demonstrates how to create visualizations of benchmark results using the
BenchmarkVisualizer class.
"""

from pathlib import Path

from kaira.benchmarks import BenchmarkConfig, BenchmarkVisualizer, StandardRunner, get_benchmark


def main():
    """Run benchmark visualization example."""
    print("Kaira Benchmark Visualization Example")
    print("=" * 50)

    # Create output directory
    output_dir = Path("./visualization_results")
    output_dir.mkdir(exist_ok=True)

    # Create benchmarks
    print("\n1. Running BER simulation benchmark...")
    ber_benchmark = get_benchmark("ber_simulation")(modulation="bpsk")

    # Configure benchmark
    config = BenchmarkConfig(snr_range=list(range(-2, 11)), num_bits=50000, verbose=True)

    # Run benchmark
    runner = StandardRunner()
    ber_result = runner.run_benchmark(ber_benchmark, **config.to_dict())

    print(f"✓ BER simulation completed in {ber_result.execution_time:.2f}s")

    # Create visualizer
    visualizer = BenchmarkVisualizer(figsize=(12, 8))

    # Plot BER curve
    print("\n2. Creating BER curve visualization...")
    visualizer.plot_ber_curve(ber_result.metrics, save_path=str(output_dir / "ber_curve.png"))
    print("✓ BER curve saved to visualization_results/ber_curve.png")

    # Run throughput benchmark
    print("\n3. Running throughput benchmark...")
    throughput_benchmark = get_benchmark("throughput_test")()
    throughput_result = runner.run_benchmark(throughput_benchmark, data_sizes=[1000, 5000, 10000, 50000, 100000], num_trials=3)

    print(f"✓ Throughput test completed in {throughput_result.execution_time:.2f}s")

    # Plot throughput results
    print("\n4. Creating throughput visualization...")
    visualizer.plot_throughput_comparison([throughput_result.metrics], labels=["Standard Implementation"], save_path=str(output_dir / "throughput_comparison.png"))
    print("✓ Throughput plot saved to visualization_results/throughput_comparison.png")

    # Create comparison plot if we have multiple results
    print("\n5. Running parameter comparison...")

    # Compare different modulation schemes
    modulations = ["bpsk", "qpsk", "16qam"]
    comparison_results = []

    for mod in modulations:
        print(f"   Running {mod.upper()} simulation...")
        mod_benchmark = get_benchmark("ber_simulation")(modulation=mod)
        mod_result = runner.run_benchmark(mod_benchmark, snr_range=list(range(0, 16, 2)), num_bits=20000)
        comparison_results.append(mod_result.metrics)

    # Plot comparison
    print("\n6. Creating modulation comparison plot...")
    visualizer.plot_ber_comparison(comparison_results, labels=[mod.upper() for mod in modulations], save_path=str(output_dir / "modulation_comparison.png"))
    print("✓ Modulation comparison saved to visualization_results/modulation_comparison.png")

    # Create summary statistics plot
    print("\n7. Creating performance summary...")
    visualizer.plot_performance_summary([ber_result, throughput_result], save_path=str(output_dir / "performance_summary.png"))
    print("✓ Performance summary saved to visualization_results/performance_summary.png")

    print("\n" + "=" * 50)
    print("✅ Visualization example completed successfully!")
    print(f"📁 All plots saved to: {output_dir.absolute()}")
    print("\nGenerated visualizations:")
    print("  • ber_curve.png - BER vs SNR curve")
    print("  • throughput_comparison.png - Throughput performance")
    print("  • modulation_comparison.png - Modulation scheme comparison")
    print("  • performance_summary.png - Overall performance summary")


if __name__ == "__main__":
    main()
