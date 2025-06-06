#!/usr/bin/env python3
"""
=================================
Benchmark Visualization Example
=================================

This example demonstrates comprehensive benchmark result visualization in Kaira,
including BER curve plotting, throughput performance, modulation comparisons,
and performance summary generation.
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

    # Configure benchmark - use block_length instead of num_bits
    config = BenchmarkConfig(snr_range=list(range(-2, 11)), block_length=50000, verbose=True)

    # Run benchmark with num_bits as runtime parameter
    runner = StandardRunner()
    ber_result = runner.run_benchmark(ber_benchmark, num_bits=50000, **config.to_dict())

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
    visualizer.plot_throughput_comparison(throughput_result.metrics, save_path=str(output_dir / "throughput_comparison.png"))
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
    # Create individual BER plots for each modulation scheme
    for i, (mod, result_metrics) in enumerate(zip(modulations, comparison_results)):
        plot_name = f"ber_curve_{mod}.png"
        visualizer.plot_ber_curve(result_metrics, save_path=str(output_dir / plot_name))
        print(f"✓ {mod.upper()} BER curve saved to visualization_results/{plot_name}")

    # Create a combined comparison plot manually using matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for mod, result_metrics in zip(modulations, comparison_results):
        snr_range = result_metrics.get("snr_range", [])
        if "ber_simulated" in result_metrics:
            plt.semilogy(snr_range, result_metrics["ber_simulated"], "o-", label=f"{mod.upper()} (Simulated)", linewidth=2, markersize=6)
        if "ber_theoretical" in result_metrics:
            plt.semilogy(snr_range, result_metrics["ber_theoretical"], "--", label=f"{mod.upper()} (Theoretical)", linewidth=2)

    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate", fontsize=12)
    plt.title("Modulation Scheme Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "modulation_comparison.png"), dpi=100, bbox_inches="tight")
    plt.close()

    print("✓ Modulation comparison saved to visualization_results/modulation_comparison.png")

    # Create summary statistics plot
    print("\n7. Creating performance summary...")
    # Create a summary of benchmark results by saving them to a JSON file first
    import json

    summary_data = {
        "summary": {"total_benchmarks": 2, "successful_benchmarks": 2, "failed_benchmarks": 0, "total_execution_time": ber_result.execution_time + throughput_result.execution_time, "average_execution_time": (ber_result.execution_time + throughput_result.execution_time) / 2},
        "benchmark_results": [
            {"benchmark_name": "BER Simulation (BPSK)", "success": True, "execution_time": ber_result.execution_time, "device": "cpu", **ber_result.metrics},
            {"benchmark_name": "Throughput Test", "success": True, "execution_time": throughput_result.execution_time, "device": "cpu", **throughput_result.metrics},
        ],
    }

    # Save temporary summary file
    summary_file = output_dir / "temp_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Create benchmark summary plot
    visualizer.plot_benchmark_summary(str(summary_file), save_path=str(output_dir / "performance_summary.png"))

    # Clean up temporary file
    summary_file.unlink()

    print("✓ Performance summary saved to visualization_results/performance_summary.png")

    print("\n" + "=" * 50)
    print("✅ Visualization example completed successfully!")
    print("📁 All plots saved to:", output_dir.absolute())
    print("\nGenerated visualizations:")
    print("  • ber_curve.png - BER vs SNR curve")
    print("  • throughput_comparison.png - Throughput performance")
    print("  • modulation_comparison.png - Modulation scheme comparison")
    print("  • performance_summary.png - Overall performance summary")


if __name__ == "__main__":
    main()
