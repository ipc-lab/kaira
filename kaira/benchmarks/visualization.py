"""Visualization utilities for benchmark results."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Visualizer for benchmark results."""

    def __init__(self, figsize: tuple = (10, 6), dpi: int = 100):
        """Initialize visualizer.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_ber_curve(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot BER vs SNR curve.

        Args:
            results: Benchmark results containing SNR and BER data
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        snr_range = results.get("snr_range", [])

        # Plot simulated BER
        if "ber_simulated" in results:
            ax.semilogy(snr_range, results["ber_simulated"], "o-", label="Simulated", linewidth=2, markersize=6)
        elif "ber_results" in results:
            ax.semilogy(snr_range, results["ber_results"], "o-", label="Simulated", linewidth=2, markersize=6)

        # Plot theoretical BER if available
        if "ber_theoretical" in results:
            ax.semilogy(snr_range, results["ber_theoretical"], "--", label="Theoretical", linewidth=2)

        # Plot coded and uncoded BER if available
        if "ber_uncoded" in results and "ber_coded" in results:
            ax.semilogy(snr_range, results["ber_uncoded"], "o-", label="Uncoded", linewidth=2, markersize=6)
            ax.semilogy(snr_range, results["ber_coded"], "s-", label="Coded", linewidth=2, markersize=6)

        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Bit Error Rate", fontsize=12)

        # Determine title from benchmark name or context
        benchmark_name = results.get("benchmark_name", "")
        if not benchmark_name:
            # Try to infer from other fields
            if "modulation" in results:
                benchmark_name = f"BER Simulation ({results['modulation'].upper()})"
            elif "constellation_size" in results:
                benchmark_name = f"{results['constellation_size']}-QAM BER"
            else:
                benchmark_name = "BER Performance"

        ax.set_title(f"BER Performance - {benchmark_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add text with key metrics
        if "rmse" in results:
            ax.text(0.02, 0.98, f'RMSE: {results["rmse"]:.2e}', transform=ax.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_throughput_comparison(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot throughput comparison.

        Args:
            results: Benchmark results containing throughput data
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if "throughput_results" in results:
            # Bar plot for different payload sizes
            payload_sizes = []
            mean_throughputs = []
            std_throughputs = []

            for size, stats in results["throughput_results"].items():
                payload_sizes.append(size)
                mean_throughputs.append(stats["mean"])
                std_throughputs.append(stats["std"])

            x_pos = torch.arange(len(payload_sizes))
            bars = ax.bar(x_pos, mean_throughputs, yerr=std_throughputs, capsize=5, alpha=0.7, edgecolor="black")

            ax.set_xlabel("Payload Size (bits)", fontsize=12)
            ax.set_ylabel("Throughput (bits/s)", fontsize=12)
            ax.set_title("Throughput vs Payload Size", fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(size) for size in payload_sizes])
            ax.grid(True, alpha=0.3)

            # Color bars based on throughput
            import matplotlib.colors as mcolors
            import numpy as np

            colors = mcolors.LinearSegmentedColormap.from_list("viridis", ["purple", "blue", "green", "yellow"])(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        elif "throughput_bps" in results:
            # Line plot for OFDM throughput vs SNR
            snr_range = results.get("snr_range", [])
            ax.plot(snr_range, results["throughput_bps"], "o-", linewidth=2, markersize=6)
            ax.set_xlabel("SNR (dB)", fontsize=12)
            ax.set_ylabel("Throughput (bits/s)", fontsize=12)
            ax.set_title("Throughput vs SNR", fontsize=14)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_latency_distribution(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot latency distribution.

        Args:
            results: Benchmark results containing latency data
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)

        # Extract latency statistics
        latency_stats = results.get("inference_latency_ms", results)

        # Box plot
        if "percentiles" in latency_stats:
            percentiles = latency_stats["percentiles"]
            box_data = [percentiles["p25"], percentiles["p50"], percentiles["p75"]]

            bp = ax1.boxplot([box_data], patch_artist=True, labels=["Latency"])
            bp["boxes"][0].set_facecolor("lightblue")
            bp["boxes"][0].set_alpha(0.7)

        ax1.set_ylabel("Latency (ms)", fontsize=12)
        ax1.set_title("Latency Distribution", fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = []
        if "mean_latency" in latency_stats:
            stats_text.append(f"Mean: {latency_stats['mean_latency']:.2f} ms")
        if "std_latency" in latency_stats:
            stats_text.append(f"Std: {latency_stats['std_latency']:.2f} ms")
        if "min_latency" in latency_stats:
            stats_text.append(f"Min: {latency_stats['min_latency']:.2f} ms")
        if "max_latency" in latency_stats:
            stats_text.append(f"Max: {latency_stats['max_latency']:.2f} ms")

        if stats_text:
            ax1.text(0.02, 0.98, "\n".join(stats_text), transform=ax1.transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Throughput bar (if available)
        if "throughput_samples_per_second" in results:
            throughput = results["throughput_samples_per_second"]
            ax2.bar(["Throughput"], [throughput], color="orange", alpha=0.7)
            ax2.set_ylabel("Samples/second", fontsize=12)
            ax2.set_title("Processing Throughput", fontsize=14)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_constellation(self, constellation: torch.Tensor, received_symbols: Optional[torch.Tensor] = None, save_path: Optional[str] = None) -> plt.Figure:
        """Plot constellation diagram.

        Args:
            constellation: Ideal constellation points
            received_symbols: Optional received symbols to overlay
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot ideal constellation
        ax.scatter(constellation.real, constellation.imag, c="red", s=100, marker="x", linewidths=3, label="Ideal")

        # Plot received symbols if provided
        if received_symbols is not None:
            # Subsample if too many points
            if len(received_symbols) > 1000:
                indices = torch.randperm(len(received_symbols))[:1000]
                received_symbols = received_symbols[indices]

            ax.scatter(received_symbols.real, received_symbols.imag, c="blue", s=20, alpha=0.6, label="Received")

        ax.set_xlabel("In-Phase", fontsize=12)
        ax.set_ylabel("Quadrature", fontsize=12)
        ax.set_title("Constellation Diagram", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_coding_gain(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot coding gain vs SNR.

        Args:
            results: Benchmark results containing coding gain data
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        snr_range = results.get("snr_range", [])
        coding_gain = results.get("coding_gain_db", [])

        # Filter out infinite values
        coding_gain_tensor = torch.tensor(coding_gain) if not isinstance(coding_gain, torch.Tensor) else coding_gain
        finite_mask = torch.isfinite(coding_gain_tensor)
        snr_range_tensor = torch.tensor(snr_range) if not isinstance(snr_range, torch.Tensor) else snr_range
        snr_finite = snr_range_tensor[finite_mask]
        gain_finite = coding_gain_tensor[finite_mask]

        ax.plot(snr_finite, gain_finite, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Coding Gain (dB)", fontsize=12)
        ax.set_title(f'Coding Gain - {results.get("code_type", "Unknown")} Code', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add average coding gain
        if "average_coding_gain" in results:
            avg_gain = results["average_coding_gain"]
            ax.axhline(y=avg_gain, color="red", linestyle="--", alpha=0.7, label=f"Average: {avg_gain:.2f} dB")
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_benchmark_summary(self, results_file: str, save_path: Optional[str] = None) -> plt.Figure:
        """Plot summary of multiple benchmark results.

        Args:
            results_file: Path to JSON file containing benchmark results
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        with open(results_file) as f:
            data = json.load(f)

        benchmarks = data.get("benchmark_results", [])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)

        # Success rate
        success_count = sum(1 for b in benchmarks if b.get("success", False))
        total_count = len(benchmarks)

        ax1.pie([success_count, total_count - success_count], labels=["Success", "Failed"], autopct="%1.1f%%", colors=["lightgreen", "lightcoral"])
        ax1.set_title("Benchmark Success Rate", fontsize=14)

        # Execution times
        execution_times = [b.get("execution_time", 0) for b in benchmarks]

        bars = ax2.bar(range(len(execution_times)), execution_times, alpha=0.7)
        ax2.set_xlabel("Benchmark Index", fontsize=12)
        ax2.set_ylabel("Execution Time (s)", fontsize=12)
        ax2.set_title("Execution Times", fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Color bars by execution time
        if execution_times:
            import matplotlib.colors as mcolors
            import numpy as np

            colors = mcolors.LinearSegmentedColormap.from_list("plasma", ["purple", "red", "orange", "yellow"])(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        # Device usage
        devices = [b.get("device", "unknown") for b in benchmarks]
        device_counts: dict[str, int] = {}
        for device in devices:
            device_counts[device] = device_counts.get(device, 0) + 1

        if device_counts:
            ax3.pie(device_counts.values(), labels=device_counts.keys(), autopct="%1.1f%%")
            ax3.set_title("Device Usage", fontsize=14)
        else:
            ax3.axis("off")

        # Summary statistics
        summary_stats = data.get("summary", {})
        stats_text = []

        if "total_benchmarks" in summary_stats:
            stats_text.append(f"Total Benchmarks: {summary_stats['total_benchmarks']}")
        if "successful_benchmarks" in summary_stats:
            stats_text.append(f"Successful: {summary_stats['successful_benchmarks']}")
        if "total_execution_time" in summary_stats:
            stats_text.append(f"Total Time: {summary_stats['total_execution_time']:.2f}s")
        if "average_execution_time" in summary_stats:
            stats_text.append(f"Avg Time: {summary_stats['average_execution_time']:.2f}s")

        ax4.text(0.1, 0.9, "\n".join(stats_text), transform=ax4.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
        ax4.set_title("Summary Statistics", fontsize=14)
        ax4.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_benchmark_report(self, results_file: str, output_dir: str = "benchmark_plots"):
        """Create a comprehensive visual report from benchmark results.

        Args:
            results_file: Path to JSON file containing benchmark results
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)

        with open(results_file) as f:
            data = json.load(f)

        benchmarks = data.get("benchmark_results", [])

        # Create summary plot
        summary_fig = self.plot_benchmark_summary(results_file, save_path=f"{output_dir}/summary.png")
        plt.close(summary_fig)

        # Create individual plots for each benchmark
        for i, benchmark in enumerate(benchmarks):
            if not benchmark.get("success", False):
                continue

            benchmark_name = benchmark.get("benchmark_name", f"benchmark_{i}")
            safe_name = benchmark_name.replace(" ", "_").replace("(", "").replace(")", "")

            try:
                # BER plots
                if any(key in benchmark for key in ["ber_simulated", "ber_results", "ber_uncoded"]):
                    ber_fig = self.plot_ber_curve(benchmark, save_path=f"{output_dir}/{safe_name}_ber.png")
                    plt.close(ber_fig)

                # Throughput plots
                if "throughput_results" in benchmark or "throughput_bps" in benchmark:
                    throughput_fig = self.plot_throughput_comparison(benchmark, save_path=f"{output_dir}/{safe_name}_throughput.png")
                    plt.close(throughput_fig)

                # Latency plots
                if "inference_latency_ms" in benchmark or "mean_latency" in benchmark:
                    latency_fig = self.plot_latency_distribution(benchmark, save_path=f"{output_dir}/{safe_name}_latency.png")
                    plt.close(latency_fig)

                # Coding gain plots
                if "coding_gain_db" in benchmark:
                    coding_fig = self.plot_coding_gain(benchmark, save_path=f"{output_dir}/{safe_name}_coding_gain.png")
                    plt.close(coding_fig)

            except Exception as e:
                print(f"Warning: Could not create plot for {benchmark_name}: {e}")

        print(f"Benchmark report saved to {output_dir}/")
