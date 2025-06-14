"""Plotting utilities for LDPC and FEC examples.

This module provides reusable plotting functions to keep example files focused on the core
algorithm demonstrations while maintaining consistent visualization across examples.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


class PlottingUtils:
    """A comprehensive plotting utility class with static methods for visualization.

    This class provides a centralized collection of plotting functions for various
    communication system analysis tasks including LDPC codes, modulation schemes,
    error rate analysis, signal processing, and more.

    All methods are static to allow easy access without instantiation:

    Example:
        fig = PlottingUtils.plot_ber_performance(snr_range, ber_values, labels)
    """

    # Color schemes and palettes as static attributes
    BELIEF_CMAP = LinearSegmentedColormap.from_list("belief", ["#d32f2f", "#ffeb3b", "#4caf50"], N=256)
    MODERN_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    MATRIX_CMAP = LinearSegmentedColormap.from_list("matrix", ["white", "#2c3e50"])

    @staticmethod
    def setup_plotting_style():
        """Set up consistent plotting style for all examples."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_context("notebook", font_scale=1.2)

    @staticmethod
    def plot_ldpc_matrix_comparison(H_matrices: List[torch.Tensor], titles: List[str], main_title: str = "LDPC Matrix Comparison") -> plt.Figure:
        """Plot comparison of LDPC code matrix structures.

        Parameters
        ----------
        H_matrices : List[torch.Tensor]
            List of parity check matrices to compare
        titles : List[str]
            Titles for each matrix
        main_title : str
            Overall plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        n_matrices = len(H_matrices)
        fig, axes = plt.subplots(1, n_matrices, figsize=(6 * n_matrices, 5), constrained_layout=True)

        if n_matrices == 1:
            axes = [axes]

        fig.suptitle(main_title, fontsize=16, fontweight="bold")

        for i, (H, title) in enumerate(zip(H_matrices, titles)):
            ax = axes[i]
            H_np = H.numpy() if isinstance(H, torch.Tensor) else H
            m, n = H_np.shape

            # Plot matrix heatmap
            im = ax.imshow(H_np, cmap=PlottingUtils.MATRIX_CMAP, interpolation="nearest", aspect="auto")

            # Add text annotations for small matrices
            if m <= 8 and n <= 12:
                for row in range(m):
                    for col in range(n):
                        color = "white" if H_np[row, col] == 1 else "black"
                        ax.text(col, row, int(H_np[row, col]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Variable Nodes", fontsize=12)
            ax.set_ylabel("Check Nodes", fontsize=12)

            # Add sparsity information
            sparsity = np.sum(H_np) / (m * n)
            ax.text(0.02, 0.98, f"Sparsity: {sparsity:.3f}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), verticalalignment="top")

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

        return fig

    @staticmethod
    def plot_ber_performance(snr_range: np.ndarray, ber_values: List[np.ndarray], labels: List[str], title: str = "BER vs SNR Performance", ylabel: str = "Bit Error Rate") -> plt.Figure:
        """Plot BER vs SNR performance curves.

        Parameters
        ----------
        snr_range : np.ndarray
            SNR values in dB
        ber_values : List[np.ndarray]
            BER values for each configuration
        labels : List[str]
            Labels for each curve
        title : str
            Plot title
        ylabel : str
            Y-axis label

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        for i, (ber, label) in enumerate(zip(ber_values, labels)):
            # Convert to numpy array if it's a list
            ber_array = np.array(ber) if isinstance(ber, list) else ber
            color = PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)]
            ax.semilogy(snr_range, ber_array, "o-", color=color, linewidth=2, markersize=6, label=label, alpha=0.8)

        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Set reasonable y-axis limits
        all_ber_arrays = [np.array(ber) if isinstance(ber, list) else ber for ber in ber_values]
        non_zero_bers = [ber_arr[ber_arr > 0] for ber_arr in all_ber_arrays if len(ber_arr[ber_arr > 0]) > 0]
        if non_zero_bers:
            min_ber = min([np.min(ber_subset) for ber_subset in non_zero_bers])
            ax.set_ylim(min_ber / 10, 1)
        else:
            ax.set_ylim(1e-6, 1)

        return fig

    @staticmethod
    def plot_complexity_comparison(code_types: List[str], metrics: Dict[str, List[float]], title: str = "Complexity Comparison") -> plt.Figure:
        """Plot complexity comparison charts.

        Parameters
        ----------
        code_types : List[str]
            Names of different code types
        metrics : Dict[str, List[float]]
            Dictionary mapping metric names to values for each code type
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), constrained_layout=True)

        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight="bold")

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[i]
            ax.bar(code_types, values, color=PlottingUtils.MODERN_PALETTE[: len(code_types)], alpha=0.8, edgecolor="black", linewidth=1)

            # Add value labels on bars
            for j, value in enumerate(values):
                ax.text(j, value + value * 0.01, f"{value:.2f}", ha="center", va="bottom", fontweight="bold")

            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Value", fontsize=11)
            ax.tick_params(axis="x", rotation=45)

        return fig

    @staticmethod
    def plot_tanner_graph(H: torch.Tensor, title: str = "LDPC Tanner Graph") -> plt.Figure:
        """Create enhanced Tanner graph visualization.

        Parameters
        ----------
        H : torch.Tensor
            Parity check matrix
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        H_np = H.numpy() if isinstance(H, torch.Tensor) else H
        m, n = H_np.shape  # m check nodes, n variable nodes

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Left plot: Bipartite graph representation
        ax1.set_title("Bipartite Graph", fontsize=14, fontweight="bold")

        # Position variable nodes in a circle (top)
        var_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        var_positions = [(2 * np.cos(angle), 2 * np.sin(angle) + 1) for angle in var_angles]

        # Position check nodes in a circle (bottom)
        check_angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
        check_positions = [(1.5 * np.cos(angle), 1.5 * np.sin(angle) - 1) for angle in check_angles]

        # Draw connections
        connection_counts = np.sum(H_np, axis=0)  # variable node degrees
        max_degree = np.max(connection_counts)

        for i in range(m):
            for j in range(n):
                if H_np[i, j] == 1:
                    thickness = 1 + 2 * (connection_counts[j] / max_degree)
                    alpha = 0.6 + 0.4 * (connection_counts[j] / max_degree)
                    line = FancyArrowPatch(check_positions[i], var_positions[j], arrowstyle="-", color="gray", linewidth=thickness, alpha=alpha, connectionstyle="arc3,rad=0.1")
                    ax1.add_patch(line)

        # Draw variable nodes
        for j, pos in enumerate(var_positions):
            size = 0.15 + 0.15 * (connection_counts[j] / max_degree)
            circle = Circle(pos, size, facecolor=PlottingUtils.MODERN_PALETTE[0], edgecolor="black", linewidth=2, zorder=10)
            ax1.add_patch(circle)
            ax1.text(pos[0], pos[1], f"v{j}", ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=11)

        # Draw check nodes
        check_degrees = np.sum(H_np, axis=1)
        max_check_degree = np.max(check_degrees)

        for i, pos in enumerate(check_positions):
            size = 0.15 + 0.15 * (check_degrees[i] / max_check_degree)
            square = Rectangle((pos[0] - size, pos[1] - size), 2 * size, 2 * size, facecolor=PlottingUtils.MODERN_PALETTE[3], edgecolor="black", linewidth=2, zorder=10)
            ax1.add_patch(square)
            ax1.text(pos[0], pos[1], f"c{i}", ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=11)

        ax1.set_xlim(-3.5, 3.5)
        ax1.set_ylim(-3.5, 3.5)
        ax1.set_aspect("equal")
        ax1.axis("off")

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PlottingUtils.MODERN_PALETTE[0], markersize=10, label="Variable Nodes"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PlottingUtils.MODERN_PALETTE[3], markersize=10, label="Check Nodes"),
            plt.Line2D([0], [0], color="gray", linewidth=2, label="Connections"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right")

        # Right plot: Matrix heatmap
        ax2.set_title("Parity Check Matrix H", fontsize=14, fontweight="bold")
        im = ax2.imshow(H_np, cmap=PlottingUtils.MATRIX_CMAP, interpolation="nearest", aspect="auto")

        # Add text annotations for reasonable-sized matrices
        if m <= 10 and n <= 15:
            for i in range(m):
                for j in range(n):
                    color = "black" if H_np[i, j] == 0 else "white"
                    ax2.text(j, i, int(H_np[i, j]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        ax2.set_xticks(range(n))
        ax2.set_yticks(range(m))
        ax2.set_xlabel("Variable Nodes", fontsize=12)
        ax2.set_ylabel("Check Nodes", fontsize=12)

        # Add colorbar and sparsity info
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1"])

        sparsity = np.sum(H_np) / (m * n)
        ax2.text(0.02, 0.98, f"Sparsity: {sparsity:.3f}\nDensity: {1-sparsity:.3f}", transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), verticalalignment="top")

        return fig

    @staticmethod
    def plot_constellation(constellation: torch.Tensor, received_symbols: Optional[torch.Tensor] = None, title: str = "Constellation Diagram") -> plt.Figure:
        """Plot constellation diagram with optional received symbols.

        Parameters
        ----------
        constellation : torch.Tensor
            Ideal constellation points
        received_symbols : Optional[torch.Tensor]
            Optional received symbols to overlay
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

        # Plot ideal constellation
        constellation_np = constellation.numpy() if isinstance(constellation, torch.Tensor) else constellation
        ax.scatter(constellation_np.real, constellation_np.imag, c="red", s=100, marker="x", linewidths=3, label="Ideal", zorder=10)

        # Plot received symbols if provided
        if received_symbols is not None:
            received_np = received_symbols.numpy() if isinstance(received_symbols, torch.Tensor) else received_symbols

            # Subsample if too many points
            if len(received_np) > 1000:
                indices = np.random.choice(len(received_np), 1000, replace=False)
                received_np = received_np[indices]

            ax.scatter(received_np.real, received_np.imag, c="blue", s=20, alpha=0.6, label="Received", zorder=5)

        ax.set_xlabel("In-Phase", fontsize=12)
        ax.set_ylabel("Quadrature", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis("equal")

        return fig

    @staticmethod
    def plot_throughput_comparison(throughput_data: Dict[str, Any], title: str = "Throughput Comparison") -> plt.Figure:
        """Plot throughput comparison across different configurations.

        Parameters
        ----------
        throughput_data : Dict[str, Any]
            Dictionary containing throughput data
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        if "throughput_results" in throughput_data:
            # Bar plot for different payload sizes
            payload_sizes = []
            mean_throughputs = []
            std_throughputs = []

            for size, stats in throughput_data["throughput_results"].items():
                payload_sizes.append(size)
                mean_throughputs.append(stats["mean"])
                std_throughputs.append(stats["std"])

            x_pos = np.arange(len(payload_sizes))
            bars = ax.bar(x_pos, mean_throughputs, yerr=std_throughputs, capsize=5, alpha=0.7, edgecolor="black")

            ax.set_xlabel("Payload Size (bits)", fontsize=12)
            ax.set_ylabel("Throughput (bits/s)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(size) for size in payload_sizes])
            ax.grid(True, alpha=0.3)

            # Color bars based on throughput
            import matplotlib.colors as mcolors

            colors = mcolors.LinearSegmentedColormap.from_list("viridis", ["purple", "blue", "green", "yellow"])(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

        elif "throughput_bps" in throughput_data:
            # Line plot for throughput vs SNR
            snr_range = throughput_data.get("snr_range", [])
            ax.plot(snr_range, throughput_data["throughput_bps"], "o-", linewidth=2, markersize=6, color=PlottingUtils.MODERN_PALETTE[0])
            ax.set_xlabel("SNR (dB)", fontsize=12)
            ax.set_ylabel("Throughput (bits/s)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)

        return fig

    @staticmethod
    def plot_latency_distribution(latency_data: Dict[str, Any], title: str = "Latency Distribution") -> plt.Figure:
        """Plot latency distribution and statistics.

        Parameters
        ----------
        latency_data : Dict[str, Any]
            Dictionary containing latency statistics
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

        # Extract latency statistics
        latency_stats = latency_data.get("inference_latency_ms", latency_data)

        # Box plot
        if "percentiles" in latency_stats:
            percentiles = latency_stats["percentiles"]
            box_data = [percentiles["p25"], percentiles["p50"], percentiles["p75"]]

            bp = ax1.boxplot([box_data], patch_artist=True, labels=["Latency"])
            bp["boxes"][0].set_facecolor(PlottingUtils.MODERN_PALETTE[0])
            bp["boxes"][0].set_alpha(0.7)

        ax1.set_ylabel("Latency (ms)", fontsize=12)
        ax1.set_title("Latency Distribution", fontsize=14, fontweight="bold")
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
        if "throughput_samples_per_second" in latency_data:
            throughput = latency_data["throughput_samples_per_second"]
            ax2.bar(["Throughput"], [throughput], color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
            ax2.set_ylabel("Samples/second", fontsize=12)
            ax2.set_title("Processing Throughput", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis("off")

        return fig

    @staticmethod
    def plot_coding_gain(snr_range: np.ndarray, coding_gain: np.ndarray, code_type: str = "Unknown", title: str = "Coding Gain") -> plt.Figure:
        """Plot coding gain vs SNR.

        Parameters
        ----------
        snr_range : np.ndarray
            SNR values in dB
        coding_gain : np.ndarray
            Coding gain values in dB
        code_type : str
            Type of error correction code
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        # Filter out infinite values
        coding_gain_array = np.array(coding_gain)
        finite_mask = np.isfinite(coding_gain_array)
        snr_finite = snr_range[finite_mask]
        gain_finite = coding_gain_array[finite_mask]

        ax.plot(snr_finite, gain_finite, "o-", linewidth=2, markersize=6, color=PlottingUtils.MODERN_PALETTE[0])
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Coding Gain (dB)", fontsize=12)
        ax.set_title(f"{title} - {code_type} Code", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add average coding gain if finite values exist
        if len(gain_finite) > 0:
            avg_gain = np.mean(gain_finite)
            ax.axhline(y=avg_gain, color="red", linestyle="--", alpha=0.7, label=f"Average: {avg_gain:.2f} dB")
            ax.legend()

        return fig

    @staticmethod
    def plot_spectral_efficiency(snr_range: np.ndarray, spectral_efficiency: np.ndarray, modulation_types: List[str], title: str = "Spectral Efficiency") -> plt.Figure:
        """Plot spectral efficiency vs SNR for different modulation schemes.

        Parameters
        ----------
        snr_range : np.ndarray
            SNR values in dB
        spectral_efficiency : np.ndarray
            Spectral efficiency values (bits/s/Hz)
        modulation_types : List[str]
            Names of modulation schemes
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        if spectral_efficiency.ndim == 1:
            # Single modulation scheme
            ax.plot(snr_range, spectral_efficiency, "o-", linewidth=2, markersize=6, color=PlottingUtils.MODERN_PALETTE[0], label=modulation_types[0] if modulation_types else "")
        else:
            # Multiple modulation schemes
            for i, mod_type in enumerate(modulation_types):
                color = PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)]
                ax.plot(snr_range, spectral_efficiency[i], "o-", linewidth=2, markersize=6, color=color, label=mod_type)

        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Spectral Efficiency (bits/s/Hz)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if modulation_types:
            ax.legend()

        return fig

    @staticmethod
    def plot_channel_effects(original_signal: torch.Tensor, received_signal: torch.Tensor, channel_name: str = "Channel", title: str = "Channel Effects") -> plt.Figure:
        """Plot the effects of a channel on transmitted signals.

        Parameters
        ----------
        original_signal : torch.Tensor
            Original transmitted signal
        received_signal : torch.Tensor
            Signal after passing through channel
        channel_name : str
            Name of the channel
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        original_np = original_signal.numpy() if isinstance(original_signal, torch.Tensor) else original_signal
        received_np = received_signal.numpy() if isinstance(received_signal, torch.Tensor) else received_signal

        # Ensure we're working with real data for time domain plots
        if np.iscomplexobj(original_np):
            original_real = original_np.real
            original_imag = original_np.imag
        else:
            original_real = original_np
            original_imag = None

        if np.iscomplexobj(received_np):
            received_real = received_np.real
            received_imag = received_np.imag
        else:
            received_real = received_np
            received_imag = None

        # Time domain - Real part
        axes[0, 0].plot(original_real[:100], "b-", label="Original", alpha=0.7)
        axes[0, 0].plot(received_real[:100], "r-", label="Received", alpha=0.7)
        axes[0, 0].set_title("Time Domain - Real Part")
        axes[0, 0].set_xlabel("Sample")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Time domain - Imaginary part (if complex)
        if original_imag is not None and received_imag is not None:
            axes[0, 1].plot(original_imag[:100], "b-", label="Original", alpha=0.7)
            axes[0, 1].plot(received_imag[:100], "r-", label="Received", alpha=0.7)
            axes[0, 1].set_title("Time Domain - Imaginary Part")
        else:
            axes[0, 1].plot(original_real[:100], "b-", label="Original", alpha=0.7)
            axes[0, 1].plot(received_real[:100], "r-", label="Received", alpha=0.7)
            axes[0, 1].set_title("Signal Comparison")
        axes[0, 1].set_xlabel("Sample")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Scatter plot for complex signals
        if np.iscomplexobj(original_np) and np.iscomplexobj(received_np):
            axes[1, 0].scatter(original_np.real, original_np.imag, c="blue", s=20, alpha=0.6, label="Original")
            axes[1, 0].scatter(received_np.real, received_np.imag, c="red", s=20, alpha=0.6, label="Received")
            axes[1, 0].set_title("I/Q Scatter Plot")
            axes[1, 0].set_xlabel("In-Phase")
            axes[1, 0].set_ylabel("Quadrature")
        else:
            axes[1, 0].scatter(range(len(original_real)), original_real, c="blue", s=20, alpha=0.6, label="Original")
            axes[1, 0].scatter(range(len(received_real)), received_real, c="red", s=20, alpha=0.6, label="Received")
            axes[1, 0].set_title("Signal Scatter Plot")
            axes[1, 0].set_xlabel("Sample Index")
            axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Error analysis
        error = received_np - original_np
        if np.iscomplexobj(error):
            error_magnitude = np.abs(error)
        else:
            error_magnitude = np.abs(error)

        axes[1, 1].plot(error_magnitude[:100], "g-", linewidth=2)
        axes[1, 1].set_title("Error Magnitude")
        axes[1, 1].set_xlabel("Sample")
        axes[1, 1].set_ylabel("Error")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{title} - {channel_name}", fontsize=16, fontweight="bold")
        return fig

    @staticmethod
    def plot_signal_analysis(signal: torch.Tensor, signal_name: str = "Signal", title: str = "Signal Analysis") -> plt.Figure:
        """Plot comprehensive signal analysis including time and frequency domain.

        Parameters
        ----------
        signal : torch.Tensor
            Input signal to analyze
        signal_name : str
            Name of the signal
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        signal_np = signal.numpy() if isinstance(signal, torch.Tensor) else signal

        # Handle complex signals
        if np.iscomplexobj(signal_np):
            signal_real = signal_np.real
            signal_imag = signal_np.imag
            signal_magnitude = np.abs(signal_np)
            signal_phase = np.angle(signal_np)
        else:
            signal_real = signal_np
            signal_imag = None
            signal_magnitude = np.abs(signal_np)
            signal_phase = None

        # Time domain - Real part
        axes[0, 0].plot(signal_real, "b-", linewidth=1.5)
        axes[0, 0].set_title(f"{signal_name} - Real Part")
        axes[0, 0].set_xlabel("Sample")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Time domain - Imaginary part or magnitude
        if signal_imag is not None:
            axes[0, 1].plot(signal_imag, "r-", linewidth=1.5)
            axes[0, 1].set_title(f"{signal_name} - Imaginary Part")
        else:
            axes[0, 1].plot(signal_magnitude, "g-", linewidth=1.5)
            axes[0, 1].set_title(f"{signal_name} - Magnitude")
        axes[0, 1].set_xlabel("Sample")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].grid(True, alpha=0.3)

        # Frequency domain - Magnitude
        fft_signal = np.fft.fft(signal_np)
        freq_magnitude = np.abs(fft_signal)
        freq_bins = np.fft.fftfreq(len(signal_np))

        axes[1, 0].plot(freq_bins, freq_magnitude, "purple", linewidth=1.5)
        axes[1, 0].set_title(f"{signal_name} - Frequency Domain")
        axes[1, 0].set_xlabel("Frequency (normalized)")
        axes[1, 0].set_ylabel("Magnitude")
        axes[1, 0].grid(True, alpha=0.3)

        # Power spectral density or phase
        if signal_phase is not None:
            axes[1, 1].plot(signal_phase, "orange", linewidth=1.5)
            axes[1, 1].set_title(f"{signal_name} - Phase")
            axes[1, 1].set_xlabel("Sample")
            axes[1, 1].set_ylabel("Phase (radians)")
        else:
            psd = freq_magnitude**2
            axes[1, 1].plot(freq_bins, psd, "brown", linewidth=1.5)
            axes[1, 1].set_title(f"{signal_name} - Power Spectral Density")
            axes[1, 1].set_xlabel("Frequency (normalized)")
            axes[1, 1].set_ylabel("Power")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight="bold")
        return fig

    @staticmethod
    def plot_capacity_analysis(snr_range: np.ndarray, capacity_data: Dict[str, np.ndarray], title: str = "Channel Capacity Analysis") -> plt.Figure:
        """Plot channel capacity analysis for different channel types.

        Parameters
        ----------
        snr_range : np.ndarray
            SNR values in dB
        capacity_data : Dict[str, np.ndarray]
            Dictionary mapping channel names to capacity values
        title : str
            Plot title

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

        # Plot Shannon capacity limit
        shannon_capacity = np.log2(1 + 10 ** (snr_range / 10))
        ax.plot(snr_range, shannon_capacity, "k--", linewidth=2, label="Shannon Limit")

        # Plot capacity for different channels
        for i, (channel_name, capacity) in enumerate(capacity_data.items()):
            color = PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)]
            ax.plot(snr_range, capacity, "o-", linewidth=2, markersize=6, color=color, label=channel_name)

        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Capacity (bits/channel use)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig
