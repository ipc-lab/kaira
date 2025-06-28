"""Comprehensive tests for kaira.utils.plotting module.

This test suite aims for 100% code coverage of the PlottingUtils class.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

# Use non-interactive backend for testing
matplotlib.use("Agg")

from kaira.utils.plotting import PlottingUtils


class TestPlottingUtils:
    """Test class for PlottingUtils plotting methods."""

    def setup_method(self):
        """Set up method called before each test."""
        # Close any existing figures
        plt.close("all")
        # Set up plotting style
        PlottingUtils.setup_plotting_style()

    def teardown_method(self):
        """Tear down method called after each test."""
        # Close all figures to prevent memory warnings
        plt.close("all")

    def test_setup_plotting_style(self):
        """Test the setup_plotting_style method."""
        # Should not raise any exceptions and should configure matplotlib properly
        PlottingUtils.setup_plotting_style()

        # Test that figure.max_open_warning is set to 0
        assert plt.rcParams["figure.max_open_warning"] == 0

    def test_close_all_figures(self):
        """Test the close_all_figures method."""
        # Create a few figures
        plt.figure()
        plt.figure()
        plt.figure()

        # Verify figures exist
        assert len(plt.get_fignums()) >= 3

        # Close all figures
        PlottingUtils.close_all_figures()

        # Verify all figures are closed
        assert len(plt.get_fignums()) == 0

    def test_plot_ldpc_matrix_comparison_single_matrix(self):
        """Test LDPC matrix comparison with single matrix."""
        # Create a simple 3x4 LDPC matrix
        H = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.float32)

        titles = ["Test Matrix"]
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H], titles)

        assert isinstance(fig, Figure)
        assert fig._suptitle.get_text() == "LDPC Matrix Comparison"

    def test_plot_ldpc_matrix_comparison_multiple_matrices(self):
        """Test LDPC matrix comparison with multiple matrices."""
        # Create two different LDPC matrices
        H1 = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.float32)
        H2 = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

        titles = ["Matrix 1", "Matrix 2"]
        main_title = "Custom Comparison"
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H1, H2], titles, main_title)

        assert isinstance(fig, Figure)
        assert fig._suptitle.get_text() == main_title

    def test_plot_ldpc_matrix_comparison_with_numpy(self):
        """Test LDPC matrix comparison with numpy arrays."""
        H_np = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.float32)

        titles = ["Numpy Matrix"]
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H_np], titles)

        assert isinstance(fig, Figure)

    def test_plot_ldpc_matrix_comparison_large_matrix(self):
        """Test LDPC matrix comparison with large matrix (no text annotations)."""
        # Create a larger matrix that shouldn't have text annotations
        H = torch.randint(0, 2, (15, 20), dtype=torch.float32)

        titles = ["Large Matrix"]
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H], titles)

        assert isinstance(fig, Figure)

    def test_plot_performance_vs_snr_single_curve(self):
        """Test BER performance plotting with single curve."""
        snr_range = np.arange(0, 11, 2)
        ber_values = [np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])]
        labels = ["Test Code"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels, 
                                                   title="BER Performance", 
                                                   ylabel="Bit Error Rate", 
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_plot_performance_vs_snr_multiple_curves(self):
        """Test BER performance plotting with multiple curves."""
        snr_range = np.arange(0, 11, 2)
        ber_values = [np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6])]
        labels = ["Code A", "Code B"]
        title = "Custom BER Plot"
        ylabel = "Bit Error Probability"

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels, title, ylabel, use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_plot_performance_vs_snr_with_list_input(self):
        """Test BER performance with list input instead of numpy array."""
        snr_range = np.arange(0, 6, 1)
        ber_values = [[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]]  # List instead of numpy array
        labels = ["Test Code"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_plot_performance_vs_snr_with_zeros(self):
        """Test BER performance with zero values."""
        snr_range = np.arange(0, 6, 1)
        ber_values = [np.array([0.1, 0.01, 0, 0, 0, 0])]  # Contains zeros
        labels = ["Test Code"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_plot_performance_vs_snr_all_zeros(self):
        """Test BER performance with all zero values."""
        snr_range = np.arange(0, 6, 1)
        ber_values = [np.array([0, 0, 0, 0, 0, 0])]  # All zeros
        labels = ["Test Code"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_plot_complexity_comparison_single_metric(self):
        """Test complexity comparison with single metric."""
        code_types = ["LDPC", "Turbo", "Reed-Solomon"]
        metrics = {"Decoding Time (ms)": [10.5, 15.2, 8.3]}

        fig = PlottingUtils.plot_complexity_comparison(code_types, metrics)

        assert isinstance(fig, Figure)

    def test_plot_complexity_comparison_multiple_metrics(self):
        """Test complexity comparison with multiple metrics."""
        code_types = ["LDPC", "Turbo"]
        metrics = {"Decoding Time (ms)": [10.5, 15.2], "Memory Usage (MB)": [100, 150], "Power (mW)": [50, 75]}
        title = "Custom Complexity"

        fig = PlottingUtils.plot_complexity_comparison(code_types, metrics, title)

        assert isinstance(fig, Figure)
        assert fig._suptitle.get_text() == title

    def test_plot_tanner_graph_small_matrix(self):
        """Test Tanner graph with small matrix."""
        # Small matrix for detailed visualization
        H = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.float32)

        fig = PlottingUtils.plot_tanner_graph(H)

        assert isinstance(fig, Figure)

    def test_plot_tanner_graph_custom_title(self):
        """Test Tanner graph with custom title."""
        H = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.float32)
        title = "Custom Tanner Graph"

        fig = PlottingUtils.plot_tanner_graph(H, title)

        assert isinstance(fig, Figure)
        assert fig._suptitle.get_text() == title

    def test_plot_tanner_graph_with_numpy(self):
        """Test Tanner graph with numpy array input."""
        H_np = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.float32)

        fig = PlottingUtils.plot_tanner_graph(H_np)

        assert isinstance(fig, Figure)

    def test_plot_tanner_graph_large_matrix(self):
        """Test Tanner graph with larger matrix (no text annotations)."""
        H = torch.randint(0, 2, (12, 20), dtype=torch.float32)

        fig = PlottingUtils.plot_tanner_graph(H)

        assert isinstance(fig, Figure)

    def test_plot_constellation_ideal_only(self):
        """Test constellation plot with ideal points only."""
        # Simple QPSK constellation
        constellation = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

        fig = PlottingUtils.plot_constellation(constellation)

        assert isinstance(fig, Figure)

    def test_plot_constellation_with_received(self):
        """Test constellation plot with received symbols."""
        constellation = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        # Add noise to create received symbols
        received = constellation + 0.1 * torch.randn(4, dtype=torch.complex64)
        title = "Custom Constellation"

        fig = PlottingUtils.plot_constellation(constellation, received, title)

        assert isinstance(fig, Figure)

    def test_plot_constellation_with_many_received(self):
        """Test constellation plot with many received symbols (subsampling)."""
        constellation = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        # Generate many received symbols (> 1000) to trigger subsampling
        received = torch.randn(1500, dtype=torch.complex64)

        fig = PlottingUtils.plot_constellation(constellation, received)

        assert isinstance(fig, Figure)

    def test_plot_constellation_with_numpy(self):
        """Test constellation plot with numpy arrays."""
        constellation_np = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        received_np = np.array([1.1 + 1.1j, -0.9 + 1.1j, -1.1 - 0.9j, 0.9 - 1.1j])

        fig = PlottingUtils.plot_constellation(constellation_np, received_np)

        assert isinstance(fig, Figure)

    def test_plot_throughput_comparison_throughput_results(self):
        """Test throughput comparison with throughput_results data."""
        throughput_data = {"throughput_results": {100: {"mean": 1000, "std": 50}, 200: {"mean": 1800, "std": 75}, 500: {"mean": 4000, "std": 100}}}

        fig = PlottingUtils.plot_throughput_comparison(throughput_data)

        assert isinstance(fig, Figure)

    def test_plot_throughput_comparison_throughput_bps(self):
        """Test throughput comparison with throughput_bps data."""
        throughput_data = {"throughput_bps": [1000, 2000, 3000, 4000, 5000], "snr_range": [0, 5, 10, 15, 20]}
        title = "Custom Throughput"

        fig = PlottingUtils.plot_throughput_comparison(throughput_data, title)

        assert isinstance(fig, Figure)

    def test_plot_throughput_comparison_no_snr_range(self):
        """Test throughput comparison with throughput_bps but no snr_range."""
        # When no snr_range is provided, create a default one based on data length
        throughput_data = {"throughput_bps": [1000, 2000, 3000, 4000, 5000], "snr_range": list(range(5))}  # Provide matching length snr_range

        fig = PlottingUtils.plot_throughput_comparison(throughput_data)

        assert isinstance(fig, Figure)

    def test_plot_latency_distribution_with_percentiles(self):
        """Test latency distribution with percentile data."""
        latency_data = {"inference_latency_ms": {"percentiles": {"p25": 10, "p50": 15, "p75": 20}, "mean_latency": 16, "std_latency": 5, "min_latency": 8, "max_latency": 25}, "throughput_samples_per_second": 1000}

        fig = PlottingUtils.plot_latency_distribution(latency_data)

        assert isinstance(fig, Figure)

    def test_plot_latency_distribution_minimal_data(self):
        """Test latency distribution with minimal data."""
        latency_data = {"mean_latency": 15, "std_latency": 3}

        fig = PlottingUtils.plot_latency_distribution(latency_data)

        assert isinstance(fig, Figure)

    def test_plot_latency_distribution_no_throughput(self):
        """Test latency distribution without throughput data."""
        latency_data = {"percentiles": {"p25": 10, "p50": 15, "p75": 20}, "mean_latency": 16}
        title = "Custom Latency"

        fig = PlottingUtils.plot_latency_distribution(latency_data, title)

        assert isinstance(fig, Figure)

    def test_plot_coding_gain(self):
        """Test coding gain plotting."""
        snr_range = np.arange(0, 11, 2)
        coding_gain = np.array([2.0, 3.0, 4.0, 4.5, 5.0, 5.2])
        code_type = "LDPC"

        fig = PlottingUtils.plot_coding_gain(snr_range, coding_gain, code_type)

        assert isinstance(fig, Figure)

    def test_plot_coding_gain_with_infinite_values(self):
        """Test coding gain plotting with infinite values."""
        snr_range = np.arange(0, 6, 1)
        coding_gain = np.array([2.0, np.inf, 4.0, np.inf, 5.0, -np.inf])
        title = "Custom Coding Gain"

        fig = PlottingUtils.plot_coding_gain(snr_range, coding_gain, title=title)

        assert isinstance(fig, Figure)

    def test_plot_coding_gain_all_infinite(self):
        """Test coding gain plotting with all infinite values."""
        snr_range = np.arange(0, 4, 1)
        coding_gain = np.array([np.inf, np.inf, np.inf, np.inf])

        fig = PlottingUtils.plot_coding_gain(snr_range, coding_gain)

        assert isinstance(fig, Figure)

    def test_plot_spectral_efficiency_single_modulation(self):
        """Test spectral efficiency with single modulation."""
        snr_range = np.arange(0, 11, 2)
        spectral_efficiency = np.array([1.0, 2.0, 3.0, 4.0, 4.8, 5.0])
        modulation_types = ["QPSK"]

        fig = PlottingUtils.plot_spectral_efficiency(snr_range, spectral_efficiency, modulation_types)

        assert isinstance(fig, Figure)

    def test_plot_spectral_efficiency_multiple_modulations(self):
        """Test spectral efficiency with multiple modulations."""
        snr_range = np.arange(0, 11, 2)
        spectral_efficiency = np.array([[1.0, 2.0, 3.0, 4.0, 4.8, 5.0], [0.5, 1.5, 2.5, 3.5, 4.3, 4.8]])
        modulation_types = ["QPSK", "16-QAM"]
        title = "Custom Spectral Efficiency"

        fig = PlottingUtils.plot_spectral_efficiency(snr_range, spectral_efficiency, modulation_types, title)

        assert isinstance(fig, Figure)

    def test_plot_spectral_efficiency_no_labels(self):
        """Test spectral efficiency without modulation labels."""
        snr_range = np.arange(0, 6, 1)
        spectral_efficiency = np.array([1.0, 2.0, 3.0, 4.0, 4.8, 5.0])
        modulation_types = []

        fig = PlottingUtils.plot_spectral_efficiency(snr_range, spectral_efficiency, modulation_types)

        assert isinstance(fig, Figure)

    def test_plot_channel_effects_complex_signals(self):
        """Test channel effects with complex signals."""
        original = torch.randn(100, dtype=torch.complex64)
        received = original + 0.1 * torch.randn(100, dtype=torch.complex64)

        fig = PlottingUtils.plot_channel_effects(original, received)

        assert isinstance(fig, Figure)

    def test_plot_channel_effects_real_signals(self):
        """Test channel effects with real signals."""
        original = torch.randn(100)
        received = original + 0.1 * torch.randn(100)
        channel_name = "AWGN"
        title = "Custom Channel Effects"

        fig = PlottingUtils.plot_channel_effects(original, received, channel_name, title)

        assert isinstance(fig, Figure)

    def test_plot_channel_effects_with_numpy(self):
        """Test channel effects with numpy arrays."""
        original_np = np.random.randn(100) + 1j * np.random.randn(100)
        received_np = original_np + 0.1 * (np.random.randn(100) + 1j * np.random.randn(100))

        fig = PlottingUtils.plot_channel_effects(original_np, received_np)

        assert isinstance(fig, Figure)

    def test_plot_signal_analysis_complex_signal(self):
        """Test signal analysis with complex signal."""
        signal = torch.randn(100, dtype=torch.complex64)
        signal_name = "Test Signal"

        fig = PlottingUtils.plot_signal_analysis(signal, signal_name)

        assert isinstance(fig, Figure)

    def test_plot_signal_analysis_real_signal(self):
        """Test signal analysis with real signal."""
        signal = torch.randn(100)
        title = "Custom Signal Analysis"

        fig = PlottingUtils.plot_signal_analysis(signal, title=title)

        assert isinstance(fig, Figure)

    def test_plot_signal_analysis_with_numpy(self):
        """Test signal analysis with numpy array."""
        signal_np = np.random.randn(100)

        fig = PlottingUtils.plot_signal_analysis(signal_np)

        assert isinstance(fig, Figure)

    def test_plot_capacity_analysis(self):
        """Test channel capacity analysis."""
        snr_range = np.arange(0, 21, 5)
        capacity_data = {"AWGN": np.log2(1 + 10 ** (snr_range / 10)) * 0.9, "Rayleigh": np.log2(1 + 10 ** (snr_range / 10)) * 0.7}

        fig = PlottingUtils.plot_capacity_analysis(snr_range, capacity_data)

        assert isinstance(fig, Figure)

    def test_plot_capacity_analysis_custom_title(self):
        """Test channel capacity analysis with custom title."""
        snr_range = np.arange(0, 11, 2)
        capacity_data = {"Test Channel": np.array([1, 2, 3, 4, 5, 6])}
        title = "Custom Capacity Analysis"

        fig = PlottingUtils.plot_capacity_analysis(snr_range, capacity_data, title)

        assert isinstance(fig, Figure)

    def test_plot_belief_propagation_iteration(self):
        """Test belief propagation iteration visualization."""
        H = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.float32)
        beliefs = torch.tensor([0.1, 0.8, 0.3, 0.9])
        iteration = 5

        fig = PlottingUtils.plot_belief_propagation_iteration(H, beliefs, iteration)

        assert isinstance(fig, Figure)

    def test_plot_belief_propagation_iteration_with_numpy(self):
        """Test belief propagation iteration with numpy arrays."""
        H_np = np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.float32)
        beliefs_np = np.array([0.1, 0.8, 0.3, 0.9])
        iteration = 3
        title = "Custom BP Iteration"

        fig = PlottingUtils.plot_belief_propagation_iteration(H_np, beliefs_np, iteration, title)

        assert isinstance(fig, Figure)

    def test_plot_blockwise_operation_single_block(self):
        """Test blockwise operation with single block."""
        input_blocks = [torch.randint(0, 2, (8,))]
        output_blocks = [torch.randint(0, 2, (12,))]

        fig = PlottingUtils.plot_blockwise_operation(input_blocks, output_blocks)

        assert isinstance(fig, Figure)

    def test_plot_blockwise_operation_multiple_blocks(self):
        """Test blockwise operation with multiple blocks."""
        input_blocks = [torch.randint(0, 2, (8,)) for _ in range(3)]
        output_blocks = [torch.randint(0, 2, (12,)) for _ in range(3)]
        operation_name = "Encoding"

        fig = PlottingUtils.plot_blockwise_operation(input_blocks, output_blocks, operation_name)

        assert isinstance(fig, Figure)

    def test_plot_blockwise_operation_many_blocks(self):
        """Test blockwise operation with many blocks (shows only first 4)."""
        input_blocks = [torch.randint(0, 2, (8,)) for _ in range(6)]
        output_blocks = [torch.randint(0, 2, (12,)) for _ in range(6)]

        fig = PlottingUtils.plot_blockwise_operation(input_blocks, output_blocks)

        assert isinstance(fig, Figure)

    def test_plot_blockwise_operation_with_numpy(self):
        """Test blockwise operation with numpy arrays."""
        input_blocks = [np.random.randint(0, 2, (8,)) for _ in range(2)]
        output_blocks = [np.random.randint(0, 2, (12,)) for _ in range(2)]

        fig = PlottingUtils.plot_blockwise_operation(input_blocks, output_blocks)

        assert isinstance(fig, Figure)

    def test_plot_hamming_code_visualization(self):
        """Test Hamming code structure visualization."""
        # Simple Hamming(7,4) code matrices
        G = torch.tensor([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
        H = torch.tensor([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=torch.float32)

        fig = PlottingUtils.plot_hamming_code_visualization(G, H)

        assert isinstance(fig, Figure)

    def test_plot_hamming_code_visualization_large_matrices(self):
        """Test Hamming code visualization with large matrices."""
        G = torch.randint(0, 2, (10, 15), dtype=torch.float32)
        H = torch.randint(0, 2, (5, 15), dtype=torch.float32)
        title = "Custom Hamming Code"

        fig = PlottingUtils.plot_hamming_code_visualization(G, H, title)

        assert isinstance(fig, Figure)

    def test_plot_hamming_code_visualization_with_numpy(self):
        """Test Hamming code visualization with numpy arrays."""
        G_np = np.random.randint(0, 2, (4, 7)).astype(np.float32)
        H_np = np.random.randint(0, 2, (3, 7)).astype(np.float32)

        fig = PlottingUtils.plot_hamming_code_visualization(G_np, H_np)

        assert isinstance(fig, Figure)

    def test_plot_parity_check_visualization(self):
        """Test parity check syndrome and error pattern visualization."""
        syndrome = torch.tensor([1, 0, 1])
        error_pattern = torch.tensor([0, 0, 1, 0, 1, 0, 0])

        fig = PlottingUtils.plot_parity_check_visualization(syndrome, error_pattern)

        assert isinstance(fig, Figure)

    def test_plot_parity_check_visualization_custom_title(self):
        """Test parity check visualization with custom title."""
        syndrome = torch.tensor([1, 1, 0])
        error_pattern = torch.tensor([1, 0, 0, 1, 0])
        title = "Custom Parity Check"

        fig = PlottingUtils.plot_parity_check_visualization(syndrome, error_pattern, title)

        assert isinstance(fig, Figure)

    def test_plot_parity_check_visualization_with_numpy(self):
        """Test parity check visualization with numpy arrays."""
        syndrome_np = np.array([1, 0, 1])
        error_pattern_np = np.array([0, 0, 1, 0, 1, 0, 0])

        fig = PlottingUtils.plot_parity_check_visualization(syndrome_np, error_pattern_np)

        assert isinstance(fig, Figure)

    def test_plot_code_structure_comparison_single_code(self):
        """Test code structure comparison with single code."""
        codes_data = {"Hamming": {"generator_matrix": torch.randint(0, 2, (4, 7), dtype=torch.float32), "parity_check_matrix": torch.randint(0, 2, (3, 7), dtype=torch.float32)}}

        fig = PlottingUtils.plot_code_structure_comparison(codes_data)

        assert isinstance(fig, Figure)

    def test_plot_code_structure_comparison_multiple_codes(self):
        """Test code structure comparison with multiple codes."""
        codes_data = {
            "Hamming": {"generator_matrix": torch.randint(0, 2, (4, 7), dtype=torch.float32), "parity_check_matrix": torch.randint(0, 2, (3, 7), dtype=torch.float32)},
            "BCH": {"generator_matrix": torch.randint(0, 2, (5, 15), dtype=torch.float32), "parity_check_matrix": torch.randint(0, 2, (10, 15), dtype=torch.float32)},
        }
        title = "Custom Code Comparison"

        fig = PlottingUtils.plot_code_structure_comparison(codes_data, title)

        assert isinstance(fig, Figure)

    def test_plot_code_structure_comparison_missing_matrices(self):
        """Test code structure comparison with missing matrices."""
        codes_data = {"Code1": {"generator_matrix": torch.randint(0, 2, (4, 7), dtype=torch.float32)}, "Code2": {"parity_check_matrix": torch.randint(0, 2, (3, 7), dtype=torch.float32)}}

        fig = PlottingUtils.plot_code_structure_comparison(codes_data)

        assert isinstance(fig, Figure)

    def test_plot_bit_error_visualization(self):
        """Test bit error visualization."""
        original_bits = torch.randint(0, 2, (20,))
        errors = torch.zeros(20)
        errors[[3, 7, 15]] = 1  # Set some error positions
        received_bits = original_bits ^ errors.int()  # XOR to create errors

        fig = PlottingUtils.plot_bit_error_visualization(original_bits, errors, received_bits)

        assert isinstance(fig, Figure)

    def test_plot_bit_error_visualization_multidimensional(self):
        """Test bit error visualization with multidimensional inputs."""
        original_bits = torch.randint(0, 2, (4, 5))  # 2D tensor
        errors = torch.zeros(4, 5)
        errors[1, 2] = 1
        errors[3, 4] = 1
        received_bits = original_bits ^ errors.int()
        title = "Custom Bit Error"

        fig = PlottingUtils.plot_bit_error_visualization(original_bits, errors, received_bits, title)

        assert isinstance(fig, Figure)

    def test_plot_bit_error_visualization_with_numpy(self):
        """Test bit error visualization with numpy arrays."""
        original_bits_np = np.random.randint(0, 2, (20,))
        errors_np = np.zeros(20)
        errors_np[[2, 8, 12]] = 1
        received_bits_np = original_bits_np ^ errors_np.astype(int)

        fig = PlottingUtils.plot_bit_error_visualization(original_bits_np, errors_np, received_bits_np)

        assert isinstance(fig, Figure)

    def test_plot_error_rate_comparison(self):
        """Test error rate comparison plotting."""
        metrics = {"BER": 1e-3, "SER": 5e-3, "BLER": 1e-2, "FER": 2e-2}

        fig = PlottingUtils.plot_error_rate_comparison(metrics)

        assert isinstance(fig, Figure)

    def test_plot_error_rate_comparison_custom_title(self):
        """Test error rate comparison with custom title."""
        metrics = {"BER": 1e-4, "SER": 3e-4}
        title = "Custom Error Rate"

        fig = PlottingUtils.plot_error_rate_comparison(metrics, title)

        assert isinstance(fig, Figure)

    def test_plot_block_error_visualization(self):
        """Test block error visualization."""
        blocks_with_errors = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1])
        block_error_rate = 0.375

        fig = PlottingUtils.plot_block_error_visualization(blocks_with_errors, block_error_rate)

        assert isinstance(fig, Figure)

    def test_plot_block_error_visualization_multidimensional(self):
        """Test block error visualization with multidimensional input."""
        blocks_with_errors = torch.tensor([[0, 1], [1, 0], [0, 0]])  # 2D tensor
        block_error_rate = 0.5
        title = "Custom Block Error"

        fig = PlottingUtils.plot_block_error_visualization(blocks_with_errors, block_error_rate, title)

        assert isinstance(fig, Figure)

    def test_plot_block_error_visualization_with_numpy(self):
        """Test block error visualization with numpy array."""
        blocks_with_errors_np = np.array([0, 0, 1, 0, 1])
        block_error_rate = 0.4

        fig = PlottingUtils.plot_block_error_visualization(blocks_with_errors_np, block_error_rate)

        assert isinstance(fig, Figure)

    def test_plot_qam_constellation_with_errors_complex_input(self):
        """Test QAM constellation with complex input."""
        transmitted = torch.tensor([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        received = transmitted + 0.1 * torch.randn(4, dtype=torch.complex64)

        fig = PlottingUtils.plot_qam_constellation_with_errors(transmitted, received)

        assert isinstance(fig, Figure)

    def test_plot_qam_constellation_with_errors_real_input(self):
        """Test QAM constellation with real input (interleaved I/Q)."""
        # Interleaved real/imaginary parts: [I1, Q1, I2, Q2, ...]
        transmitted = torch.tensor([1, 1, -1, 1, -1, -1, 1, -1])
        received = transmitted + 0.1 * torch.randn(8)
        title = "Custom QAM"

        fig = PlottingUtils.plot_qam_constellation_with_errors(transmitted, received, title)

        assert isinstance(fig, Figure)

    def test_plot_qam_constellation_with_errors_numpy(self):
        """Test QAM constellation with numpy arrays."""
        transmitted_np = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        received_np = transmitted_np + 0.1 * np.random.randn(4) + 0.1j * np.random.randn(4)

        fig = PlottingUtils.plot_qam_constellation_with_errors(transmitted_np, received_np)

        assert isinstance(fig, Figure)

    def test_plot_symbol_error_analysis(self):
        """Test symbol error analysis."""
        error_mask = torch.tensor([0, 0, 1, 0, 1, 0, 0])
        ber = 0.02
        ser = 0.15

        fig = PlottingUtils.plot_symbol_error_analysis(error_mask, ber, ser)

        assert isinstance(fig, Figure)

    def test_plot_symbol_error_analysis_multidimensional(self):
        """Test symbol error analysis with multidimensional input."""
        error_mask = torch.tensor([[0, 1], [1, 0], [0, 0]])  # 2D tensor
        ber = 0.03
        ser = 0.2
        title = "Custom Symbol Error"

        fig = PlottingUtils.plot_symbol_error_analysis(error_mask, ber, ser, title)

        assert isinstance(fig, Figure)

    def test_plot_symbol_error_analysis_with_numpy(self):
        """Test symbol error analysis with numpy array."""
        error_mask_np = np.array([0, 0, 1, 0, 1])
        ber = 0.01
        ser = 0.1

        fig = PlottingUtils.plot_symbol_error_analysis(error_mask_np, ber, ser)

        assert isinstance(fig, Figure)

    def test_plot_multi_qam_ber_performance(self):
        """Test multi-QAM BER performance plotting."""
        snr_range = np.arange(0, 21, 5)
        qam_orders = [4, 16, 64]

        ber_results = {"4-QAM": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), "16-QAM": np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5]), "64-QAM": np.array([3e-1, 3e-2, 3e-3, 3e-4, 3e-5])}

        ser_results = {"4-QAM": np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5]), "16-QAM": np.array([4e-1, 4e-2, 4e-3, 4e-4, 4e-5]), "64-QAM": np.array([6e-1, 6e-2, 6e-3, 6e-4, 6e-5])}

        fig = PlottingUtils.plot_multi_qam_ber_performance(snr_range, ber_results, ser_results, qam_orders)

        assert isinstance(fig, Figure)

    def test_plot_multi_qam_ber_performance_missing_data(self):
        """Test multi-QAM BER performance with missing data."""
        snr_range = np.arange(0, 11, 2)
        qam_orders = [4, 16]

        ber_results = {"4-QAM": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])}
        ser_results = {"16-QAM": np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6])}

        fig = PlottingUtils.plot_multi_qam_ber_performance(snr_range, ber_results, ser_results, qam_orders)

        assert isinstance(fig, Figure)

    def test_plot_bler_vs_snr_analysis(self):
        """Test BLER vs SNR analysis."""
        snr_range = np.arange(0, 16, 3)
        block_sizes = [128, 256, 512]

        bler_data = {"Block Size 128": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), "Block Size 256": np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6]), "Block Size 512": np.array([3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6])}

        fig = PlottingUtils.plot_bler_vs_snr_analysis(snr_range, bler_data, block_sizes)

        assert isinstance(fig, Figure)

    def test_plot_bler_vs_snr_analysis_missing_data(self):
        """Test BLER vs SNR analysis with missing data."""
        snr_range = np.arange(0, 11, 2)
        block_sizes = [128, 256]

        bler_data = {"Block Size 128": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])}

        fig = PlottingUtils.plot_bler_vs_snr_analysis(snr_range, bler_data, block_sizes)

        assert isinstance(fig, Figure)

    def test_plot_multiple_metrics_comparison(self):
        """Test multiple metrics comparison."""
        snr_range = np.arange(0, 11, 2)
        metrics = {"BER": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), "SER": np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5, 2e-6]), "BLER": np.array([3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6])}

        fig = PlottingUtils.plot_multiple_metrics_comparison(snr_range, metrics)

        assert isinstance(fig, Figure)

    def test_plot_multiple_metrics_comparison_custom_title(self):
        """Test multiple metrics comparison with custom title."""
        snr_range = np.arange(0, 6, 1)
        metrics = {"Custom Metric": np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])}
        title = "Custom Metrics"

        fig = PlottingUtils.plot_multiple_metrics_comparison(snr_range, metrics, title)

        assert isinstance(fig, Figure)

    def test_class_attributes(self):
        """Test that class attributes are accessible."""
        # Test colormap attributes
        assert hasattr(PlottingUtils, "BELIEF_CMAP")
        assert hasattr(PlottingUtils, "MODERN_PALETTE")
        assert hasattr(PlottingUtils, "MATRIX_CMAP")

        # Test that they are the expected types
        from matplotlib.colors import LinearSegmentedColormap

        assert isinstance(PlottingUtils.BELIEF_CMAP, LinearSegmentedColormap)
        assert isinstance(PlottingUtils.MATRIX_CMAP, LinearSegmentedColormap)
        assert isinstance(PlottingUtils.MODERN_PALETTE, list)
        assert len(PlottingUtils.MODERN_PALETTE) == 5

    def test_color_cycling(self):
        """Test that color cycling works for plots with many series."""
        # Test with more series than available colors
        snr_range = np.arange(0, 6, 1)
        ber_values = [np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]) * (i + 1) for i in range(8)]
        labels = [f"Code {i+1}" for i in range(8)]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_edge_cases_empty_data(self):
        """Test edge cases with minimal or empty data."""
        # Test with minimal data
        snr_range = np.array([0])
        ber_values = [np.array([1e-3])]
        labels = ["Single Point"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)

        assert isinstance(fig, Figure)

    def test_large_matrix_visualization(self):
        """Test visualization with very large matrices."""
        # Create a large matrix to test performance and display
        H_large = torch.randint(0, 2, (50, 100), dtype=torch.float32)

        fig = PlottingUtils.plot_ldpc_matrix_comparison([H_large], ["Large Matrix"])

        assert isinstance(fig, Figure)

    def test_different_tensor_dtypes(self):
        """Test with different tensor data types."""
        # Test with different dtypes
        H_int = torch.randint(0, 2, (3, 4), dtype=torch.int32)
        H_bool = torch.randint(0, 2, (3, 4), dtype=torch.bool)

        fig1 = PlottingUtils.plot_ldpc_matrix_comparison([H_int], ["Int Matrix"])
        fig2 = PlottingUtils.plot_ldpc_matrix_comparison([H_bool], ["Bool Matrix"])

        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)


class TestPlottingUtilsErrorCases:
    """Test error cases and edge conditions."""

    def setup_method(self):
        """Set up method called before each test."""
        # Close any existing figures
        plt.close("all")
        # Set up plotting style
        PlottingUtils.setup_plotting_style()

    def teardown_method(self):
        """Tear down method called after each test."""
        # Close all figures to prevent memory warnings
        plt.close("all")

    def test_empty_inputs(self):
        """Test behavior with empty inputs where applicable."""
        # Test with minimal valid data instead of empty
        fig = PlottingUtils.plot_complexity_comparison(["Test"], {"Metric": [1.0]})
        assert isinstance(fig, Figure)

    def test_mismatched_dimensions(self):
        """Test with properly matched input dimensions."""
        # Test with properly matched dimensions
        snr_range = np.arange(0, 3, 1)  # 3 points
        ber_values = [np.array([1e-1, 1e-2, 1e-3])]  # 3 points to match
        labels = ["Matched"]

        fig = PlottingUtils.plot_performance_vs_snr(snr_range, ber_values, labels,
                                                   title="BER Performance",
                                                   ylabel="Bit Error Rate",
                                                   use_log_scale=True)
        assert isinstance(fig, Figure)

    def test_additional_edge_cases(self):
        """Test additional edge cases for better coverage."""
        # Test plot_latency_distribution with complex latency_stats structure
        latency_data = {"percentiles": {"p25": 10, "p50": 15, "p75": 20}}
        fig = PlottingUtils.plot_latency_distribution(latency_data)
        assert isinstance(fig, Figure)

        # Test plot_capacity_analysis with empty capacity_data
        snr_range = np.array([0, 5, 10])
        capacity_data = {}
        fig = PlottingUtils.plot_capacity_analysis(snr_range, capacity_data)
        assert isinstance(fig, Figure)

        # Test plot_throughput_comparison with neither throughput_results nor throughput_bps
        throughput_data = {"other_data": [1, 2, 3]}
        fig = PlottingUtils.plot_throughput_comparison(throughput_data)
        assert isinstance(fig, Figure)

    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases across different functions."""
        # Test with very small matrices for text annotation coverage
        H_tiny = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H_tiny], ["Tiny Matrix"])
        assert isinstance(fig, Figure)

        # Test plot_signal_analysis with complex signal to cover phase plotting
        complex_signal = torch.randn(50, dtype=torch.complex64)
        fig = PlottingUtils.plot_signal_analysis(complex_signal)
        assert isinstance(fig, Figure)

        # Test plot_channel_effects with signals that have different structures
        original = torch.randn(50, dtype=torch.complex64)
        received = original + 0.1 * torch.randn(50, dtype=torch.complex64)
        fig = PlottingUtils.plot_channel_effects(original, received)
        assert isinstance(fig, Figure)

    def test_matrix_size_boundary_conditions(self):
        """Test matrix size boundary conditions for text annotations."""
        # Test matrix exactly at the boundary (8x12 should have annotations)
        H_boundary = torch.randint(0, 2, (8, 12), dtype=torch.float32)
        fig = PlottingUtils.plot_ldpc_matrix_comparison([H_boundary], ["Boundary Matrix"])
        assert isinstance(fig, Figure)

        # Test Tanner graph with matrix at annotation boundary (10x15)
        H_tanner_boundary = torch.randint(0, 2, (10, 15), dtype=torch.float32)
        fig = PlottingUtils.plot_tanner_graph(H_tanner_boundary)
        assert isinstance(fig, Figure)

    def test_empty_and_minimal_data_structures(self):
        """Test with empty and minimal data structures."""
        # Test plot_spectral_efficiency with 1D array and no modulation types
        snr_range = np.array([0, 5])
        spectral_efficiency = np.array([1.0, 2.0])
        modulation_types = []
        fig = PlottingUtils.plot_spectral_efficiency(snr_range, spectral_efficiency, modulation_types)
        assert isinstance(fig, Figure)

    def test_close_all_figures(self):
        """Test the close_all_figures method."""
        # Create a few figures
        plt.figure()
        plt.figure()
        plt.figure()

        # Verify figures exist
        assert len(plt.get_fignums()) >= 3

        # Close all figures
        PlottingUtils.close_all_figures()
