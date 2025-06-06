"""Tests for advanced communication system benchmarks."""

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from kaira.benchmarks import (
    BenchmarkRegistry,
    BenchmarkVisualizer,
    StandardRunner,
    create_benchmark,
)


class TestQAMBenchmark:
    """Test QAM modulation benchmark."""

    def test_qam_benchmark_registration(self):
        """Test that QAM benchmark is properly registered."""
        assert "qam_ber" in BenchmarkRegistry.list_available()

    def test_qam_benchmark_16qam(self):
        """Test 16-QAM benchmark execution."""
        runner = StandardRunner()

        # Create benchmark instance
        benchmark = create_benchmark("qam_ber", constellation_size=16)

        # Run with minimal parameters for speed
        results = runner.run_benchmark(benchmark, snr_range=torch.arange(0, 10, 5).tolist(), num_symbols=1000)

        assert results.metrics["success"]
        assert "ber_results" in results.metrics
        assert "constellation_size" in results.metrics
        assert results.metrics["constellation_size"] == 16
        assert results.metrics["bits_per_symbol"] == 4
        assert len(results.metrics["ber_results"]) == len(results.metrics["snr_range"])

        # BER should decrease with increasing SNR
        ber_values = results.metrics["ber_results"]
        assert ber_values[0] > ber_values[-1], "BER should decrease with increasing SNR"

    def test_qam_benchmark_4qam(self):
        """Test 4-QAM (QPSK) benchmark execution."""
        runner = StandardRunner()

        benchmark = create_benchmark("qam_ber", constellation_size=4)

        results = runner.run_benchmark(benchmark, snr_range=[0, 5, 10], num_symbols=1000)

        assert results.metrics["success"]
        assert results.metrics["constellation_size"] == 4
        assert results.metrics["bits_per_symbol"] == 2

    def test_qam_benchmark_invalid_constellation(self):
        """Test QAM benchmark with invalid constellation size."""
        # Test non-square constellation size should raise ValueError
        with pytest.raises(ValueError, match="Constellation size must be a perfect square"):
            benchmark = create_benchmark("qam_ber", constellation_size=12)  # Not a perfect square
            runner = StandardRunner()
            runner.run_benchmark(benchmark, snr_range=[0, 5], num_symbols=100)


class TestOFDMBenchmark:
    """Test OFDM performance benchmark."""

    def test_ofdm_benchmark_registration(self):
        """Test that OFDM benchmark is properly registered."""
        assert "ofdm_performance" in BenchmarkRegistry.list_available()

    def test_ofdm_benchmark_execution(self):
        """Test OFDM benchmark execution."""
        runner = StandardRunner()

        benchmark = create_benchmark("ofdm_performance", num_subcarriers=64, cp_length=16)

        results = runner.run_benchmark(benchmark, snr_range=torch.arange(0, 15, 5).tolist(), num_symbols=100, modulation="qpsk")

        assert results.metrics["success"]
        assert "ber_results" in results.metrics
        assert "throughput_bps" in results.metrics
        assert results.metrics["num_subcarriers"] == 64
        assert results.metrics["cp_length"] == 16
        assert results.metrics["modulation"] == "qpsk"
        assert results.metrics["spectral_efficiency"] == 2  # QPSK = 2 bits/symbol

        # Check that throughput values are reasonable
        throughput_values = results.metrics["throughput_bps"]
        assert all(t > 0 for t in throughput_values), "All throughput values should be positive"

    def test_ofdm_benchmark_different_sizes(self):
        """Test OFDM with different subcarrier configurations."""
        runner = StandardRunner()

        # Test different OFDM configurations
        configs = [
            {"num_subcarriers": 32, "cp_length": 8},
            {"num_subcarriers": 128, "cp_length": 32},
        ]

        for config in configs:
            benchmark = create_benchmark("ofdm_performance", **config)

            results = runner.run_benchmark(benchmark, snr_range=[0, 10], num_symbols=50)

            assert results.metrics["success"]
            assert results.metrics["num_subcarriers"] == config["num_subcarriers"]
            assert results.metrics["cp_length"] == config["cp_length"]


class TestChannelCodingBenchmark:
    """Test channel coding benchmark."""

    def test_coding_benchmark_registration(self):
        """Test that channel coding benchmark is properly registered."""
        assert "channel_coding" in BenchmarkRegistry.list_available()

    def test_repetition_coding(self):
        """Test repetition coding benchmark."""
        runner = StandardRunner()

        benchmark = create_benchmark("channel_coding", code_type="repetition", code_rate=1 / 3)  # 3-repetition code

        results = runner.run_benchmark(benchmark, snr_range=torch.arange(-5, 5, 5).tolist(), num_bits=1000)

        assert results.metrics["success"]
        assert "ber_uncoded" in results.metrics
        assert "ber_coded" in results.metrics
        assert "coding_gain_db" in results.metrics
        assert results.metrics["code_type"] == "repetition"
        assert results.metrics["code_rate"] == 1 / 3

        # Coded BER should be better than uncoded BER
        ber_uncoded = results.metrics["ber_uncoded"]
        ber_coded = results.metrics["ber_coded"]

        # At least for some SNR values, coded should be better
        improvements = [unc > cod for unc, cod in zip(ber_uncoded, ber_coded)]
        assert any(improvements), "Coding should improve BER for some SNR values"

    def test_coding_gain_calculation(self):
        """Test coding gain calculation."""
        runner = StandardRunner()

        benchmark = create_benchmark("channel_coding", code_type="repetition", code_rate=1 / 3)  # 3-repetition code for better gain

        # Test at low SNR where coding gain is more apparent
        results = runner.run_benchmark(benchmark, snr_range=[-2, 0, 2], num_bits=5000)  # Lower SNR range

        assert results.metrics["success"]
        assert "average_coding_gain" in results.metrics

        # At low SNR, repetition codes should provide gain
        # Check individual gains rather than average to be more flexible
        gains = results.metrics["coding_gain_db"]
        finite_gains = [g for g in gains if torch.isfinite(torch.tensor(g)).item()]
        assert len(finite_gains) > 0, "Should have at least some finite coding gains"

        # At least one SNR point should show positive gain
        assert any(g > 0 for g in finite_gains), "Should have positive coding gain for at least one SNR point"


class TestBenchmarkVisualization:
    """Test benchmark visualization capabilities."""

    @pytest.fixture
    def visualizer(self):
        """Create a benchmark visualizer."""
        return BenchmarkVisualizer(figsize=(8, 6), dpi=80)

    @pytest.fixture
    def sample_ber_results(self):
        """Create sample BER results for testing."""
        snr_range = torch.arange(0, 10, 2)
        ber_simulated = 0.5 * torch.exp(-snr_range / 2)  # Synthetic BER curve
        ber_theoretical = 0.5 * torch.exp(-snr_range / 2.2)  # Slightly different

        return {"benchmark_name": "Test BER Benchmark", "snr_range": snr_range.tolist(), "ber_simulated": ber_simulated.tolist(), "ber_theoretical": ber_theoretical.tolist(), "rmse": 0.001}

    @pytest.fixture
    def sample_throughput_results(self):
        """Create sample throughput results for testing."""
        return {"benchmark_name": "Test Throughput Benchmark", "throughput_results": {100: {"mean": 1000, "std": 50, "min": 950, "max": 1100}, 1000: {"mean": 5000, "std": 200, "min": 4800, "max": 5300}, 10000: {"mean": 15000, "std": 500, "min": 14200, "max": 15800}}}

    def test_plot_ber_curve(self, visualizer, sample_ber_results):
        """Test BER curve plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "ber_test.png"

            fig = visualizer.plot_ber_curve(sample_ber_results, str(save_path))

            assert isinstance(fig, plt.Figure)
            assert save_path.exists()

            plt.close(fig)

    def test_plot_throughput_comparison(self, visualizer, sample_throughput_results):
        """Test throughput comparison plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "throughput_test.png"

            fig = visualizer.plot_throughput_comparison(sample_throughput_results, str(save_path))

            assert isinstance(fig, plt.Figure)
            assert save_path.exists()

            plt.close(fig)

    def test_plot_constellation(self, visualizer):
        """Test constellation diagram plotting."""
        # Create sample constellation
        constellation = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / torch.sqrt(torch.tensor(2.0))
        received = constellation + 0.1 * (torch.randn(4) + 1j * torch.randn(4))

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "constellation_test.png"

            fig = visualizer.plot_constellation(constellation, received, str(save_path))

            assert isinstance(fig, plt.Figure)
            assert save_path.exists()

            plt.close(fig)

    def test_benchmark_report_creation(self, visualizer):
        """Test comprehensive benchmark report creation."""
        # Create sample benchmark results file
        sample_data = {
            "summary": {"total_benchmarks": 3, "successful_benchmarks": 3, "failed_benchmarks": 0, "total_execution_time": 45.2, "average_execution_time": 15.1},
            "benchmark_results": [
                {"benchmark_name": "Test BER", "success": True, "execution_time": 12.5, "device": "cpu", "snr_range": [0, 5, 10], "ber_simulated": [0.1, 0.01, 0.001], "ber_theoretical": [0.11, 0.011, 0.0011], "rmse": 0.0005},
                {"benchmark_name": "Test Throughput", "success": True, "execution_time": 15.3, "device": "cpu", "throughput_results": {"100": {"mean": 1000, "std": 50}, "1000": {"mean": 5000, "std": 200}}},
                {"benchmark_name": "Test Coding", "success": True, "execution_time": 17.4, "device": "cuda", "snr_range": [0, 5], "coding_gain_db": [2.5, 3.1], "average_coding_gain": 2.8, "code_type": "repetition"},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sample results
            results_file = Path(temp_dir) / "results.json"
            with open(results_file, "w") as f:
                json.dump(sample_data, f)

            # Create report
            output_dir = Path(temp_dir) / "plots"
            visualizer.create_benchmark_report(str(results_file), str(output_dir))

            # Check that plots were created
            assert output_dir.exists()
            assert (output_dir / "summary.png").exists()

            # Should have individual plots for each benchmark
            plot_files = list(output_dir.glob("*.png"))
            assert len(plot_files) >= 4  # At least summary + 3 benchmark plots


class TestBenchmarkIntegration:
    """Test integration of new benchmarks with existing system."""

    def test_all_new_benchmarks_registered(self):
        """Test that all new benchmarks are properly registered."""
        expected_benchmarks = ["qam_ber", "ofdm_performance", "channel_coding"]
        registered_benchmarks = BenchmarkRegistry.list_available()

        for benchmark in expected_benchmarks:
            assert benchmark in registered_benchmarks, f"Benchmark {benchmark} not registered"

    def test_benchmark_suite_with_new_benchmarks(self):
        """Test running a suite containing new benchmarks."""
        from kaira.benchmarks.base import BenchmarkSuite

        suite = BenchmarkSuite("Advanced Communication Tests")

        # Add new benchmarks
        qam_benchmark = create_benchmark("qam_ber", constellation_size=4)
        ofdm_benchmark = create_benchmark("ofdm_performance", num_subcarriers=32)
        coding_benchmark = create_benchmark("channel_coding", code_type="repetition", code_rate=0.5)

        suite.add_benchmark(qam_benchmark)
        suite.add_benchmark(ofdm_benchmark)
        suite.add_benchmark(coding_benchmark)

        # Run suite manually since we need to pass parameters
        runner = StandardRunner()
        results = []

        # Run QAM benchmark
        result1 = runner.run_benchmark(qam_benchmark, snr_range=[0, 10], num_symbols=500)
        results.append(result1)

        # Run OFDM benchmark
        result2 = runner.run_benchmark(ofdm_benchmark, snr_range=[0, 10], num_symbols=50)
        results.append(result2)

        # Run coding benchmark
        result3 = runner.run_benchmark(coding_benchmark, snr_range=[0, 5], num_bits=1000)
        results.append(result3)

        assert len(results) == 3
        assert all(result.metrics["success"] for result in results)

    def test_parallel_execution_new_benchmarks(self):
        """Test parallel execution of new benchmarks."""
        from kaira.benchmarks.runners import ParallelRunner

        # Create benchmark instances
        qam_benchmark = create_benchmark("qam_ber", constellation_size=4)
        ofdm_benchmark = create_benchmark("ofdm_performance", num_subcarriers=32)

        benchmarks = [qam_benchmark, ofdm_benchmark]

        runner = ParallelRunner(max_workers=2)

        # Use the correct interface for ParallelRunner
        results = runner.run_benchmarks(benchmarks, snr_range=[0, 5], num_symbols=200)

        assert len(results) == 2
        assert all(result.metrics["success"] for result in results)

    def test_benchmark_comparison(self):
        """Test comparing different configurations of new benchmarks."""
        from kaira.benchmarks.runners import ComparisonRunner

        # Create different QAM benchmark configurations
        qam4_benchmark = create_benchmark("qam_ber", constellation_size=4)
        qam16_benchmark = create_benchmark("qam_ber", constellation_size=16)

        benchmarks = [qam4_benchmark, qam16_benchmark]

        runner = ComparisonRunner()

        # Use the correct interface for ComparisonRunner
        results = runner.run_comparison(benchmarks, comparison_name="QAM_Comparison", snr_range=[0, 10], num_symbols=500)

        assert len(results) == 2
        assert all(result.metrics["success"] for result in results.values())

        # Get comparison summary
        summary = runner.get_comparison_summary("QAM_Comparison")
        assert "comparison_name" in summary
        assert summary["comparison_name"] == "QAM_Comparison"
