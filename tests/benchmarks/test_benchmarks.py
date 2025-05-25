"""Tests for the Kaira benchmarking system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from kaira.benchmarks import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    StandardMetrics,
    StandardRunner,
    get_benchmark,
    register_benchmark,
)


class TestBaseBenchmark:
    """Test the base benchmark functionality."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation and serialization."""
        result = BenchmarkResult(benchmark_id="test-123", name="Test Benchmark", description="A test benchmark", metrics={"accuracy": 0.95, "loss": 0.1}, execution_time=1.5, timestamp="2025-05-25 10:00:00")

        assert result.benchmark_id == "test-123"
        assert result.name == "Test Benchmark"
        assert result.metrics["accuracy"] == 0.95
        assert result.execution_time == 1.5

        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["name"] == "Test Benchmark"

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "Test Benchmark" in json_str

    def test_benchmark_result_save_load(self):
        """Test saving and loading benchmark results."""
        result = BenchmarkResult(benchmark_id="test-123", name="Test Benchmark", description="A test benchmark", metrics={"accuracy": 0.95}, execution_time=1.5, timestamp="2025-05-25 10:00:00")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            result.save(f.name)

            # Load and verify
            with open(f.name) as load_f:
                loaded_data = json.load(load_f)
                assert loaded_data["name"] == "Test Benchmark"
                assert loaded_data["metrics"]["accuracy"] == 0.95


class SimpleBenchmark(BaseBenchmark):
    """Simple benchmark for testing."""

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.setup_called = True

    def run(self, **kwargs):
        return {"success": True, "test_metric": kwargs.get("test_value", 42), "random_value": np.random.random()}

    def teardown(self):
        super().teardown()
        self.teardown_called = True


class TestBenchmarkExecution:
    """Test benchmark execution."""

    def test_simple_benchmark_execution(self):
        """Test executing a simple benchmark."""
        benchmark = SimpleBenchmark("Test Benchmark", "A simple test")

        result = benchmark.execute(test_value=100)

        assert result.name == "Test Benchmark"
        assert result.metrics["success"] is True
        assert result.metrics["test_metric"] == 100
        assert result.execution_time > 0
        assert benchmark._setup_called
        assert benchmark._teardown_called

    def test_benchmark_suite(self):
        """Test benchmark suite functionality."""
        suite = BenchmarkSuite("Test Suite", "A test suite")

        # Add benchmarks
        benchmark1 = SimpleBenchmark("Benchmark 1")
        benchmark2 = SimpleBenchmark("Benchmark 2")
        suite.add_benchmark(benchmark1)
        suite.add_benchmark(benchmark2)

        assert len(suite.benchmarks) == 2

        # Run suite
        results = suite.run_all(test_value=50)

        assert len(results) == 2
        assert all(r.metrics["test_metric"] == 50 for r in results)

        # Get summary
        summary = suite.get_summary()
        assert summary["total_benchmarks"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0


class TestStandardMetrics:
    """Test standard metrics calculations."""

    def test_bit_error_rate(self):
        """Test BER calculation."""
        transmitted = np.array([0, 1, 0, 1, 0, 1])
        received = np.array([0, 1, 1, 1, 0, 0])  # 2 errors out of 6 bits

        ber = StandardMetrics.bit_error_rate(transmitted, received)
        assert abs(ber - 2 / 6) < 1e-10

        # Test with torch tensors
        transmitted_torch = torch.tensor(transmitted)
        received_torch = torch.tensor(received)
        ber_torch = StandardMetrics.bit_error_rate(transmitted_torch, received_torch)
        assert abs(ber_torch - 2 / 6) < 1e-10

    def test_block_error_rate(self):
        """Test BLER calculation."""
        transmitted = np.array([0, 1, 0, 1, 0, 1, 1, 0])  # 8 bits
        received = np.array([0, 1, 1, 1, 0, 1, 1, 1])  # Error in first and last block

        bler = StandardMetrics.block_error_rate(transmitted, received, block_size=4)
        assert abs(bler - 2 / 2) < 1e-10  # Both blocks have errors

    def test_signal_to_noise_ratio(self):
        """Test SNR calculation."""
        signal = np.array([1.0, 2.0, 3.0])
        noise = np.array([0.1, 0.1, 0.1])

        snr = StandardMetrics.signal_to_noise_ratio(signal, noise)
        assert snr > 0  # Should be positive for this case

    def test_throughput(self):
        """Test throughput calculation."""
        throughput = StandardMetrics.throughput(1000, 2.0)  # 1000 bits in 2 seconds
        assert throughput == 500.0  # 500 bits/second

        # Test edge case
        throughput_zero_time = StandardMetrics.throughput(1000, 0)
        assert throughput_zero_time == 0.0

    def test_latency_statistics(self):
        """Test latency statistics."""
        latencies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = StandardMetrics.latency_statistics(latencies)

        assert stats["mean_latency"] == 3.0
        assert stats["median_latency"] == 3.0
        assert stats["min_latency"] == 1.0
        assert stats["max_latency"] == 5.0
        assert "p95_latency" in stats
        assert "p99_latency" in stats

    def test_channel_capacity(self):
        """Test channel capacity calculation."""
        capacity = StandardMetrics.channel_capacity(10.0, 1.0)  # 10 dB SNR, 1 Hz bandwidth
        assert capacity > 0

        # Higher SNR should give higher capacity
        capacity_high = StandardMetrics.channel_capacity(20.0, 1.0)
        assert capacity_high > capacity

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        data = np.random.normal(0, 1, 100)

        lower, upper = StandardMetrics.confidence_interval(data, confidence=0.95)

        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(name="test_config", description="Test configuration", num_trials=5, verbose=False)

        assert config.name == "test_config"
        assert config.num_trials == 5
        assert config.verbose is False
        assert config.seed == 42  # Default value

    def test_config_serialization(self):
        """Test config serialization."""
        config = BenchmarkConfig(name="test", num_trials=3)

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test"
        assert config_dict["num_trials"] == 3

        # Test to_json
        json_str = config.to_json()
        assert isinstance(json_str, str)
        assert "test" in json_str

        # Test from_dict
        new_config = BenchmarkConfig.from_dict(config_dict)
        assert new_config.name == "test"
        assert new_config.num_trials == 3

        # Test from_json
        json_config = BenchmarkConfig.from_json(json_str)
        assert json_config.name == "test"

    def test_config_update(self):
        """Test updating configuration."""
        config = BenchmarkConfig()

        config.update(name="updated", custom_param=123)

        assert config.name == "updated"
        assert config.get("custom_param") == 123
        assert config.get("nonexistent", "default") == "default"


class TestStandardRunner:
    """Test standard benchmark runner."""

    def test_run_single_benchmark(self):
        """Test running a single benchmark."""
        runner = StandardRunner(verbose=False)
        benchmark = SimpleBenchmark("Test", "Description")

        result = runner.run_benchmark(benchmark, test_value=42)

        assert result.name == "Test"
        assert result.metrics["test_metric"] == 42
        assert len(runner.results) == 1

    def test_run_benchmark_suite(self):
        """Test running a benchmark suite."""
        runner = StandardRunner(verbose=False)
        suite = BenchmarkSuite("Test Suite")

        suite.add_benchmark(SimpleBenchmark("Benchmark 1"))
        suite.add_benchmark(SimpleBenchmark("Benchmark 2"))

        results = runner.run_suite(suite, test_value=100)

        assert len(results) == 2
        assert all(r.metrics["test_metric"] == 100 for r in results)


@register_benchmark("test_registered")
class RegisteredBenchmark(BaseBenchmark):
    """Test benchmark for registry functionality."""

    def setup(self, **kwargs):
        super().setup(**kwargs)

    def run(self, **kwargs):
        return {"success": True, "registry_test": True}


class TestBenchmarkRegistry:
    """Test benchmark registry functionality."""

    def test_benchmark_registration(self):
        """Test benchmark registration and retrieval."""
        # Test getting registered benchmark
        benchmark_class = get_benchmark("test_registered")
        assert benchmark_class is not None

        # Test creating instance
        benchmark = benchmark_class("Test Instance")
        assert benchmark.name == "Test Instance"

        result = benchmark.execute()
        assert result.metrics["registry_test"] is True


class TestStandardBenchmarks:
    """Test the standard benchmark implementations."""

    def test_channel_capacity_benchmark(self):
        """Test channel capacity benchmark."""
        benchmark_class = get_benchmark("channel_capacity")
        if benchmark_class is not None:
            benchmark = benchmark_class(channel_type="awgn")
            result = benchmark.execute(bandwidth=1.0)

            assert result.metrics["success"] is True
            assert "capacities" in result.metrics
            assert "max_capacity" in result.metrics
            assert len(result.metrics["capacities"]) > 0

    def test_ber_simulation_benchmark(self):
        """Test BER simulation benchmark."""
        benchmark_class = get_benchmark("ber_simulation")
        if benchmark_class is not None:
            benchmark = benchmark_class(modulation="bpsk")
            result = benchmark.execute(num_bits=1000)

            assert result.metrics["success"] is True
            assert "ber_simulated" in result.metrics
            assert "ber_theoretical" in result.metrics
            assert len(result.metrics["ber_simulated"]) > 0

    def test_throughput_benchmark(self):
        """Test throughput benchmark."""
        benchmark_class = get_benchmark("throughput_test")
        if benchmark_class is not None:
            benchmark = benchmark_class()
            result = benchmark.execute(payload_sizes=[100, 1000], num_trials=2)

            assert result.metrics["success"] is True
            assert "throughput_results" in result.metrics
            assert "peak_throughput" in result.metrics

    def test_latency_benchmark(self):
        """Test latency benchmark."""
        benchmark_class = get_benchmark("latency_test")
        if benchmark_class is not None:
            benchmark = benchmark_class()
            result = benchmark.execute(num_measurements=10, packet_size=100)

            assert result.metrics["success"] is True
            assert "mean_latency" in result.metrics
            assert "p95_latency" in result.metrics


# Integration tests
class TestBenchmarkIntegration:
    """Integration tests for the complete benchmarking system."""

    def test_end_to_end_workflow(self):
        """Test complete benchmarking workflow."""
        # Create configuration
        config = BenchmarkConfig(name="integration_test", num_trials=1, verbose=False)

        # Create suite
        suite = BenchmarkSuite("Integration Test Suite")
        suite.add_benchmark(SimpleBenchmark("Test 1"))
        suite.add_benchmark(SimpleBenchmark("Test 2"))

        # Run with runner
        runner = StandardRunner(verbose=False)
        results = runner.run_suite(suite, **config.to_dict())

        # Verify results
        assert len(results) == 2
        assert all(r.metrics["success"] for r in results)

        # Test saving results
        with tempfile.TemporaryDirectory() as tmpdir:
            suite.save_results(tmpdir)

            # Check files were created
            result_files = list(Path(tmpdir).glob("*.json"))
            assert len(result_files) >= 2  # At least 2 result files + summary

            # Verify summary file
            summary_file = Path(tmpdir) / "summary.json"
            assert summary_file.exists()

            with open(summary_file) as f:
                summary = json.load(f)
                assert summary["total_benchmarks"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
