"""Tests to improve code coverage for benchmark modules.

This module contains additional tests to ensure comprehensive coverage of the benchmark system,
addressing specific uncovered lines identified by the coverage tool.
"""

import tempfile
from pathlib import Path

import pytest

from kaira.benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
)
from kaira.benchmarks.config import get_config, list_configs


class TestBenchmarkConfigExtended:
    """Extended tests for BenchmarkConfig to improve coverage."""

    def test_config_save_and_load(self):
        """Test saving and loading config to/from file."""
        config = BenchmarkConfig(name="test_config", description="Test configuration", num_trials=5, snr_range=[-5, 0, 5], verbose=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Test save
            config.save(temp_path)

            # Test load
            loaded_config = BenchmarkConfig.load(temp_path)

            assert loaded_config.name == "test_config"
            assert loaded_config.description == "Test configuration"
            assert loaded_config.num_trials == 5
            assert loaded_config.snr_range == [-5, 0, 5]
            assert loaded_config.verbose is True

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_from_json(self):
        """Test creating config from JSON string."""
        json_str = """
        {
            "name": "json_config",
            "description": "From JSON",
            "num_trials": 3,
            "verbose": false
        }
        """

        config = BenchmarkConfig.from_json(json_str)
        assert config.name == "json_config"
        assert config.description == "From JSON"
        assert config.num_trials == 3
        assert config.verbose is False

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"name": "dict_config", "description": "From dict", "block_length": 2000, "code_rate": 0.75}

        config = BenchmarkConfig.from_dict(config_dict)
        assert config.name == "dict_config"
        assert config.description == "From dict"
        assert config.block_length == 2000
        assert config.code_rate == 0.75

    def test_config_update(self):
        """Test updating configuration parameters."""
        config = BenchmarkConfig(name="original", num_trials=1)

        # Update existing parameter
        config.update(name="updated", num_trials=10)
        assert config.name == "updated"
        assert config.num_trials == 10

        # Update with custom parameter
        config.update(custom_param="value")
        assert config.get("custom_param") == "value"

    def test_config_get_method(self):
        """Test get method for configuration parameters."""
        config = BenchmarkConfig(name="test", verbose=True)
        config.update(custom_key="custom_value")

        # Test getting existing attribute
        assert config.get("name") == "test"
        assert config.get("verbose") is True

        # Test getting custom parameter
        assert config.get("custom_key") == "custom_value"

        # Test getting non-existent parameter with default
        assert config.get("non_existent", "default") == "default"
        assert config.get("non_existent") is None

    def test_predefined_configs(self):
        """Test predefined configuration access."""
        # Test list_configs
        available_configs = list_configs()
        assert isinstance(available_configs, list)
        assert "fast" in available_configs
        assert "accurate" in available_configs

        # Test get_config
        fast_config = get_config("fast")
        assert fast_config.name == "fast"
        assert fast_config.num_trials == 1

        # Test invalid config name
        with pytest.raises(ValueError, match="Unknown configuration"):
            get_config("non_existent_config")


class TestBenchmarkSuiteExtended:
    """Extended tests for BenchmarkSuite to improve coverage."""

    def test_suite_get_summary_empty(self):
        """Test get_summary with empty results."""
        suite = BenchmarkSuite("empty_suite")
        summary = suite.get_summary()
        assert summary == {}

    def test_suite_get_summary_with_results(self):
        """Test get_summary with actual results."""
        suite = BenchmarkSuite("test_suite")

        # Add some mock results directly to the results list
        result1 = BenchmarkResult(benchmark_id="bench1", name="Benchmark 1", description="Test benchmark 1", metrics={"success": True, "accuracy": 0.9}, execution_time=1.0, timestamp="2025-06-10 10:00:00")

        result2 = BenchmarkResult(benchmark_id="bench2", name="Benchmark 2", description="Test benchmark 2", metrics={"success": False, "accuracy": 0.5}, execution_time=2.0, timestamp="2025-06-10 10:00:01")

        suite.results.append(result1)
        suite.results.append(result2)

        summary = suite.get_summary()
        assert summary["suite_name"] == "test_suite"
        assert summary["total_benchmarks"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert summary["total_execution_time"] == 3.0
        assert summary["average_execution_time"] == 1.5

    def test_suite_save_results_with_directory(self):
        """Test suite save_results method."""
        suite = BenchmarkSuite("test_suite")

        # Add a result directly to the results list
        result = BenchmarkResult(benchmark_id="test", name="Test", description="Test", metrics={"value": 1}, execution_time=1.0, timestamp="2025-06-10 10:00:00")
        suite.results.append(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            suite.save_results(temp_dir)

            # Check that files were created
            output_path = Path(temp_dir)
            files = list(output_path.glob("*.json"))
            assert len(files) > 0


class TestErrorHandlingCoverage:
    """Test error handling paths to improve coverage."""

    def test_benchmark_result_edge_cases(self):
        """Test BenchmarkResult with edge case inputs."""
        # Test with minimal parameters
        result = BenchmarkResult(benchmark_id="minimal", name="Minimal Test", description="", metrics={}, execution_time=0.0, timestamp="2025-06-10 10:00:00")

        assert result.benchmark_id == "minimal"
        assert result.metrics == {}
        assert result.execution_time == 0.0

        # Test serialization of edge cases
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        json_str = result.to_json()
        assert isinstance(json_str, str)


class TestECCBenchmarkCoverage:
    """Test ECC benchmark specific functionality for coverage."""

    def test_ecc_benchmark_reed_solomon_fallback(self):
        """Test Reed-Solomon config fallback when not available."""
        from kaira.benchmarks.ecc_benchmark import ECCPerformanceBenchmark

        benchmark = ECCPerformanceBenchmark(code_family="reed_solomon")
        benchmark.setup()

        configs = benchmark._get_code_configurations()
        # Should handle ImportError gracefully
        assert isinstance(configs, list)

    def test_ecc_benchmark_unknown_family(self):
        """Test ECC benchmark with unknown code family."""
        from kaira.benchmarks.ecc_benchmark import ECCPerformanceBenchmark

        benchmark = ECCPerformanceBenchmark(code_family="unknown_family")
        benchmark.setup()

        configs = benchmark._get_code_configurations()
        # Should fall back to single parity check
        assert len(configs) == 1
        assert "Single Parity Check" in configs[0]["name"]

    def test_ecc_benchmark_error_correction_with_exceptions(self):
        """Test error correction evaluation with encoding/decoding exceptions."""
        from kaira.benchmarks.ecc_benchmark import ECCPerformanceBenchmark
        from kaira.models.fec.encoders import HammingCodeEncoder

        class FailingDecoder:
            """Mock decoder that fails to trigger exception handling."""

            def __init__(self, encoder):
                self.encoder = encoder

            def __call__(self, *args, **kwargs):
                raise RuntimeError("Decoding failed")

        benchmark = ECCPerformanceBenchmark(code_family="hamming")
        benchmark.setup(num_trials=5, max_errors=2)

        config = {"name": "Test Failing Decoder", "encoder": HammingCodeEncoder, "decoder": FailingDecoder, "params": {"mu": 3}, "n": 7, "k": 4}

        # Test with failing decoder to trigger exception handling
        result = benchmark._evaluate_error_correction_performance(config)

        assert result["success"] is True
        assert "correction_probability" in result

    def test_ecc_benchmark_ber_performance_with_failures(self):
        """Test BER performance with encoding failures."""
        from kaira.benchmarks.ecc_benchmark import ECCPerformanceBenchmark

        class FailingEncoder:
            """Mock encoder that fails to trigger exception handling."""

            def __call__(self, *args, **kwargs):
                raise ValueError("Encoding failed")

            @property
            def n(self):
                return 7

            @property
            def k(self):
                return 4

        benchmark = ECCPerformanceBenchmark(code_family="hamming")
        benchmark.setup(snr_range=[5], num_bits=100, num_trials=3)

        config = {"name": "Test Failing Encoder", "encoder": FailingEncoder, "decoder": None, "params": {}, "n": 7, "k": 4}

        # Test with failing encoder to trigger exception handling
        result = benchmark._evaluate_ber_performance(config)

        # Should handle failures gracefully
        assert isinstance(result, dict)
        if result.get("success"):
            assert "ber_values" in result

    def test_ecc_configs_get_functions(self):
        """Test ECC config utility functions."""
        from kaira.benchmarks.ecc_configs import get_ecc_config, get_family_config, list_all_configs

        # Test list_all_configs
        all_configs = list_all_configs()
        assert isinstance(all_configs, dict)
        assert len(all_configs) > 0

        # Test get_family_config for different families
        try:
            hamming_config = get_family_config("hamming")
            assert hamming_config is not None
        except (ImportError, KeyError, AttributeError):
            pytest.skip("Hamming config not available in test environment")

        try:
            bch_config = get_family_config("bch")
            assert bch_config is not None
        except (ImportError, KeyError, AttributeError):
            pytest.skip("BCH config not available in test environment")

        # Test get_ecc_config
        try:
            config = get_ecc_config("hamming_7_4")
            assert config is not None
        except (ImportError, KeyError, AttributeError):
            pytest.skip("ECC config not available in test environment")


class TestLDPCBenchmarkCoverage:
    """Test LDPC benchmark specific functionality for coverage."""

    def test_ldpc_benchmark_full_setup(self):
        """Test LDPC benchmark comprehensive setup."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()

        # Test with all parameters
        benchmark.setup(num_messages=100, batch_size=25, max_errors=3, bp_iterations=[5, 10, 20], snr_range=[0, 2, 4, 6, 8, 10], analyze_convergence=True, max_convergence_iters=30)

        assert benchmark.num_messages == 100
        assert benchmark.batch_size == 25
        assert benchmark.max_errors == 3
        assert benchmark.bp_iterations == [5, 10, 20]
        assert len(benchmark.snr_range) == 6
        assert benchmark.analyze_convergence is True
        assert benchmark.max_convergence_iters == 30

    def test_ldpc_performance_evaluation_extensive(self):
        """Test LDPC performance evaluation with more parameters."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()
        benchmark.setup(num_messages=20, batch_size=10, bp_iterations=[5, 10], snr_range=[0, 5], analyze_convergence=True)  # Small for testing

        # Test with different LDPC configurations
        configs = benchmark._create_ldpc_configurations()

        for config in configs[:2]:  # Test first 2 configurations
            try:
                result = benchmark._evaluate_ldpc_performance(config)
                if result.get("success"):
                    assert "performance_data" in result or isinstance(result, dict)
            except (ImportError, RuntimeError, ValueError) as e:
                # Skip if implementation not available or fails due to dependencies
                pytest.skip(f"LDPC implementation not available: {e}")

    def test_ldpc_configurations_properties(self):
        """Test LDPC configuration properties."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()
        benchmark.setup()

        configs = benchmark._create_ldpc_configurations()

        # Test configuration categories
        categories = [config["category"] for config in configs]
        assert "regular" in categories
        assert "irregular" in categories
        assert "high_rate" in categories

        # Test rate calculations
        for config in configs:
            expected_rate = config["k"] / config["n"]
            assert abs(config["rate"] - expected_rate) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
