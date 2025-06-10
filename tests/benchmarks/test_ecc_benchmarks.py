"""Tests for Error Correction Codes benchmarks.

This module provides comprehensive tests for the ECC benchmarking system, ensuring correctness and
reliability of benchmark implementations.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from kaira.benchmarks import StandardRunner, create_benchmark
from kaira.benchmarks.ecc_benchmark import ECCComparisonBenchmark, ECCPerformanceBenchmark
from kaira.benchmarks.ecc_configs import (
    create_custom_ecc_config,
    get_ecc_config,
    get_family_config,
    get_suite_config,
    list_all_configs,
)


class TestECCPerformanceBenchmark:
    """Test the ECC performance benchmark."""

    def test_benchmark_registration(self):
        """Test that ECC benchmarks are properly registered."""
        # Test individual family benchmark registration
        benchmark = create_benchmark("ecc_performance", code_family="hamming")
        assert benchmark is not None
        assert isinstance(benchmark, ECCPerformanceBenchmark)

        # Test comparison benchmark registration
        comparison_benchmark = create_benchmark("ecc_comparison")
        assert comparison_benchmark is not None
        assert isinstance(comparison_benchmark, ECCComparisonBenchmark)

    def test_hamming_family_benchmark(self):
        """Test Hamming code family benchmark."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="hamming")

        # Use minimal configuration for testing
        result = runner.run_benchmark(benchmark, snr_range=[-2, 0, 2], num_bits=1000, num_trials=10, max_errors=3, evaluate_complexity=False, evaluate_throughput=False)

        assert result.metrics["success"]
        assert result.metrics["code_family"] == "hamming"
        assert len(result.metrics["configurations"]) > 0

        # Check that we have results for at least one configuration
        config_names = [config["name"] for config in result.metrics["configurations"]]
        assert "Hamming(7,4)" in config_names

        # Check error correction results
        for config_name in config_names:
            if config_name in result.metrics["error_correction_results"]:
                ec_result = result.metrics["error_correction_results"][config_name]
                if ec_result["success"]:
                    assert "correction_probability" in ec_result
                    assert len(ec_result["correction_probability"]) == 4  # 0 to 3 errors

    def test_bch_family_benchmark(self):
        """Test BCH code family benchmark."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="bch")

        result = runner.run_benchmark(benchmark, snr_range=[0, 5], num_bits=500, num_trials=5, max_errors=2, evaluate_complexity=False, evaluate_throughput=False)

        assert result.metrics["success"]
        assert result.metrics["code_family"] == "bch"

        # Check that at least one configuration ran successfully
        successful_configs = [config for config in result.metrics["configurations"] if result.metrics["ber_performance_results"][config["name"]]["success"]]
        assert len(successful_configs) > 0

    def test_golay_family_benchmark(self):
        """Test Golay code family benchmark."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="golay")

        result = runner.run_benchmark(benchmark, snr_range=[2, 8], num_bits=240, num_trials=5, max_errors=3, evaluate_complexity=False, evaluate_throughput=False)  # Multiple of 12 for Golay

        assert result.metrics["success"]
        assert result.metrics["code_family"] == "golay"

        # Check for both standard and extended Golay codes
        config_names = [config["name"] for config in result.metrics["configurations"]]
        assert "Golay(23,12)" in config_names
        assert "Extended Golay(24,12)" in config_names

    def test_repetition_family_benchmark(self):
        """Test repetition code family benchmark."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="repetition")

        result = runner.run_benchmark(benchmark, snr_range=[-5, 0, 5], num_bits=100, num_trials=10, max_errors=2, evaluate_complexity=False, evaluate_throughput=False)  # Small for repetition codes

        assert result.metrics["success"]
        assert result.metrics["code_family"] == "repetition"

        # Repetition codes should show good performance at low SNR
        for config in result.metrics["configurations"]:
            config_name = config["name"]
            if config_name in result.metrics["ber_performance_results"]:
                ber_result = result.metrics["ber_performance_results"][config_name]
                if ber_result["success"]:
                    # Should have coding gain
                    gains = [g for g in ber_result["coding_gain_ber"] if torch.isfinite(torch.tensor(g))]
                    if gains:
                        assert max(gains) > 0  # Should provide some coding gain

    def test_invalid_family(self):
        """Test behavior with invalid code family."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="invalid_family")

        result = runner.run_benchmark(benchmark, snr_range=[0], num_bits=100, num_trials=1, max_errors=1)

        # Should still succeed but with empty or minimal results
        assert result.metrics["success"]
        # Invalid family should result in default configuration (single parity check)
        assert len(result.metrics["configurations"]) > 0


class TestECCComparisonBenchmark:
    """Test the ECC comparison benchmark."""

    def test_family_comparison(self):
        """Test comparison of multiple ECC families."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_comparison")

        result = runner.run_benchmark(benchmark, snr_range=[0, 5], num_bits=500, families=["hamming", "repetition"])  # Use simple families for testing

        assert result.metrics["success"]
        assert set(result.metrics["families_compared"]) == {"hamming", "repetition"}

        # Check that we have results for both families
        for family in ["hamming", "repetition"]:
            assert family in result.metrics["family_results"]
            family_result = result.metrics["family_results"][family]
            assert family_result["success"]

        # Check comparison summary
        assert "comparison_summary" in result.metrics
        summary = result.metrics["comparison_summary"]
        assert "best_for_ber_gain" in summary
        assert "families_evaluated" in summary
        assert summary["families_evaluated"] >= 2

    def test_single_family_comparison(self):
        """Test comparison with single family."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_comparison")

        result = runner.run_benchmark(benchmark, snr_range=[0], num_bits=100, families=["repetition"])

        assert result.metrics["success"]
        assert result.metrics["families_compared"] == ["repetition"]


class TestECCConfigurations:
    """Test ECC configuration utilities."""

    def test_predefined_configs(self):
        """Test predefined configuration access."""
        # Test all predefined configurations
        config_names = ["fast", "standard", "comprehensive", "high_snr", "low_complexity"]

        for config_name in config_names:
            config = get_ecc_config(config_name)
            assert config.name == f"ecc_{config_name}_evaluation"
            assert isinstance(config.snr_range, list)
            # Check custom_params for num_bits and num_trials
            assert config.custom_params.get("num_bits", config.block_length) > 0
            assert config.custom_params.get("num_trials", 100) > 0

    def test_invalid_config_name(self):
        """Test behavior with invalid configuration name."""
        with pytest.raises(KeyError):
            get_ecc_config("nonexistent_config")

    def test_family_configs(self):
        """Test family-specific configurations."""
        families = ["hamming", "bch", "golay", "repetition", "reed_solomon"]

        for family in families:
            family_config = get_family_config(family)
            assert "codes_to_test" in family_config
            assert "focus_metrics" in family_config
            assert "recommended_snr_range" in family_config
            assert len(family_config["codes_to_test"]) > 0

    def test_suite_configs(self):
        """Test benchmark suite configurations."""
        suites = ["academic_comparison", "industry_evaluation", "satellite_communications", "iot_embedded"]

        for suite in suites:
            suite_config = get_suite_config(suite)
            assert "name" in suite_config
            assert "description" in suite_config
            assert "families" in suite_config
            assert "base_config" in suite_config

    def test_custom_config_creation(self):
        """Test creation of custom configurations."""
        custom_config = create_custom_ecc_config(name="test_custom", snr_range=[0, 5, 10], num_bits=1000, num_trials=50, max_errors=5, description="Test custom configuration")

        assert custom_config.name == "test_custom"
        assert custom_config.snr_range == [0, 5, 10]
        assert custom_config.block_length == 1000
        assert custom_config.custom_params["num_bits"] == 1000
        assert custom_config.custom_params["num_trials"] == 50
        assert custom_config.custom_params["max_errors"] == 5

    def test_list_all_configs(self):
        """Test listing all available configurations."""
        all_configs = list_all_configs()

        assert "benchmark_configs" in all_configs
        assert "family_configs" in all_configs
        assert "suite_configs" in all_configs

        assert "fast" in all_configs["benchmark_configs"]
        assert "hamming" in all_configs["family_configs"]
        assert "academic_comparison" in all_configs["suite_configs"]


class TestECCBenchmarkIntegration:
    """Test integration of ECC benchmarks with existing system."""

    def test_result_saving(self):
        """Test that results can be saved properly."""
        runner = StandardRunner()
        benchmark = create_benchmark("ecc_performance", code_family="repetition")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.run_benchmark(benchmark, snr_range=[0], num_bits=50, num_trials=2, max_errors=1, output_directory=temp_dir)

            # Save result
            result_path = Path(temp_dir) / "test_result.json"
            result.save(result_path)

            # Check that file was created and contains expected data
            assert result_path.exists()
            with open(result_path) as f:
                import json

                saved_data = json.load(f)
                assert saved_data["name"] == result.name
                assert "success" in saved_data["metrics"]

    def test_benchmark_with_standard_runner(self):
        """Test ECC benchmarks work with standard benchmark runner."""
        runner = StandardRunner(verbose=False)

        # Test with different configurations
        configs = [("ecc_performance", {"code_family": "hamming"}), ("ecc_comparison", {"families": ["repetition"]})]

        for benchmark_name, params in configs:
            benchmark = create_benchmark(benchmark_name, **params)
            result = runner.run_benchmark(benchmark, snr_range=[0], num_bits=50, num_trials=1, max_errors=1)

            assert result.metrics["success"]
            assert result.execution_time > 0

    def test_error_handling(self):
        """Test error handling in ECC benchmarks."""
        runner = StandardRunner()

        # Test with extreme parameters that might cause issues
        benchmark = create_benchmark("ecc_performance", code_family="hamming")

        # This should not crash, even with extreme parameters
        result = runner.run_benchmark(benchmark, snr_range=[100], num_bits=10, num_trials=1, max_errors=100)  # Very high SNR  # Very few bits  # Single trial  # More errors than bits

        # Should still complete, even if some measurements fail
        assert isinstance(result.metrics, dict)
        assert "success" in result.metrics


# Utility function for running all tests
def run_ecc_benchmark_tests():
    """Run all ECC benchmark tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_ecc_benchmark_tests()
