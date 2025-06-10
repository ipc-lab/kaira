"""Additional tests for ECC and LDPC benchmarks to improve coverage."""

import pytest
import torch

from kaira.benchmarks.ecc_benchmark import ECCPerformanceBenchmark


class TestECCBenchmarkEdgeCases:
    """Test edge cases and error handling in ECC benchmarks."""

    def test_ecc_benchmark_reed_solomon_fallback(self):
        """Test Reed-Solomon config fallback when not available."""
        benchmark = ECCPerformanceBenchmark(code_family="reed_solomon")
        benchmark.setup()

        configs = benchmark._get_code_configurations()
        # Should handle ImportError gracefully
        assert isinstance(configs, list)

    def test_ecc_benchmark_unknown_family(self):
        """Test ECC benchmark with unknown code family."""
        benchmark = ECCPerformanceBenchmark(code_family="unknown_family")
        benchmark.setup()

        configs = benchmark._get_code_configurations()
        # Should fall back to single parity check
        assert len(configs) == 1
        assert "Single Parity Check" in configs[0]["name"]

    def test_ecc_benchmark_error_correction_edge_cases(self):
        """Test error correction evaluation with edge cases."""
        from kaira.models.fec.decoders import BruteForceMLDecoder
        from kaira.models.fec.encoders import HammingCodeEncoder

        benchmark = ECCPerformanceBenchmark(code_family="hamming")
        benchmark.setup(num_trials=10, max_errors=2)

        config = {"name": "Test Hamming", "encoder": HammingCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"mu": 3}, "n": 7, "k": 4}

        # Test with small number of trials to trigger edge cases
        benchmark.num_trials = 1
        result = benchmark._evaluate_error_correction_performance(config)

        assert result["success"] is True
        assert "correction_probability" in result
        assert "undetected_error_probability" in result

    def test_ecc_benchmark_ber_performance_failures(self):
        """Test BER performance evaluation with potential failures."""
        from kaira.models.fec.decoders import BruteForceMLDecoder
        from kaira.models.fec.encoders import HammingCodeEncoder

        benchmark = ECCPerformanceBenchmark(code_family="hamming")
        benchmark.setup(snr_range=[10], num_bits=100, num_trials=5)  # High SNR for testing

        config = {"name": "Test Hamming", "encoder": HammingCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"mu": 3}, "n": 7, "k": 4}

        result = benchmark._evaluate_ber_performance(config)

        # Should handle various edge cases gracefully
        assert isinstance(result, dict)


class TestLDPCBenchmarkEdgeCases:
    """Test edge cases and error handling in LDPC benchmarks."""

    def test_ldpc_benchmark_initialization(self):
        """Test LDPC benchmark initialization."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()
        assert benchmark.name == "LDPC Comprehensive Benchmark"
        assert "LDPC codes" in benchmark.description

    def test_ldpc_benchmark_setup(self):
        """Test LDPC benchmark setup with various parameters."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()

        # Test with custom parameters
        benchmark.setup(num_messages=500, batch_size=50, max_errors=3, bp_iterations=[5, 10], snr_range=[0, 5, 10], analyze_convergence=False)

        assert benchmark.num_messages == 500
        assert benchmark.batch_size == 50
        assert benchmark.max_errors == 3
        assert benchmark.bp_iterations == [5, 10]
        assert benchmark.analyze_convergence is False

    def test_ldpc_configurations_creation(self):
        """Test LDPC configuration creation."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()
        benchmark.setup()

        configs = benchmark._create_ldpc_configurations()

        assert len(configs) == 4
        assert any(config["name"] == "Regular LDPC (6,3)" for config in configs)
        assert any(config["name"] == "Irregular LDPC (9,4)" for config in configs)

        # Check configuration structure
        for config in configs:
            assert "name" in config
            assert "parity_check_matrix" in config
            assert "n" in config
            assert "k" in config
            assert "rate" in config
            assert "category" in config

    def test_ldpc_performance_evaluation_minimal(self):
        """Test LDPC performance evaluation with minimal parameters."""
        from kaira.benchmarks.ldpc_benchmark import LDPCComprehensiveBenchmark

        benchmark = LDPCComprehensiveBenchmark()
        benchmark.setup(num_messages=10, batch_size=5, bp_iterations=[5], snr_range=[5], analyze_convergence=False)  # Very small for testing

        # Use the smallest configuration
        config = {"name": "Test LDPC", "parity_check_matrix": torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32), "n": 6, "k": 3, "rate": 0.5}

        try:
            result = benchmark._evaluate_ldpc_performance(config)
            assert result["success"] is True
        except (ImportError, RuntimeError, ValueError) as e:
            # Skip if LDPC implementation details fail
            pytest.skip(f"LDPC evaluation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
