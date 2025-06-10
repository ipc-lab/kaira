"""Advanced LDPC Codes Benchmark for Kaira Framework.

This module extends the existing kaira.benchmarks system with specialized
LDPC code evaluation capabilities, including:
- Multiple LDPC code configurations
- Belief propagation decoder analysis :cite:`kschischang2001factor`
- Convergence behavior studies
- Performance vs complexity trade-offs

References:
    :cite:`gallager1962low`, :cite:`mackay2003information`
"""

import time
from typing import Any, Dict, List

import numpy as np
import torch

from kaira.benchmarks.base import CommunicationBenchmark
from kaira.benchmarks.registry import register_benchmark
from kaira.channels.analog import AWGNChannel
from kaira.metrics.signal import BitErrorRate, BlockErrorRate
from kaira.models.fec.decoders import BeliefPropagationDecoder
from kaira.models.fec.encoders import LDPCCodeEncoder


@register_benchmark("ldpc_comprehensive")
class LDPCComprehensiveBenchmark(CommunicationBenchmark):
    """Comprehensive benchmark for LDPC codes with belief propagation decoding."""

    def __init__(self, **kwargs):
        """Initialize LDPC comprehensive benchmark."""
        super().__init__(name="LDPC Comprehensive Benchmark", description="Advanced evaluation of LDPC codes with different configurations")

    def setup(self, **kwargs):
        """Setup benchmark parameters."""
        super().setup(**kwargs)

        # Benchmark configuration
        self.num_messages = kwargs.get("num_messages", 1000)
        self.batch_size = kwargs.get("batch_size", 100)
        self.max_errors = kwargs.get("max_errors", 5)
        self.bp_iterations = kwargs.get("bp_iterations", [5, 10, 20])
        self.snr_range = kwargs.get("snr_range", np.arange(0, 11, 2))
        self.analyze_convergence = kwargs.get("analyze_convergence", True)
        self.max_convergence_iters = kwargs.get("max_convergence_iters", 50)

        # Define LDPC code configurations
        self.ldpc_configs = self._create_ldpc_configurations()

    def _create_ldpc_configurations(self) -> List[Dict[str, Any]]:
        """Create different LDPC code configurations for benchmarking."""

        configs = []

        # Configuration 1: Small regular LDPC (rate 1/2)
        H1 = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        configs.append({"name": "Regular LDPC (6,3)", "parity_check_matrix": H1, "n": 6, "k": 3, "rate": 0.5, "description": "Small regular LDPC, rate=1/2", "category": "regular"})

        # Configuration 2: Larger regular code
        H2 = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        configs.append({"name": "Regular LDPC (8,3)", "parity_check_matrix": H2, "n": 8, "k": 3, "rate": 3 / 8, "description": "Regular LDPC, rate=3/8", "category": "regular"})

        # Configuration 3: Irregular LDPC
        H3 = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)

        configs.append({"name": "Irregular LDPC (9,4)", "parity_check_matrix": H3, "n": 9, "k": 4, "rate": 4 / 9, "description": "Irregular LDPC, rate=4/9", "category": "irregular"})

        # Configuration 4: High-rate code
        H4 = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.float32)

        configs.append({"name": "High-rate LDPC (10,8)", "parity_check_matrix": H4, "n": 10, "k": 8, "rate": 4 / 5, "description": "High-rate LDPC, rate=4/5", "category": "high_rate"})

        return configs

    def _evaluate_ldpc_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate LDPC code performance across SNR and BP iterations."""

        H = config["parity_check_matrix"]
        encoder = LDPCCodeEncoder(check_matrix=H)

        k = config["k"]  # Information bits
        results: Dict[str, Any] = {"success": True}

        # Results storage
        performance_data: Dict[int, Dict[str, List[float]]] = {}

        for bp_iters in self.bp_iterations:
            decoder = BeliefPropagationDecoder(encoder, bp_iters=bp_iters)

            ber_values: List[float] = []
            bler_values: List[float] = []
            decoding_times: List[float] = []
            throughput_values: List[float] = []

            for snr_db in self.snr_range:
                channel = AWGNChannel(snr_db=snr_db)

                # Initialize metrics
                ber_metric = BitErrorRate()
                bler_metric = BlockErrorRate()

                total_decoding_time = 0.0
                total_bits_processed = 0
                num_batches = 0

                # Process in batches
                for batch_idx in range(0, self.num_messages, self.batch_size):
                    current_batch_size = min(self.batch_size, self.num_messages - batch_idx)

                    # Generate test data
                    messages = torch.randint(0, 2, (current_batch_size, k), dtype=torch.float32)

                    # Encode
                    codewords = encoder(messages)

                    # Channel transmission
                    bipolar_codewords = 1 - 2.0 * codewords
                    received_soft = channel(bipolar_codewords)

                    # Decode and measure performance
                    start_time = time.time()
                    decoded_messages = decoder(received_soft)
                    decoding_time = time.time() - start_time

                    total_decoding_time += decoding_time
                    total_bits_processed += messages.numel()
                    num_batches += 1

                    # Update metrics
                    ber_metric.update(messages, decoded_messages)
                    bler_metric.update(messages, decoded_messages)

                # Compute metrics
                ber = ber_metric.compute().item()
                bler = bler_metric.compute().item()
                avg_decoding_time: float = total_decoding_time / num_batches if num_batches > 0 else 0.0
                throughput: float = total_bits_processed / total_decoding_time if total_decoding_time > 0 else 0.0

                ber_values.append(ber)
                bler_values.append(bler)
                decoding_times.append(avg_decoding_time)
                throughput_values.append(throughput)

            performance_data[bp_iters] = {"ber": ber_values, "bler": bler_values, "decoding_time": decoding_times, "throughput": throughput_values}

        results["performance_data"] = performance_data
        return results

    def _analyze_convergence(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze BP convergence behavior."""

        if not self.analyze_convergence:
            return {"success": False, "reason": "Convergence analysis disabled"}

        H = config["parity_check_matrix"]
        encoder = LDPCCodeEncoder(check_matrix=H)

        k = config["k"]  # Information bits
        snr_db = 4.0  # Fixed SNR for convergence analysis
        num_test_messages = 200

        channel = AWGNChannel(snr_db=snr_db)

        # Generate test data
        messages = torch.randint(0, 2, (num_test_messages, k), dtype=torch.float32)
        codewords = encoder(messages)
        bipolar_codewords = 1 - 2.0 * codewords
        received_soft = channel(bipolar_codewords)

        # Test different iteration counts
        iterations_range = np.arange(1, self.max_convergence_iters + 1)
        ber_convergence = []

        for bp_iters in iterations_range:
            decoder = BeliefPropagationDecoder(encoder, bp_iters=bp_iters)
            decoded_messages = decoder(received_soft)

            # Calculate BER
            errors = torch.sum((messages != decoded_messages).float())
            total_bits = messages.numel()
            ber = (errors / total_bits).item()
            ber_convergence.append(ber)

        return {"success": True, "iterations": iterations_range.tolist(), "ber_convergence": ber_convergence, "snr_db": snr_db}

    def _evaluate_complexity_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate computational complexity metrics."""

        H = config["parity_check_matrix"]
        n, k = config["n"], config["k"]

        # Matrix density (sparsity metric)
        total_elements = H.numel()
        nonzero_elements = torch.sum(H).item()
        density = nonzero_elements / total_elements

        # Degree distributions
        var_degrees = torch.sum(H, dim=0)  # Variable node degrees
        check_degrees = torch.sum(H, dim=1)  # Check node degrees

        # Average degrees
        avg_var_degree = torch.mean(var_degrees.float()).item()
        avg_check_degree = torch.mean(check_degrees.float()).item()

        # Estimate computational complexity per iteration
        # Complexity is roughly proportional to number of edges in Tanner graph
        num_edges = int(torch.sum(H).item())

        return {"matrix_density": density, "avg_variable_degree": avg_var_degree, "avg_check_degree": avg_check_degree, "num_edges": num_edges, "estimated_ops_per_iteration": num_edges * 2, "code_rate": k / n}  # Rough estimate

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the LDPC comprehensive benchmark."""

        results: Dict[str, Any] = {
            "benchmark_name": self.name,
            "timestamp": time.time(),
            "config": {"num_messages": self.num_messages, "batch_size": self.batch_size, "bp_iterations": self.bp_iterations, "snr_range": self.snr_range.tolist() if hasattr(self.snr_range, "tolist") else list(self.snr_range), "num_codes_tested": len(self.ldpc_configs)},
            "codes": {},
        }

        print(f"Running {self.name}")
        print(f"Testing {len(self.ldpc_configs)} LDPC configurations")
        print(f"SNR range: {self.snr_range[0]} to {self.snr_range[-1]} dB")
        print(f"BP iterations: {self.bp_iterations}")

        for i, config in enumerate(self.ldpc_configs):
            code_name = config["name"]
            print(f"\n[{i+1}/{len(self.ldpc_configs)}] Evaluating {code_name}...")

            code_results = {"config": config, "complexity_metrics": self._evaluate_complexity_metrics(config), "performance": self._evaluate_ldpc_performance(config), "convergence": self._analyze_convergence(config)}

            results["codes"][code_name] = code_results

            # Print quick summary
            if code_results["performance"]["success"]:
                best_ber = min(code_results["performance"]["performance_data"][self.bp_iterations[-1]]["ber"])
                print(f"  Best BER: {best_ber:.2e} (BP={self.bp_iterations[-1]} iters)")
                print(f"  Code rate: {config['rate']:.3f}")
                print(f"  Matrix density: {code_results['complexity_metrics']['matrix_density']:.3f}")

        # Generate summary statistics
        results["summary"] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""

        summary: Dict[str, Any] = {"total_codes_tested": len(results["codes"]), "successful_evaluations": 0, "best_performers": {}, "complexity_analysis": {}, "convergence_analysis": {}}

        # Collect performance data for analysis
        best_ber_overall = float("inf")
        best_ber_code = None
        highest_rate = 0
        highest_rate_code = None
        lowest_complexity = float("inf")
        lowest_complexity_code = None

        for code_name, code_results in results["codes"].items():
            if code_results["performance"]["success"]:
                summary["successful_evaluations"] += 1

                # Find best BER performance
                bp_iters = self.bp_iterations[-1]  # Use highest iteration count
                # Cast to avoid mypy issues with nested dictionary access
                performance_data = code_results["performance"]["performance_data"]
                ber_values = performance_data[bp_iters]["ber"]
                best_ber = min(ber_values)

                if best_ber < best_ber_overall:
                    best_ber_overall = best_ber
                    best_ber_code = code_name

                # Find highest rate
                rate = code_results["config"]["rate"]
                if rate > highest_rate:
                    highest_rate = rate
                    highest_rate_code = code_name

                # Find lowest complexity
                complexity = code_results["complexity_metrics"]["estimated_ops_per_iteration"]
                if complexity < lowest_complexity:
                    lowest_complexity = complexity
                    lowest_complexity_code = code_name

        summary["best_performers"] = {"best_ber": {"code": best_ber_code, "value": best_ber_overall}, "highest_rate": {"code": highest_rate_code, "value": highest_rate}, "lowest_complexity": {"code": lowest_complexity_code, "value": lowest_complexity}}

        return summary


@register_benchmark("ldpc_quick")
class LDPCQuickBenchmark(CommunicationBenchmark):
    """Quick LDPC benchmark for rapid evaluation."""

    def __init__(self, **kwargs):
        """Initialize quick LDPC benchmark."""
        super().__init__(name="LDPC Quick Benchmark", description="Fast evaluation of basic LDPC code performance")

    def setup(self, **kwargs):
        """Setup quick benchmark parameters."""
        super().setup(**kwargs)

        # Reduced parameters for quick evaluation
        self.num_messages = kwargs.get("num_messages", 100)
        self.batch_size = kwargs.get("batch_size", 50)
        self.bp_iterations = [10]  # Single iteration count
        self.snr_range = np.array([2, 6, 10])  # Limited SNR range

        # Simple LDPC configuration
        H = torch.tensor([[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

        self.ldpc_config = {"name": "Quick Test LDPC (6,3)", "parity_check_matrix": H, "n": 6, "k": 3, "rate": 0.5}

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run quick LDPC benchmark."""

        print(f"Running {self.name}")

        H = self.ldpc_config["parity_check_matrix"]
        encoder = LDPCCodeEncoder(check_matrix=H)
        decoder = BeliefPropagationDecoder(encoder, bp_iters=self.bp_iterations[0])

        k = self.ldpc_config["k"]
        ber_values = []

        for snr_db in self.snr_range:
            channel = AWGNChannel(snr_db=snr_db)
            ber_metric = BitErrorRate()

            # Generate test data
            messages = torch.randint(0, 2, (self.num_messages, k), dtype=torch.float32)
            codewords = encoder(messages)

            # Channel transmission
            bipolar_codewords = 1 - 2.0 * codewords
            received_soft = channel(bipolar_codewords)

            # Decode
            decoded_messages = decoder(received_soft)

            # Calculate BER
            ber_metric.update(messages, decoded_messages)
            ber = ber_metric.compute().item()
            ber_values.append(ber)

        results = {"benchmark_name": self.name, "timestamp": time.time(), "config": self.ldpc_config, "snr_range": self.snr_range.tolist() if hasattr(self.snr_range, "tolist") else list(self.snr_range), "ber_values": ber_values, "bp_iterations": self.bp_iterations[0]}

        print("Quick benchmark completed:")
        for snr, ber in zip(self.snr_range, ber_values):
            print(f"  SNR {snr} dB: BER = {ber:.2e}")

        return results


# Example usage and testing
if __name__ == "__main__":
    print("LDPC Benchmarks for Kaira Framework")
    print("=" * 50)

    # Run quick benchmark
    print("\n1. Running Quick LDPC Benchmark...")
    quick_benchmark = LDPCQuickBenchmark()
    quick_benchmark.setup()
    quick_results = quick_benchmark.run()

    # Run comprehensive benchmark (commented out for demo)
    # print("\n2. Running Comprehensive LDPC Benchmark...")
    # comprehensive_benchmark = LDPCComprehensiveBenchmark()
    # comprehensive_benchmark.setup(num_messages=200)  # Reduced for demo
    # comprehensive_results = comprehensive_benchmark.run()

    print("\nBenchmark demonstrations completed!")
