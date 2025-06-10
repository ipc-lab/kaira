"""Advanced Error Correction Codes Benchmark for Kaira.

This module provides comprehensive benchmarking capabilities for Forward Error Correction (FEC)
codes, extending the existing kaira.benchmarks system with specialized ECC evaluation tools.
"""

import time
from typing import Any, Dict, List

import numpy as np
import torch

from kaira.benchmarks.base import CommunicationBenchmark
from kaira.benchmarks.metrics import StandardMetrics
from kaira.benchmarks.registry import register_benchmark
from kaira.models.fec.decoders import (
    BerlekampMasseyDecoder,
    BruteForceMLDecoder,
    SyndromeLookupDecoder,
)
from kaira.models.fec.encoders import (
    BCHCodeEncoder,
    GolayCodeEncoder,
    HammingCodeEncoder,
    ReedSolomonCodeEncoder,
    RepetitionCodeEncoder,
    SingleParityCheckCodeEncoder,
)


@register_benchmark("ecc_performance")
class ECCPerformanceBenchmark(CommunicationBenchmark):
    """Comprehensive benchmark for error correction code performance evaluation."""

    def __init__(self, code_family: str = "hamming", **kwargs):
        """Initialize ECC performance benchmark.

        Args:
            code_family: Family of codes to benchmark ('hamming', 'bch', 'golay', etc.)
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"ECC Performance ({code_family.upper()})", description=f"Comprehensive performance evaluation for {code_family} codes")
        self.code_family = code_family.lower()

    def setup(self, **kwargs):
        """Setup benchmark parameters."""
        super().setup(**kwargs)
        # Get parameters from kwargs or config with more conservative defaults
        self.num_bits = kwargs.get("num_bits", kwargs.get("block_length", 1000))  # Reduced from 10000
        self.num_trials = kwargs.get("num_trials", 10)  # Reduced from 100
        self.max_errors = kwargs.get("max_errors", 5)  # Reduced from 10
        self.evaluate_complexity = kwargs.get("evaluate_complexity", True)
        self.evaluate_throughput = kwargs.get("evaluate_throughput", True)

        # Define code configurations based on family
        self.code_configs = self._get_code_configurations()

    def _get_code_configurations(self) -> List[Dict[str, Any]]:
        """Get code configurations for the selected family."""
        configs = []

        if self.code_family == "hamming":
            configs = [
                {"name": "Hamming(7,4)", "encoder": HammingCodeEncoder, "decoder": SyndromeLookupDecoder, "params": {"mu": 3}, "n": 7, "k": 4, "d": 3, "t": 1},
                {"name": "Hamming(15,11)", "encoder": HammingCodeEncoder, "decoder": SyndromeLookupDecoder, "params": {"mu": 4}, "n": 15, "k": 11, "d": 3, "t": 1},
                {"name": "Hamming(31,26)", "encoder": HammingCodeEncoder, "decoder": SyndromeLookupDecoder, "params": {"mu": 5}, "n": 31, "k": 26, "d": 3, "t": 1},
            ]
        elif self.code_family == "bch":
            configs = [
                {"name": "BCH(15,7)", "encoder": BCHCodeEncoder, "decoder": BerlekampMasseyDecoder, "params": {"mu": 4, "delta": 5}, "n": 15, "k": 7, "d": 5, "t": 2},
                {"name": "BCH(31,16)", "encoder": BCHCodeEncoder, "decoder": BerlekampMasseyDecoder, "params": {"mu": 5, "delta": 7}, "n": 31, "k": 16, "d": 7, "t": 3},
                {"name": "BCH(63,36)", "encoder": BCHCodeEncoder, "decoder": BerlekampMasseyDecoder, "params": {"mu": 6, "delta": 11}, "n": 63, "k": 36, "d": 11, "t": 5},
            ]
        elif self.code_family == "golay":
            configs = [
                {"name": "Golay(23,12)", "encoder": GolayCodeEncoder, "decoder": SyndromeLookupDecoder, "params": {"extended": False}, "n": 23, "k": 12, "d": 7, "t": 3},
                {"name": "Extended Golay(24,12)", "encoder": GolayCodeEncoder, "decoder": SyndromeLookupDecoder, "params": {"extended": True}, "n": 24, "k": 12, "d": 8, "t": 3},
            ]
        elif self.code_family == "repetition":
            configs = [
                {"name": "Repetition(3,1)", "encoder": RepetitionCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"repetition_factor": 3}, "n": 3, "k": 1, "d": 3, "t": 1},
                {"name": "Repetition(5,1)", "encoder": RepetitionCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"repetition_factor": 5}, "n": 5, "k": 1, "d": 5, "t": 2},
                {"name": "Repetition(7,1)", "encoder": RepetitionCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"repetition_factor": 7}, "n": 7, "k": 1, "d": 7, "t": 3},
            ]
        elif self.code_family == "reed_solomon":
            try:
                configs = [
                    {"name": "Reed-Solomon(15,11)", "encoder": ReedSolomonCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"n": 15, "k": 11}, "n": 15, "k": 11, "d": 5, "t": 2},
                    {"name": "Reed-Solomon(31,19)", "encoder": ReedSolomonCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"n": 31, "k": 19}, "n": 31, "k": 19, "d": 13, "t": 6},
                ]
            except ImportError:
                # Fallback if Reed-Solomon not available
                configs = []
        else:
            # Default to single parity check
            configs = [{"name": "Single Parity Check(8,7)", "encoder": SingleParityCheckCodeEncoder, "decoder": BruteForceMLDecoder, "params": {"info_length": 7}, "n": 8, "k": 7, "d": 2, "t": 0}]

        return configs

    def _evaluate_error_correction_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate error correction performance for a specific code configuration."""
        encoder_class = config["encoder"]
        decoder_class = config["decoder"]

        try:
            encoder = encoder_class(**config["params"])
            decoder = decoder_class(encoder)
        except Exception as e:
            return {"success": False, "error": str(e), "correction_probability": [], "undetected_error_probability": []}

        correction_probs = []
        undetected_error_probs = []

        # Test different numbers of errors
        for num_errors in range(self.max_errors + 1):
            corrections = 0
            undetected_errors = 0

            for _ in range(self.num_trials):
                # Generate random information
                info_bits = torch.randint(0, 2, (config["k"],), dtype=torch.float32, device=self.device)

                # Encode (use forward method)
                try:
                    codeword = encoder(info_bits)
                except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                    # Skip trials with encoding failures (dimension mismatches, invalid parameters, etc.)
                    continue

                # Add random errors
                error_pattern = torch.zeros_like(codeword)
                if num_errors > 0:
                    error_positions = torch.randperm(len(codeword))[:num_errors]
                    error_pattern[error_positions] = 1

                received = (codeword + error_pattern) % 2

                # Decode (use forward method)
                try:
                    decoded_info = decoder(received)

                    if torch.equal(info_bits, decoded_info):
                        corrections += 1
                    else:
                        # This is an undetected error
                        undetected_errors += 1

                except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                    # Decoding failure - this could be error detection
                    # Count as detection for codes with error detection capability
                    pass

            correction_prob = corrections / self.num_trials if self.num_trials > 0 else 0
            undetected_prob = undetected_errors / self.num_trials if self.num_trials > 0 else 0

            correction_probs.append(correction_prob)
            undetected_error_probs.append(undetected_prob)

        return {"success": True, "correction_probability": correction_probs, "undetected_error_probability": undetected_error_probs}

    def _evaluate_ber_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate BER performance over SNR range."""
        encoder_class = config["encoder"]
        decoder_class = config["decoder"]

        try:
            encoder = encoder_class(**config["params"])
            decoder = decoder_class(encoder)
        except Exception as e:
            return {"success": False, "error": str(e), "ber_coded": [], "ber_uncoded": [], "bler_coded": [], "bler_uncoded": [], "coding_gain_ber": [], "coding_gain_bler": []}

        ber_coded, ber_uncoded = [], []
        bler_coded, bler_uncoded = [], []
        coding_gain_ber, coding_gain_bler = [], []

        for snr_db in self.snr_range:
            # Generate test data
            num_blocks = max(1, self.num_bits // config["k"])
            info_bits_blocks = [torch.randint(0, 2, (config["k"],), dtype=torch.float32, device=self.device) for _ in range(num_blocks)]

            # Encode all blocks (use forward method)
            try:
                coded_blocks = [encoder(block) for block in info_bits_blocks]
                coded_bits = torch.cat(coded_blocks) if coded_blocks else torch.tensor([], dtype=torch.float32)
                info_bits = torch.cat(info_bits_blocks)
            except Exception:
                # Handle encoding failures
                ber_coded.append(1.0)
                ber_uncoded.append(1.0)
                bler_coded.append(1.0)
                bler_uncoded.append(1.0)
                coding_gain_ber.append(0.0)
                coding_gain_bler.append(0.0)
                continue

            if len(coded_bits) == 0:
                ber_coded.append(1.0)
                ber_uncoded.append(1.0)
                bler_coded.append(1.0)
                bler_uncoded.append(1.0)
                coding_gain_ber.append(0.0)
                coding_gain_bler.append(0.0)
                continue

            # BPSK modulation and AWGN channel
            coded_symbols = 2 * coded_bits.float() - 1
            uncoded_symbols = 2 * info_bits.float() - 1

            # Add noise
            snr_linear = 10 ** (snr_db / 10)
            noise_power = 1 / snr_linear
            noise_std = torch.sqrt(torch.tensor(noise_power / 2, device=self.device))

            coded_received = coded_symbols + noise_std * torch.randn_like(coded_symbols)
            uncoded_received = uncoded_symbols + noise_std * torch.randn_like(uncoded_symbols)

            # Hard decision
            coded_hard = (coded_received > 0).int()
            uncoded_hard = (uncoded_received > 0).int()

            # Decode coded transmission (use forward method)
            coded_hard_blocks = coded_hard.reshape(-1, config["n"])
            decoded_blocks = []

            for block in coded_hard_blocks:
                try:
                    decoded_blocks.append(decoder(block))
                except Exception:
                    # Use all-zeros for failed decoding
                    decoded_blocks.append(torch.zeros(config["k"], dtype=torch.float32, device=self.device))

            decoded_bits = torch.cat(decoded_blocks) if decoded_blocks else torch.tensor([], dtype=torch.float32)

            # Calculate BER
            if len(decoded_bits) > 0 and len(info_bits) > 0:
                min_len = min(len(info_bits), len(decoded_bits))
                ber_c = StandardMetrics.bit_error_rate(info_bits[:min_len], decoded_bits[:min_len])
                ber_coded.append(ber_c)
            else:
                ber_coded.append(1.0)

            # Uncoded BER
            if len(uncoded_hard) > 0 and len(info_bits) > 0:
                min_len = min(len(info_bits), len(uncoded_hard))
                ber_u = StandardMetrics.bit_error_rate(info_bits[:min_len], uncoded_hard[:min_len])
                ber_uncoded.append(ber_u)
            else:
                ber_uncoded.append(1.0)

            # Calculate BLER
            info_blocks = info_bits.reshape(-1, config["k"])
            decoded_blocks_tensor = decoded_bits.reshape(-1, config["k"]) if len(decoded_bits) > 0 else torch.zeros_like(info_blocks)
            uncoded_blocks = uncoded_hard.reshape(-1, config["k"]) if len(uncoded_hard) >= len(info_bits) else torch.zeros_like(info_blocks)

            # Block errors
            block_errors_coded = ~torch.all(info_blocks == decoded_blocks_tensor, dim=1)
            block_errors_uncoded = ~torch.all(info_blocks == uncoded_blocks, dim=1)

            bler_c = torch.mean(block_errors_coded.float()).item()
            bler_u = torch.mean(block_errors_uncoded.float()).item()
            bler_coded.append(bler_c)
            bler_uncoded.append(bler_u)

            # Coding gains
            gain_ber = 10 * torch.log10(torch.tensor(ber_u / ber_c)).item() if ber_c > 0 else float("inf")
            gain_bler = 10 * torch.log10(torch.tensor(bler_u / bler_c)).item() if bler_c > 0 else float("inf")
            coding_gain_ber.append(gain_ber)
            coding_gain_bler.append(gain_bler)

        return {"success": True, "ber_coded": ber_coded, "ber_uncoded": ber_uncoded, "bler_coded": bler_coded, "bler_uncoded": bler_uncoded, "coding_gain_ber": coding_gain_ber, "coding_gain_bler": coding_gain_bler}

    def _evaluate_complexity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate computational complexity."""
        if not self.evaluate_complexity:
            return {"success": False, "reason": "Complexity evaluation disabled"}

        encoder_class = config["encoder"]
        decoder_class = config["decoder"]

        try:
            encoder = encoder_class(**config["params"])
            decoder = decoder_class(encoder)
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Measure encoding complexity
        info_bits = torch.randint(0, 2, (config["k"],), dtype=torch.float32, device=self.device)

        # Warm up
        for _ in range(10):
            try:
                _ = encoder(info_bits)
            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                # Skip failed warm-up attempts
                pass

        # Measure encoding time
        encode_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            try:
                _ = encoder(info_bits)
                end_time = time.perf_counter()
                encode_times.append(end_time - start_time)
            except Exception:
                encode_times.append(float("inf"))

        # Measure decoding complexity
        try:
            codeword = encoder(info_bits)
        except Exception:
            return {"success": False, "error": "Failed to generate codeword for complexity testing"}

        # Warm up
        for _ in range(10):
            try:
                _ = decoder(codeword)
            except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                # Skip failed warm-up attempts
                pass

        # Measure decoding time
        decode_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            try:
                _ = decoder(codeword)
                end_time = time.perf_counter()
                decode_times.append(end_time - start_time)
            except Exception:
                decode_times.append(float("inf"))

        return {
            "success": True,
            "avg_encode_time": np.mean([t for t in encode_times if np.isfinite(t)]) if encode_times else float("inf"),
            "avg_decode_time": np.mean([t for t in decode_times if np.isfinite(t)]) if decode_times else float("inf"),
            "encode_time_std": np.std([t for t in encode_times if np.isfinite(t)]) if encode_times else 0,
            "decode_time_std": np.std([t for t in decode_times if np.isfinite(t)]) if decode_times else 0,
        }

    def _evaluate_throughput(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate encoding/decoding throughput."""
        if not self.evaluate_throughput:
            return {"success": False, "reason": "Throughput evaluation disabled"}

        encoder_class = config["encoder"]
        decoder_class = config["decoder"]

        try:
            encoder = encoder_class(**config["params"])
            decoder = decoder_class(encoder)
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Test different payload sizes
        payload_sizes = [100, 1000, 10000]
        throughput_results = {}

        for payload_size in payload_sizes:
            num_blocks = max(1, payload_size // config["k"])
            total_info_bits = num_blocks * config["k"]

            # Generate test data
            info_blocks = [torch.randint(0, 2, (config["k"],), dtype=torch.float32, device=self.device) for _ in range(num_blocks)]

            # Measure encoding throughput (use forward method)
            start_time = time.perf_counter()
            encoded_blocks = []
            for block in info_blocks:
                try:
                    encoded_blocks.append(encoder(block))
                except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                    # Skip failed encoding attempts for throughput measurement
                    pass
            encode_time = time.perf_counter() - start_time

            encode_throughput = total_info_bits / encode_time if encode_time > 0 else 0

            # Measure decoding throughput (use forward method)
            if encoded_blocks:
                start_time = time.perf_counter()
                for block in encoded_blocks:
                    try:
                        _ = decoder(block)
                    except (RuntimeError, ValueError, TypeError, AttributeError, IndexError):
                        # Skip failed decoding attempts for throughput measurement
                        pass
                decode_time = time.perf_counter() - start_time

                decode_throughput = total_info_bits / decode_time if decode_time > 0 else 0
            else:
                decode_throughput = 0

            throughput_results[payload_size] = {"encode_throughput": encode_throughput, "decode_throughput": decode_throughput, "total_info_bits": total_info_bits}

        return {"success": True, "throughput_results": throughput_results}

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run ECC performance benchmark."""
        results: Dict[str, Any] = {"success": True, "code_family": self.code_family, "configurations": [], "error_correction_results": {}, "ber_performance_results": {}, "complexity_results": {}, "throughput_results": {}, "summary": {}}

        print(f"Running ECC performance benchmark for {self.code_family.upper()} codes...")
        print(f"Configurations to test: {len(self.code_configs)}")

        for i, config in enumerate(self.code_configs):
            config_name = config["name"]
            print(f"Evaluating {config_name} ({i+1}/{len(self.code_configs)})...")

            # Store configuration info
            config_info = {"name": config_name, "n": config["n"], "k": config["k"], "d": config["d"], "t": config["t"], "code_rate": config["k"] / config["n"], "redundancy": config["n"] - config["k"]}
            results["configurations"].append(config_info)

            # Evaluate error correction performance
            ec_results = self._evaluate_error_correction_performance(config)
            results["error_correction_results"][config_name] = ec_results

            # Evaluate BER performance
            ber_results = self._evaluate_ber_performance(config)
            results["ber_performance_results"][config_name] = ber_results

            # Evaluate complexity
            complexity_results = self._evaluate_complexity(config)
            results["complexity_results"][config_name] = complexity_results

            # Evaluate throughput
            throughput_results = self._evaluate_throughput(config)
            results["throughput_results"][config_name] = throughput_results

        # Generate summary statistics
        successful_configs = [config for config in results["configurations"] if results["ber_performance_results"][config["name"]]["success"]]

        if successful_configs:
            # Best performing code (highest average coding gain)
            best_config = None
            best_gain = -float("inf")

            for config in successful_configs:
                config_name = config["name"]
                ber_results = results["ber_performance_results"][config_name]
                gains = [g for g in ber_results["coding_gain_ber"] if np.isfinite(g)]
                avg_gain = np.mean(gains) if gains else 0

                if avg_gain > best_gain:
                    best_gain = avg_gain
                    best_config = config

            results["summary"] = {
                "total_configurations": len(self.code_configs),
                "successful_configurations": len(successful_configs),
                "best_performing_code": best_config["name"] if best_config else None,
                "best_average_coding_gain": best_gain,
                "code_rates_tested": [c["code_rate"] for c in successful_configs],
                "block_lengths_tested": [c["n"] for c in successful_configs],
            }

        return results


@register_benchmark("ecc_comparison")
class ECCComparisonBenchmark(CommunicationBenchmark):
    """Benchmark for comparing different ECC families side-by-side."""

    def __init__(self, **kwargs):
        """Initialize ECC comparison benchmark."""
        super().__init__(name="ECC Family Comparison", description="Side-by-side comparison of different error correction code families")

    def setup(self, **kwargs):
        """Setup benchmark parameters."""
        super().setup(**kwargs)
        self.num_bits = kwargs.get("num_bits", 5000)
        self.families_to_compare = kwargs.get("families", ["hamming", "bch", "golay", "repetition"])

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run ECC family comparison benchmark."""
        results = {"success": True, "families_compared": self.families_to_compare, "family_results": {}, "comparison_summary": {}}

        print(f"Running ECC family comparison for: {', '.join(self.families_to_compare)}")

        # Run individual family benchmarks
        for family in self.families_to_compare:
            print(f"Evaluating {family.upper()} family...")

            # Create and run family benchmark
            family_benchmark = ECCPerformanceBenchmark(code_family=family)
            family_benchmark.setup(snr_range=self.snr_range, num_bits=self.num_bits, num_trials=50, max_errors=5, device=self.device)

            family_result = family_benchmark.run()
            results["family_results"][family] = family_result

        # Generate comparison summary
        best_families = {}
        metrics = ["coding_gain_ber", "coding_gain_bler"]

        for metric in metrics:
            best_gain = -float("inf")
            best_family = None

            for family, family_result in results["family_results"].items():
                if not family_result["success"]:
                    continue

                # Find best performing code in this family
                for config_name, ber_results in family_result["ber_performance_results"].items():
                    if not ber_results["success"]:
                        continue

                    gains = [g for g in ber_results[metric] if np.isfinite(g)]
                    avg_gain = np.mean(gains) if gains else 0

                    if avg_gain > best_gain:
                        best_gain = avg_gain
                        best_family = family

            best_families[metric] = {"family": best_family, "gain": best_gain}

        results["comparison_summary"] = {"best_for_ber_gain": best_families.get("coding_gain_ber", {}), "best_for_bler_gain": best_families.get("coding_gain_bler", {}), "families_evaluated": len([f for f in results["family_results"].values() if f["success"]])}

        return results
