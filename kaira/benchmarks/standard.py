"""Standard benchmark implementations for communication systems."""

import time
from typing import Any, Dict, Optional

import torch

from .base import CommunicationBenchmark
from .metrics import StandardMetrics
from .registry import register_benchmark


@register_benchmark("channel_capacity")
class ChannelCapacityBenchmark(CommunicationBenchmark):
    """Benchmark for channel capacity calculations."""

    def __init__(self, channel_type: str = "awgn", **kwargs):
        """Initialize channel capacity benchmark.

        Args:
            channel_type: Type of channel ('awgn' for AWGN channel)
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"Channel Capacity ({channel_type.upper()})", description=f"Benchmark channel capacity for {channel_type} channel")
        self.channel_type = channel_type

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Benchmark configuration including bandwidth
        """
        super().setup(**kwargs)
        self.bandwidth = kwargs.get("bandwidth", 1.0)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run channel capacity benchmark."""
        capacities = []

        for snr_db in self.snr_range:
            capacity = StandardMetrics.channel_capacity(snr_db, self.bandwidth)
            capacities.append(capacity)

        return {"success": True, "snr_range": self.snr_range, "capacities": capacities, "max_capacity": max(capacities), "min_capacity": min(capacities), "channel_type": self.channel_type, "bandwidth": self.bandwidth}


@register_benchmark("ber_simulation")
class BERSimulationBenchmark(CommunicationBenchmark):
    """Benchmark for Bit Error Rate simulation."""

    def __init__(self, modulation: str = "bpsk", **kwargs):
        """Initialize BER simulation benchmark.

        Args:
            modulation: Modulation scheme ('bpsk')
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"BER Simulation ({modulation.upper()})", description=f"Benchmark BER performance for {modulation} modulation")
        self.modulation = modulation

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including num_bits and batch_size
        """
        super().setup(**kwargs)
        self.num_bits = kwargs.get("num_bits", 100000)
        self.batch_size = kwargs.get("batch_size", 10000)

    def _generate_bits(self, num_bits: int) -> torch.Tensor:
        """Generate random bits."""
        return torch.randint(0, 2, (num_bits,), device=self.device)

    def _modulate_bpsk(self, bits: torch.Tensor) -> torch.Tensor:
        """BPSK modulation: maps bits 0->-1, 1->+1."""
        return 2 * bits.float() - 1

    def _add_awgn(self, symbols: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Additive White Gaussian Noise to symbols.

        Args:
            symbols: Input symbols to add noise to
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            Noisy symbols
        """
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear
        noise = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * (torch.randn(len(symbols), device=self.device) + 1j * torch.randn(len(symbols), device=self.device))
        return symbols + noise.real  # Take real part for BPSK

    def _demodulate_bpsk(self, received: torch.Tensor) -> torch.Tensor:
        """BPSK demodulation using threshold detection.

        Args:
            received: Received noisy symbols

        Returns:
            Decoded bits (0 or 1)
        """
        return (received > 0).int()

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run BER simulation benchmark."""
        ber_results = []
        theoretical_ber = []

        for snr_db in self.snr_range:
            # Generate bits
            bits = self._generate_bits(self.num_bits)

            # Modulate
            if self.modulation.lower() == "bpsk":
                symbols = self._modulate_bpsk(bits)
            else:
                raise NotImplementedError(f"Modulation {self.modulation} not implemented")

            # Add noise
            received = self._add_awgn(symbols, snr_db)

            # Demodulate
            if self.modulation.lower() == "bpsk":
                decoded_bits = self._demodulate_bpsk(received)
            else:
                raise NotImplementedError(f"Demodulation {self.modulation} not implemented")

            # Calculate BER
            ber = StandardMetrics.bit_error_rate(bits, decoded_bits)
            ber_results.append(ber)

            # Theoretical BER for BPSK
            if self.modulation.lower() == "bpsk":
                snr_linear = 10 ** (snr_db / 10)
                theo_ber = 0.5 * torch.special.erfc(torch.sqrt(torch.tensor(snr_linear, device=self.device))).item()
                theoretical_ber.append(theo_ber)

        return {"success": True, "snr_range": self.snr_range, "ber_simulated": ber_results, "ber_theoretical": theoretical_ber, "modulation": self.modulation, "num_bits": self.num_bits, "rmse": torch.sqrt(torch.mean((torch.tensor(ber_results) - torch.tensor(theoretical_ber)) ** 2)).item()}


@register_benchmark("throughput_test")
class ThroughputBenchmark(CommunicationBenchmark):
    """Benchmark for system throughput."""

    def __init__(self, **kwargs):
        """Initialize throughput benchmark.

        Args:
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name="Throughput Test", description="Benchmark system throughput performance")

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including payload_sizes and num_trials
        """
        super().setup(**kwargs)
        self.payload_sizes = kwargs.get("payload_sizes", [100, 1000, 10000, 100000])
        self.num_trials = kwargs.get("num_trials", 10)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run throughput benchmark."""
        throughput_results = {}

        for payload_size in self.payload_sizes:
            throughputs = []

            for _ in range(self.num_trials):
                # Generate payload
                payload = torch.randint(0, 2, (payload_size,), device=self.device)

                # Measure transmission time
                start_time = time.time()

                # Simulate processing (encoding, modulation, etc.)
                processed = payload.clone()
                kernel = torch.tensor([1, 1], dtype=torch.float32, device=self.device)
                for _ in range(10):  # Simulate some processing
                    processed = torch.nn.functional.conv1d(processed.float().unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze()[:payload_size].int()

                end_time = time.time()

                # Calculate throughput
                transmission_time = end_time - start_time
                throughput = StandardMetrics.throughput(payload_size, transmission_time)
                throughputs.append(throughput)

            throughput_results[payload_size] = {"mean": torch.tensor(throughputs).mean().item(), "std": torch.tensor(throughputs).std().item(), "min": torch.tensor(throughputs).min().item(), "max": torch.tensor(throughputs).max().item()}

        return {"success": True, "payload_sizes": self.payload_sizes, "throughput_results": throughput_results, "peak_throughput": max(result["max"] for result in throughput_results.values())}


@register_benchmark("latency_test")
class LatencyBenchmark(CommunicationBenchmark):
    """Benchmark for system latency."""

    def __init__(self, **kwargs):
        """Initialize latency benchmark.

        Args:
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name="Latency Test", description="Benchmark system latency performance")

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including num_measurements and packet_size
        """
        super().setup(**kwargs)
        self.num_measurements = kwargs.get("num_measurements", 1000)
        self.packet_size = kwargs.get("packet_size", 1000)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run latency benchmark."""
        latencies = []

        for _ in range(self.num_measurements):
            # Generate packet
            packet = torch.randint(0, 2, (self.packet_size,), device=self.device)

            # Measure processing latency
            start_time = time.perf_counter()

            # Simulate packet processing
            processed = packet.clone()
            processed = torch.roll(processed, 1)  # Simulate minimal processing

            end_time = time.perf_counter()

            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

        latency_stats = StandardMetrics.latency_statistics(torch.tensor(latencies))

        return {"success": True, "num_measurements": self.num_measurements, "packet_size": self.packet_size, **latency_stats}


@register_benchmark("model_complexity")
class ModelComplexityBenchmark(CommunicationBenchmark):
    """Benchmark for model computational complexity."""

    def __init__(self, model: Optional[torch.nn.Module] = None, **kwargs):
        """Initialize model complexity benchmark.

        Args:
            model: PyTorch model to analyze (creates default if None)
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name="Model Complexity", description="Benchmark model computational complexity")
        self.model = model

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including input_shape
        """
        super().setup(**kwargs)
        if self.model is None:
            # Create a simple test model
            self.model = torch.nn.Sequential(torch.nn.Linear(100, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

        self.input_shape = kwargs.get("input_shape", (100,))
        self.model.to(self.device)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run model complexity benchmark."""
        if self.model is None:
            raise ValueError("Model must be set before running benchmark")

        # Calculate model complexity
        complexity = StandardMetrics.computational_complexity(self.model, self.input_shape)

        # Measure inference time
        batch_size = kwargs.get("batch_size", 1000)
        num_trials = kwargs.get("num_trials", 100)

        with torch.no_grad():
            # Warm up
            dummy_input = torch.randn(10, *self.input_shape).to(self.device)
            for _ in range(10):
                _ = self.model(dummy_input)

            # Measure inference time
            inference_times = []
            test_input = torch.randn(batch_size, *self.input_shape).to(self.device)

            for _ in range(num_trials):
                start_time = time.perf_counter()
                _ = self.model(test_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                inference_times.append((end_time - start_time) * 1000)  # ms

        latency_stats = StandardMetrics.latency_statistics(torch.tensor(inference_times))

        return {"success": True, "model_complexity": complexity, "inference_latency_ms": latency_stats, "throughput_samples_per_second": batch_size / (latency_stats["mean_latency"] / 1000), "batch_size": batch_size, "device": str(self.device)}


@register_benchmark("qam_ber")
class QAMBERBenchmark(CommunicationBenchmark):
    """Benchmark for QAM modulation BER performance."""

    def __init__(self, constellation_size: int = 16, **kwargs):
        """Initialize QAM BER benchmark.

        Args:
            constellation_size: QAM constellation size (must be perfect square)
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"{constellation_size}-QAM BER", description=f"Benchmark BER performance for {constellation_size}-QAM modulation")
        self.constellation_size = constellation_size
        self.bits_per_symbol = int(torch.log2(torch.tensor(constellation_size)).item())

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including num_symbols and batch_size
        """
        super().setup(**kwargs)
        self.num_symbols = kwargs.get("num_symbols", 50000)
        self.batch_size = kwargs.get("batch_size", 10000)

        # Generate QAM constellation
        self._generate_constellation()

    def _generate_constellation(self):
        """Generate QAM constellation points.

        Creates a square QAM constellation with normalized average power.
        The constellation size must be a perfect square.

        Raises:
            ValueError: If constellation size is not a perfect square
        """
        sqrt_M = int(torch.sqrt(torch.tensor(self.constellation_size)).item())
        if sqrt_M**2 != self.constellation_size:
            raise ValueError("Constellation size must be a perfect square")

        # Create constellation
        real_levels = torch.arange(-sqrt_M + 1, sqrt_M, 2, dtype=torch.float32)
        imag_levels = torch.arange(-sqrt_M + 1, sqrt_M, 2, dtype=torch.float32)

        constellation = []
        for i in real_levels:
            for q in imag_levels:
                constellation.append(complex(i.item(), q.item()))

        self.constellation = torch.tensor(constellation, dtype=torch.complex64, device=self.device)

        # Normalize average power to 1
        avg_power = torch.mean(torch.abs(self.constellation) ** 2)
        self.constellation = self.constellation / torch.sqrt(avg_power)

    def _bits_to_symbols(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert bits to QAM symbols.

        Groups bits into symbols based on bits_per_symbol and maps them
        to constellation points.

        Args:
            bits: Input bit array

        Returns:
            Complex QAM symbols
        """
        # Reshape bits to groups
        bits_reshaped = bits[: len(bits) // self.bits_per_symbol * self.bits_per_symbol]
        bits_grouped = bits_reshaped.reshape(-1, self.bits_per_symbol)

        # Convert to decimal indices manually (more reliable than packbits)
        indices = []
        for bit_group in bits_grouped:
            decimal_val = 0
            for i, bit in enumerate(bit_group):
                decimal_val += bit.item() * (2 ** (self.bits_per_symbol - 1 - i))
            indices.append(decimal_val)
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)

        # Map to constellation
        return self.constellation[indices]

    def _symbols_to_bits(self, symbols: torch.Tensor) -> torch.Tensor:
        """Convert received symbols to bits using minimum distance decoding.

        Finds the closest constellation point for each received symbol
        and converts the symbol index back to bits.

        Args:
            symbols: Received complex symbols

        Returns:
            Decoded bit array
        """
        # Find closest constellation point for each symbol
        distances = torch.abs(symbols[:, None] - self.constellation[None, :])
        indices = torch.argmin(distances, dim=1)

        # Convert indices to bits manually
        bits = []
        for idx in indices:
            bit_array = []
            for i in range(self.bits_per_symbol):
                bit = (idx.item() >> (self.bits_per_symbol - 1 - i)) & 1
                bit_array.append(bit)
            bits.extend(bit_array)

        return torch.tensor(bits, dtype=torch.int32, device=self.device)

    def _add_awgn(self, symbols: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Additive White Gaussian Noise to complex symbols.

        Args:
            symbols: Complex input symbols
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            Noisy complex symbols
        """
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear

        noise_real = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(symbols), device=self.device)
        noise_imag = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(symbols), device=self.device)
        noise = noise_real + 1j * noise_imag

        return symbols + noise

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run QAM BER benchmark."""
        ber_results = []

        for snr_db in self.snr_range:
            # Generate random bits
            num_bits = self.num_symbols * self.bits_per_symbol
            bits = torch.randint(0, 2, (num_bits,), device=self.device)

            # Modulate to QAM symbols
            symbols = self._bits_to_symbols(bits)

            # Add AWGN
            received = self._add_awgn(symbols, snr_db)

            # Demodulate
            decoded_bits = self._symbols_to_bits(received)

            # Calculate BER
            # Ensure same length for comparison
            min_len = min(len(bits), len(decoded_bits))
            ber = StandardMetrics.bit_error_rate(bits[:min_len], decoded_bits[:min_len])
            ber_results.append(ber)

        return {"success": True, "snr_range": self.snr_range, "ber_results": ber_results, "constellation_size": self.constellation_size, "bits_per_symbol": self.bits_per_symbol, "num_symbols": self.num_symbols, "average_ber": torch.tensor(ber_results).mean().item()}


@register_benchmark("ofdm_performance")
class OFDMPerformanceBenchmark(CommunicationBenchmark):
    """Benchmark for OFDM system performance."""

    def __init__(self, num_subcarriers: int = 64, cp_length: int = 16, **kwargs):
        """Initialize OFDM performance benchmark.

        Args:
            num_subcarriers: Number of OFDM subcarriers
            cp_length: Cyclic prefix length
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"OFDM Performance (N={num_subcarriers})", description=f"Benchmark OFDM performance with {num_subcarriers} subcarriers")
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including num_symbols and modulation
        """
        super().setup(**kwargs)
        self.num_symbols = kwargs.get("num_symbols", 1000)
        self.modulation = kwargs.get("modulation", "qpsk")

        # QPSK constellation
        if self.modulation.lower() == "qpsk":
            self.constellation = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=torch.complex64, device=self.device) / torch.sqrt(torch.tensor(2.0, device=self.device))
            self.bits_per_symbol = 2
        else:
            raise NotImplementedError(f"Modulation {self.modulation} not implemented")

    def _generate_ofdm_symbol(self, data_bits: torch.Tensor) -> torch.Tensor:
        """Generate OFDM symbol from data bits.

        Modulates data bits, performs IFFT, and adds cyclic prefix.

        Args:
            data_bits: Input data bits

        Returns:
            Time-domain OFDM symbol with cyclic prefix
        """
        # Group bits for modulation
        bits_grouped = data_bits.reshape(-1, self.bits_per_symbol)

        # QPSK modulation - convert bits to indices manually
        indices: list[int] = []
        for bit_group in bits_grouped:
            decimal_val = 0
            for i, bit in enumerate(bit_group):
                decimal_val += bit.item() * (2 ** (self.bits_per_symbol - 1 - i))
            indices.append(decimal_val)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)

        modulated = self.constellation[indices_tensor % len(self.constellation)]

        # Pad or truncate to fit subcarriers
        if len(modulated) < self.num_subcarriers:
            modulated = torch.nn.functional.pad(modulated, (0, self.num_subcarriers - len(modulated)))
        elif len(modulated) > self.num_subcarriers:
            modulated = modulated[: self.num_subcarriers]

        # IFFT
        time_domain = torch.fft.ifft(modulated, self.num_subcarriers)

        # Add cyclic prefix
        cp = time_domain[-self.cp_length :]
        ofdm_symbol = torch.cat([cp, time_domain])

        return ofdm_symbol

    def _demodulate_ofdm_symbol(self, received_symbol: torch.Tensor) -> torch.Tensor:
        """Demodulate OFDM symbol to bits.

        Removes cyclic prefix, performs FFT, and demodulates subcarriers.

        Args:
            received_symbol: Received time-domain OFDM symbol

        Returns:
            Decoded data bits
        """
        # Remove cyclic prefix
        time_domain = received_symbol[self.cp_length :]

        # FFT
        freq_domain = torch.fft.fft(time_domain, self.num_subcarriers)

        # Demodulate QPSK (minimum distance)
        distances = torch.abs(freq_domain[:, None] - self.constellation[None, :])
        indices = torch.argmin(distances, dim=1)

        # Convert to bits
        bits = []
        for idx in indices:
            bit_array = []
            idx_val = idx.item()
            for i in range(self.bits_per_symbol):
                bit = (idx_val >> (self.bits_per_symbol - 1 - i)) & 1
                bit_array.append(bit)
            bits.extend(bit_array)

        return torch.tensor(bits, dtype=torch.int32, device=self.device)

    def _add_channel_effects(self, ofdm_symbol: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add channel effects including AWGN and optional multipath.

        Args:
            ofdm_symbol: Input OFDM symbol
            snr_db: Signal-to-noise ratio in decibels

        Returns:
            OFDM symbol with channel effects applied
        """
        # AWGN
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1 / snr_linear

        noise_real = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(ofdm_symbol), device=self.device)
        noise_imag = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(ofdm_symbol), device=self.device)
        noise = noise_real + 1j * noise_imag

        # Simple multipath (optional)
        multipath_enabled = False  # Can be enabled for more realistic simulation
        if multipath_enabled:
            # Simple 2-tap channel
            h = torch.tensor([1.0, 0.3 * torch.exp(1j * torch.tensor(torch.pi / 4, device=self.device))], dtype=torch.complex64, device=self.device)
            ofdm_symbol = torch.nn.functional.conv1d(ofdm_symbol.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0)).squeeze()

        return ofdm_symbol + noise

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run OFDM performance benchmark."""
        ber_results = []
        throughput_results = []

        for snr_db in self.snr_range:
            total_bits = 0
            total_errors = 0
            start_time = time.time()

            for _ in range(self.num_symbols):
                # Generate random data bits
                data_bits = torch.randint(0, 2, (self.num_subcarriers * self.bits_per_symbol,), device=self.device)

                # Generate OFDM symbol
                ofdm_symbol = self._generate_ofdm_symbol(data_bits)

                # Add channel effects
                received = self._add_channel_effects(ofdm_symbol, snr_db)

                # Demodulate
                decoded_bits = self._demodulate_ofdm_symbol(received)

                # Count errors
                min_len = min(len(data_bits), len(decoded_bits))
                errors = torch.sum(data_bits[:min_len] != decoded_bits[:min_len]).item()
                total_errors += errors
                total_bits += min_len

            end_time = time.time()

            # Calculate metrics
            ber = total_errors / total_bits if total_bits > 0 else 0
            ber_results.append(ber)

            # Calculate throughput
            processing_time = end_time - start_time
            throughput = total_bits / processing_time
            throughput_results.append(throughput)

        return {
            "success": True,
            "snr_range": self.snr_range,
            "ber_results": ber_results,
            "throughput_bps": throughput_results,
            "num_subcarriers": self.num_subcarriers,
            "cp_length": self.cp_length,
            "modulation": self.modulation,
            "num_symbols": self.num_symbols,
            "spectral_efficiency": self.bits_per_symbol,
            "average_ber": torch.tensor(ber_results).mean().item(),
            "peak_throughput": max(throughput_results),
        }


@register_benchmark("channel_coding")
class ChannelCodingBenchmark(CommunicationBenchmark):
    """Benchmark for channel coding performance."""

    def __init__(self, code_type: str = "repetition", code_rate: float = 0.5, **kwargs):
        """Initialize channel coding benchmark.

        Args:
            code_type: Type of channel code ('repetition')
            code_rate: Code rate (0 < rate <= 1)
            **kwargs: Additional benchmark arguments
        """
        super().__init__(name=f"Channel Coding ({code_type}, R={code_rate})", description=f"Benchmark {code_type} coding with rate {code_rate}")
        self.code_type = code_type
        self.code_rate = code_rate

    def setup(self, **kwargs):
        """Setup benchmark parameters.

        Args:
            **kwargs: Configuration including num_bits
        """
        super().setup(**kwargs)
        self.num_bits = kwargs.get("num_bits", 10000)

        if self.code_type == "repetition":
            self.repetition_factor = int(1 / self.code_rate)
        else:
            raise NotImplementedError(f"Code type {self.code_type} not implemented")

    def _encode_repetition(self, bits: torch.Tensor) -> torch.Tensor:
        """Repetition encoder that repeats each bit multiple times.

        Args:
            bits: Input information bits

        Returns:
            Encoded bits with repetition
        """
        return torch.repeat_interleave(bits, self.repetition_factor)

    def _decode_repetition(self, received: torch.Tensor) -> torch.Tensor:
        """Repetition decoder using majority voting.

        Groups received bits and uses majority vote to decide
        the most likely transmitted bit.

        Args:
            received: Received encoded bits

        Returns:
            Decoded information bits
        """
        # Group received bits
        received_grouped = received.reshape(-1, self.repetition_factor)

        # Majority vote
        decoded = (torch.sum(received_grouped, dim=1) > self.repetition_factor / 2).int()

        return decoded

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run channel coding benchmark."""
        ber_uncoded = []
        ber_coded = []
        coding_gain = []

        for snr_db in self.snr_range:
            # Generate random bits
            info_bits = torch.randint(0, 2, (self.num_bits,), device=self.device)

            # Uncoded transmission
            uncoded_symbols = 2 * info_bits.float() - 1  # BPSK
            snr_linear = 10 ** (snr_db / 10)
            noise_power = 1 / snr_linear
            noise = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(uncoded_symbols), device=self.device)
            uncoded_received = uncoded_symbols + noise
            uncoded_decoded = (uncoded_received > 0).int()
            ber_unc = StandardMetrics.bit_error_rate(info_bits, uncoded_decoded)
            ber_uncoded.append(ber_unc)

            # Coded transmission
            if self.code_type == "repetition":
                coded_bits = self._encode_repetition(info_bits)
            else:
                raise NotImplementedError(f"Code type {self.code_type} not implemented")

            # Transmit coded bits
            coded_symbols = 2 * coded_bits.float() - 1  # BPSK
            coded_noise = torch.sqrt(torch.tensor(noise_power / 2, device=self.device)) * torch.randn(len(coded_symbols), device=self.device)
            coded_received = coded_symbols + coded_noise

            # Hard decision
            coded_hard = (coded_received > 0).int()

            # Decode
            if self.code_type == "repetition":
                coded_decoded = self._decode_repetition(coded_hard)
            else:
                raise NotImplementedError(f"Code type {self.code_type} not implemented")

            # Calculate BER
            min_len = min(len(info_bits), len(coded_decoded))
            ber_cod = StandardMetrics.bit_error_rate(info_bits[:min_len], coded_decoded[:min_len])
            ber_coded.append(ber_cod)

            # Calculate coding gain
            gain = 10 * torch.log10(torch.tensor(ber_unc / ber_cod)).item() if ber_cod > 0 else float("inf")
            coding_gain.append(gain)

        finite_gains = [g for g in coding_gain if torch.isfinite(torch.tensor(g))]
        avg_gain = torch.tensor(finite_gains).mean().item() if finite_gains else 0.0

        return {"success": True, "snr_range": self.snr_range, "ber_uncoded": ber_uncoded, "ber_coded": ber_coded, "coding_gain_db": coding_gain, "code_type": self.code_type, "code_rate": self.code_rate, "average_coding_gain": avg_gain}
