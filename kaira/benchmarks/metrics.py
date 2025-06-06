"""Standard metrics for benchmarking communication systems."""

from typing import Any, Dict, Union

import torch
from scipy import stats


class StandardMetrics:
    """Collection of standard metrics for communication system evaluation."""

    @staticmethod
    def bit_error_rate(transmitted: Union[torch.Tensor, torch.Tensor], received: Union[torch.Tensor, torch.Tensor]) -> float:
        """Calculate Bit Error Rate (BER)."""
        if not isinstance(transmitted, torch.Tensor):
            transmitted = torch.tensor(transmitted)
        if not isinstance(received, torch.Tensor):
            received = torch.tensor(received)

        errors = torch.sum(transmitted != received)
        total_bits = transmitted.numel()
        return float(errors / total_bits)

    @staticmethod
    def block_error_rate(transmitted: Union[torch.Tensor, torch.Tensor], received: Union[torch.Tensor, torch.Tensor], block_size: int) -> float:
        """Calculate Block Error Rate (BLER)."""
        if not isinstance(transmitted, torch.Tensor):
            transmitted = torch.tensor(transmitted)
        if not isinstance(received, torch.Tensor):
            received = torch.tensor(received)

        # Reshape into blocks
        n_blocks = len(transmitted) // block_size
        transmitted_blocks = transmitted[: n_blocks * block_size].reshape(-1, block_size)
        received_blocks = received[: n_blocks * block_size].reshape(-1, block_size)

        # Count block errors
        block_errors = torch.sum(torch.any(transmitted_blocks != received_blocks, dim=1))
        return float(block_errors / n_blocks)

    @staticmethod
    def signal_to_noise_ratio(signal: Union[torch.Tensor, torch.Tensor], noise: Union[torch.Tensor, torch.Tensor]) -> float:
        """Calculate Signal-to-Noise Ratio (SNR) in dB."""
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal)
        if not isinstance(noise, torch.Tensor):
            noise = torch.tensor(noise)

        signal_power = torch.mean(torch.abs(signal) ** 2)
        noise_power = torch.mean(torch.abs(noise) ** 2)

        if noise_power == 0:
            return float("inf")

        snr_linear = signal_power / noise_power
        return float(10 * torch.log10(snr_linear))

    @staticmethod
    def mutual_information(x: Union[torch.Tensor, torch.Tensor], y: Union[torch.Tensor, torch.Tensor], bins: int = 50) -> float:
        """Estimate mutual information between two variables."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        # Flatten tensors
        x = x.flatten()
        y = y.flatten()

        # Calculate histograms using torch operations
        # Get min/max values for binning
        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)

        # Create bin edges
        x_edges = torch.linspace(x_min, x_max, bins + 1)
        y_edges = torch.linspace(y_min, y_max, bins + 1)

        # Create 2D histogram manually
        xy = torch.zeros(bins, bins, dtype=torch.float32)
        x_hist = torch.zeros(bins, dtype=torch.float32)
        y_hist = torch.zeros(bins, dtype=torch.float32)

        # Compute bin indices
        x_indices = torch.searchsorted(x_edges[1:], x, right=False)
        y_indices = torch.searchsorted(y_edges[1:], y, right=False)

        # Clamp indices to valid range
        x_indices = torch.clamp(x_indices, 0, bins - 1)
        y_indices = torch.clamp(y_indices, 0, bins - 1)

        # Fill histograms
        for i in range(len(x)):
            xy[x_indices[i], y_indices[i]] += 1
            x_hist[x_indices[i]] += 1
            y_hist[y_indices[i]] += 1

        xy = xy / torch.sum(xy)
        x_hist = x_hist / torch.sum(x_hist)
        y_hist = y_hist / torch.sum(y_hist)

        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if xy[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                    mi += xy[i, j] * torch.log2(xy[i, j] / (x_hist[i] * y_hist[j]))

        return float(mi)

    @staticmethod
    def throughput(bits_transmitted: int, time_elapsed: float) -> float:
        """Calculate throughput in bits per second."""
        if time_elapsed <= 0:
            return 0.0
        return float(bits_transmitted / time_elapsed)

    @staticmethod
    def latency_statistics(latencies: Union[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not isinstance(latencies, torch.Tensor):
            latencies = torch.tensor(latencies)

        return {
            "mean_latency": float(torch.mean(latencies)),
            "median_latency": float(torch.median(latencies)),
            "min_latency": float(torch.min(latencies)),
            "max_latency": float(torch.max(latencies)),
            "std_latency": float(torch.std(latencies)),
            "p95_latency": float(torch.quantile(latencies, 0.95)),
            "p99_latency": float(torch.quantile(latencies, 0.99)),
        }

    @staticmethod
    def computational_complexity(model: torch.nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """Estimate computational complexity of a PyTorch model."""
        try:
            from ptflops import get_model_complexity_info

            macs, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=False, verbose=False)
            return {"macs": macs, "parameters": params, "model_size_mb": params * 4 / (1024**2)}  # Assuming float32
        except ImportError:
            # Fallback to parameter counting only
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {"total_parameters": total_params, "trainable_parameters": trainable_params, "model_size_mb": total_params * 4 / (1024**2)}

    @staticmethod
    def channel_capacity(snr_db: float, bandwidth: float = 1.0) -> float:
        """Calculate Shannon channel capacity."""
        snr_linear = 10 ** (snr_db / 10)
        capacity = bandwidth * torch.log2(torch.tensor(1 + snr_linear))
        return float(capacity)

    @staticmethod
    def confidence_interval(data: Union[torch.Tensor, torch.Tensor], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for data."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        # Convert to numpy for scipy stats
        data_np = data.detach().cpu().numpy()
        mean = torch.mean(data)
        sem = torch.std(data, correction=1) / torch.sqrt(torch.tensor(len(data), dtype=torch.float))
        interval = sem * stats.t.ppf((1 + confidence) / 2, len(data_np) - 1)

        return float(mean - interval), float(mean + interval)
