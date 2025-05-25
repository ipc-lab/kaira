"""Standard metrics for benchmarking communication systems."""

from typing import Any, Dict, Union

import numpy as np
import torch
from scipy import stats


class StandardMetrics:
    """Collection of standard metrics for communication system evaluation."""

    @staticmethod
    def bit_error_rate(transmitted: Union[np.ndarray, torch.Tensor], received: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate Bit Error Rate (BER)."""
        if isinstance(transmitted, torch.Tensor):
            transmitted = transmitted.detach().cpu().numpy()
        if isinstance(received, torch.Tensor):
            received = received.detach().cpu().numpy()

        errors = np.sum(transmitted != received)
        total_bits = transmitted.size
        return float(errors / total_bits)

    @staticmethod
    def block_error_rate(transmitted: Union[np.ndarray, torch.Tensor], received: Union[np.ndarray, torch.Tensor], block_size: int) -> float:
        """Calculate Block Error Rate (BLER)."""
        if isinstance(transmitted, torch.Tensor):
            transmitted = transmitted.detach().cpu().numpy()
        if isinstance(received, torch.Tensor):
            received = received.detach().cpu().numpy()

        # Reshape into blocks
        n_blocks = len(transmitted) // block_size
        transmitted_blocks = transmitted[: n_blocks * block_size].reshape(-1, block_size)
        received_blocks = received[: n_blocks * block_size].reshape(-1, block_size)

        # Count block errors
        block_errors = np.sum(np.any(transmitted_blocks != received_blocks, axis=1))
        return float(block_errors / n_blocks)

    @staticmethod
    def signal_to_noise_ratio(signal: Union[np.ndarray, torch.Tensor], noise: Union[np.ndarray, torch.Tensor]) -> float:
        """Calculate Signal-to-Noise Ratio (SNR) in dB."""
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()
        if isinstance(noise, torch.Tensor):
            noise = noise.detach().cpu().numpy()

        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)

        if noise_power == 0:
            return float("inf")

        snr_linear = signal_power / noise_power
        return float(10 * np.log10(snr_linear))

    @staticmethod
    def mutual_information(x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor], bins: int = 50) -> float:
        """Estimate mutual information between two variables."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # Flatten arrays
        x = x.flatten()
        y = y.flatten()

        # Calculate histograms
        xy = np.histogram2d(x, y, bins=bins)[0]
        x_hist = np.histogram(x, bins=bins)[0]
        y_hist = np.histogram(y, bins=bins)[0]

        # Normalize to get probabilities
        xy = xy / np.sum(xy)
        x_hist = x_hist / np.sum(x_hist)
        y_hist = y_hist / np.sum(y_hist)

        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if xy[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                    mi += xy[i, j] * np.log2(xy[i, j] / (x_hist[i] * y_hist[j]))

        return float(mi)

    @staticmethod
    def throughput(bits_transmitted: int, time_elapsed: float) -> float:
        """Calculate throughput in bits per second."""
        if time_elapsed <= 0:
            return 0.0
        return float(bits_transmitted / time_elapsed)

    @staticmethod
    def latency_statistics(latencies: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if isinstance(latencies, torch.Tensor):
            latencies = latencies.detach().cpu().numpy()

        return {
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "std_latency": float(np.std(latencies)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "p99_latency": float(np.percentile(latencies, 99)),
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
        capacity = bandwidth * np.log2(1 + snr_linear)
        return float(capacity)

    @staticmethod
    def confidence_interval(data: Union[np.ndarray, torch.Tensor], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for data."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        mean = np.mean(data)
        sem = stats.sem(data)
        interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

        return float(mean - interval), float(mean + interval)
