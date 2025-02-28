"""Utility functions for metrics."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .base import BaseMetric


def compute_multiple_metrics(
    metrics: Dict[str, BaseMetric], preds: Tensor, targets: Tensor
) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]:
    """Compute multiple metrics at once.

    Args:
        metrics (Dict[str, BaseMetric]): Dictionary of metric names to metric instances
        preds (Tensor): Predicted values
        targets (Tensor): Target values

    Returns:
        Dict[str, Union[Tensor, Tuple[Tensor, Tensor]]]: Dictionary of metric results
    """
    results = {}
    for name, metric in metrics.items():
        if hasattr(metric, "compute_with_stats"):
            results[name] = metric.compute_with_stats(preds, targets)
        else:
            results[name] = metric(preds, targets)
    return results


def format_metric_results(results: Dict[str, Any]) -> str:
    """Format metric results as a string.

    Args:
        results (Dict[str, Any]): Dictionary of metric results

    Returns:
        str: Formatted string representation of metrics
    """
    lines = []
    for name, value in results.items():
        if isinstance(value, tuple) and len(value) == 2:
            mean, std = value
            lines.append(f"{name}: {mean:.4f} ± {std:.4f}")
        else:
            lines.append(f"{name}: {value:.4f}")
    return ", ".join(lines)


def save_metrics_to_json(results: Dict[str, Any], filename: str) -> None:
    """Save metrics results to a JSON file.

    Args:
        results (Dict[str, Any]): Dictionary of metric results
        filename (str): Filename to save to
    """
    # Convert tensors to Python types
    serializable_results = {}
    for name, value in results.items():
        if isinstance(value, tuple) and len(value) == 2:
            mean, std = value
            serializable_results[name] = {
                "mean": float(
                    mean.detach().cpu().numpy() if isinstance(mean, torch.Tensor) else mean
                ),
                "std": float(std.detach().cpu().numpy() if isinstance(std, torch.Tensor) else std),
            }
        elif isinstance(value, torch.Tensor):
            serializable_results[name] = float(value.detach().cpu().numpy())
        else:
            serializable_results[name] = value

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)


def visualize_metrics_comparison(
    results_list: List[Dict[str, Any]],
    labels: List[str],
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = "Metrics Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Visualize a comparison of metrics across multiple experiments.

    Args:
        results_list (List[Dict[str, Any]]): List of metric result dictionaries
        labels (List[str]): List of labels for each result set
        figsize (Tuple[int, int]): Figure size
        title (Optional[str]): Plot title
        save_path (Optional[str]): Path to save the figure
    """
    if not results_list:
        raise ValueError("No results provided for visualization")

    # Extract metrics common to all result sets
    common_metrics = set(results_list[0].keys())
    for results in results_list[1:]:
        common_metrics = common_metrics.intersection(results.keys())

    plt.figure(figsize=figsize)
    metric_count = len(common_metrics)

    # Set up bar positions
    bar_width = 0.8 / len(results_list)
    bar_indices = np.arange(metric_count)

    for i, (results, label) in enumerate(zip(results_list, labels)):
        means = []
        errors = []

        for j, metric in enumerate(common_metrics):
            value = results[metric]
            if isinstance(value, tuple) and len(value) == 2:
                mean, std = value
                means.append(
                    float(mean.detach().cpu().numpy() if isinstance(mean, torch.Tensor) else mean)
                )
                errors.append(
                    float(std.detach().cpu().numpy() if isinstance(std, torch.Tensor) else std)
                )
            else:
                means.append(
                    float(
                        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    )
                )
                errors.append(0)

        # Plot bars with error bars
        plt.bar(
            bar_indices + i * bar_width - (len(results_list) - 1) * bar_width / 2,
            means,
            width=bar_width,
            yerr=errors,
            label=label,
            capsize=5,
        )

    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(bar_indices, list(common_metrics), rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def benchmark_metrics(
    metrics: Dict[str, BaseMetric], preds: Tensor, targets: Tensor, repeat: int = 10
) -> Dict[str, Dict[str, float]]:
    """Benchmark execution time of metrics.

    Args:
        metrics (Dict[str, BaseMetric]): Dictionary of metrics to benchmark
        preds (Tensor): Prediction tensor
        targets (Tensor): Target tensor
        repeat (int): Number of repetitions for timing

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing timing results
    """
    import time

    results = {}

    for name, metric in metrics.items():
        time.time()

        # Warm-up run
        _ = metric(preds, targets)
        torch.cuda.synchronize() if preds.is_cuda else None

        # Timed runs
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = metric(preds, targets)
            torch.cuda.synchronize() if preds.is_cuda else None
            times.append(time.time() - start)

        results[name] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
        }

    return results


def batch_metrics_to_table(
    metrics_dict: Dict[str, List[float]],
    precision: int = 4,
    include_std: bool = True,
) -> List[List[str]]:
    """Convert batch metrics to a table format.

    Args:
        metrics_dict (Dict[str, List[float]]): Dictionary mapping metric names to lists of values
        precision (int): Number of decimal places to display
        include_std (bool): Whether to include standard deviation

    Returns:
        List[List[str]]: Table data as list of rows
    """
    import numpy as np

    headers = ["Metric", "Mean"]
    if include_std:
        headers.append("Std")

    rows = [headers]

    for name, values in metrics_dict.items():
        values_array = np.array(values)
        row = [name, f"{values_array.mean():.{precision}f}"]
        if include_std:
            row.append(f"{values_array.std():.{precision}f}")
        rows.append(row)

    return rows


def print_metric_table(table: List[List[str]], column_widths: Optional[List[int]] = None) -> None:
    """Print a formatted table of metrics.

    Args:
        table (List[List[str]]): Table data as list of rows
        column_widths (Optional[List[int]]): Optional list of column widths
    """
    if not column_widths:
        # Calculate column widths based on content
        column_widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]

    # Print header
    header = table[0]
    print(" | ".join(h.ljust(w) for h, w in zip(header, column_widths)))
    print("-" * (sum(column_widths) + 3 * (len(column_widths) - 1)))

    # Print data rows
    for row in table[1:]:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, column_widths)))


def summarize_metrics_over_batches(metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize metrics collected over multiple batches.

    Args:
        metrics_history (List[Dict[str, Any]]): List of metric dictionaries, one per batch

    Returns:
        Dict[str, Any]: Summary statistics for each metric
    """
    import numpy as np

    # Initialize summary dict
    summary = {}

    # Collect all metrics
    for batch_metrics in metrics_history:
        for name, value in batch_metrics.items():
            if name not in summary:
                summary[name] = []

            # Handle both scalar values and (mean, std) tuples
            if isinstance(value, tuple) and len(value) == 2:
                # Store just the mean value for computing overall stats
                if isinstance(value[0], torch.Tensor):
                    summary[name].append(value[0].item())
                else:
                    summary[name].append(value[0])
            else:
                if isinstance(value, torch.Tensor):
                    summary[name].append(value.item())
                else:
                    summary[name].append(value)

    # Compute statistics
    result = {}
    for name, values in summary.items():
        values_array = np.array(values)
        result[name] = {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "n_samples": len(values_array),
        }

    return result
