"""Benchmarking utilities for metrics module."""

import time
from typing import Dict, List, Any, Union, Optional, Tuple
import json
from pathlib import Path
import textwrap

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from .base import BaseMetric


class MetricBenchmark:
    """Utility class for benchmarking metrics performance."""
    
    def __init__(self, 
                 metrics: Dict[str, BaseMetric], 
                 device: Optional[Union[str, torch.device]] = None):
        """Initialize the benchmark with metrics to test.
        
        Args:
            metrics (Dict[str, BaseMetric]): Dictionary of metrics to benchmark
            device (Optional[Union[str, torch.device]]): Device to run benchmarks on
        """
        self.metrics = metrics
        self.device = device
        self._results = {}
        
    def run_speed_benchmark(self,
                           input_shapes: List[Tuple[int, ...]],
                           repeat: int = 5,
                           warmup: int = 2) -> Dict[str, Any]:
        """Run speed benchmark on different input shapes.
        
        Args:
            input_shapes (List[Tuple[int, ...]]): List of input shapes to test
            repeat (int): Number of repetitions for each test
            warmup (int): Number of warmup runs
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {}
        
        for shape in input_shapes:
            shape_results = {}
            
            # Create random inputs of specified shape
            x = torch.rand(*shape)
            y = torch.rand(*shape)
            
            if self.device:
                x = x.to(self.device)
                y = y.to(self.device)
            
            for name, metric in self.metrics.items():
                # Warmup runs
                for _ in range(warmup):
                    _ = metric(x, y)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Timed runs
                times = []
                for _ in range(repeat):
                    start_time = time.time()
                    _ = metric(x, y)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start_time)
                
                # Store results
                shape_results[name] = {
                    "mean_time": float(np.mean(times)),
                    "std_time": float(np.std(times)),
                    "min_time": float(np.min(times)),
                    "max_time": float(np.max(times)),
                    "shape": shape
                }
            
            # Add the results for this shape
            shape_str = "x".join(str(dim) for dim in shape)
            results[shape_str] = shape_results
        
        self._results["speed"] = results
        return results
    
    def run_accuracy_benchmark(self, 
                              reference_data: Dict[str, Tensor], 
                              test_data: Dict[str, Tensor]) -> Dict[str, Any]:
        """Run accuracy benchmark against reference data.
        
        Args:
            reference_data (Dict[str, Tensor]): Dictionary mapping metric names to reference values
            test_data (Dict[str, Tensor]): Dictionary with prediction and target tensors
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        if "preds" not in test_data or "targets" not in test_data:
            raise ValueError("test_data must contain 'preds' and 'targets' keys")
        
        results = {}
        preds = test_data["preds"]
        targets = test_data["targets"]
        
        if self.device:
            preds = preds.to(self.device)
            targets = targets.to(self.device)
        
        for name, metric in self.metrics.items():
            if name not in reference_data:
                print(f"Warning: No reference data for {name}, skipping")
                continue
                
            value = metric(preds, targets)
            reference = reference_data[name]
            
            # Convert to float if tensor
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = float(value.item())
                else:
                    value = value.detach().cpu().numpy().tolist()
            
            if isinstance(reference, torch.Tensor):
                if reference.numel() == 1:
                    reference = float(reference.item())
                else:
                    reference = reference.detach().cpu().numpy().tolist()
            
            # Calculate difference if both are scalar
            diff = None
            if isinstance(value, (int, float)) and isinstance(reference, (int, float)):
                diff = abs(value - reference)
                rel_diff = abs(diff / reference) if reference != 0 else float('inf')
            
            results[name] = {
                "value": value,
                "reference": reference,
                "difference": diff,
                "relative_difference": rel_diff if diff is not None else None
            }
        
        self._results["accuracy"] = results
        return results
    
    def plot_speed_results(self, 
                          figsize: Tuple[int, int] = (12, 6),
                          title: str = "Metric Speed Benchmark", 
                          save_path: Optional[str] = None) -> None:
        """Plot speed benchmark results.
        
        Args:
            figsize (Tuple[int, int]): Figure size
            title (str): Plot title
            save_path (Optional[str]): Path to save the plot
        """
        if "speed" not in self._results:
            raise ValueError("No speed benchmark results to plot. Run run_speed_benchmark first.")
        
        results = self._results["speed"]
        
        plt.figure(figsize=figsize)
        
        shapes = list(results.keys())
        metric_names = list(self.metrics.keys())
        
        x = np.arange(len(shapes))
        width = 0.8 / len(metric_names)
        
        for i, name in enumerate(metric_names):
            times = [results[shape][name]["mean_time"] * 1000 for shape in shapes]  # Convert to ms
            errors = [results[shape][name]["std_time"] * 1000 for shape in shapes]  # Convert to ms
            
            plt.bar(
                x + i * width - (len(metric_names) - 1) * width / 2,
                times,
                width=width,
                yerr=errors,
                label=name,
                capsize=5
            )
        
        plt.xlabel("Input Shape")
        plt.ylabel("Time (ms)")
        plt.title(title)
        plt.xticks(x, shapes)
        plt.legend(loc="upper left")
        plt.yscale("log")  # Log scale often works better for timing comparisons
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def save_results(self, path: str) -> None:
        """Save benchmark results to a JSON file.
        
        Args:
            path (str): Path to save the results
        """
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2)
    
    def load_results(self, path: str) -> Dict:
        """Load benchmark results from a JSON file.
        
        Args:
            path (str): Path to load the results from
            
        Returns:
            Dict: The loaded results
        """
        with open(path, "r") as f:
            self._results = json.load(f)
        return self._results
    
    def generate_report(self, path: Optional[str] = None) -> str:
        """Generate a text report of benchmark results.
        
        Args:
            path (Optional[str]): Path to save the report
            
        Returns:
            str: The report text
        """
        report = []
        report.append("# Metrics Benchmark Report")
        report.append("")
        
        if "speed" in self._results:
            report.append("## Speed Benchmark")
            report.append("")
            report.append("| Metric | Input Shape | Mean Time (ms) | Std Dev (ms) |")
            report.append("|--------|-------------|---------------|--------------|")
            
            for shape, shape_results in self._results["speed"].items():
                for name, result in shape_results.items():
                    mean = result["mean_time"] * 1000  # Convert to ms
                    std = result["std_time"] * 1000    # Convert to ms
                    report.append(f"| {name} | {shape} | {mean:.4f} | {std:.4f} |")
            
            report.append("")
        
        if "accuracy" in self._results:
            report.append("## Accuracy Benchmark")
            report.append("")
            report.append("| Metric | Computed Value | Reference Value | Difference | Relative Difference |")
            report.append("|--------|---------------|----------------|------------|---------------------|")
            
            for name, result in self._results["accuracy"].items():
                value = result["value"]
                ref = result["reference"]
                diff = result["difference"]
                rel_diff = result["relative_difference"]
                
                value_str = f"{value:.6f}" if isinstance(value, (int, float)) else str(value)
                ref_str = f"{ref:.6f}" if isinstance(ref, (int, float)) else str(ref)
                diff_str = f"{diff:.6f}" if diff is not None else "N/A"
                rel_str = f"{rel_diff:.6f}" if rel_diff is not None else "N/A"
                
                report.append(f"| {name} | {value_str} | {ref_str} | {diff_str} | {rel_str} |")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if path:
            with open(path, "w") as f:
                f.write(report_text)
        
        return report_text


def demonstrate_benchmark():
    """Run a demonstration of the benchmarking capabilities.
    
    This function creates sample metrics, runs benchmarks, and displays results.
    """
    from .image import PSNR, SSIM, LPIPS
    
    # Create metrics
    metrics = {
        "PSNR": PSNR(data_range=1.0),
        "SSIM": SSIM(data_range=1.0),
        "LPIPS": LPIPS(net_type="alex")
    }
    
    # Sample input shapes to test
    shapes = [
        (1, 3, 64, 64),
        (1, 3, 128, 128),
        (4, 3, 64, 64),
        (4, 3, 128, 128)
    ]
    
    # Create benchmark
    benchmark = MetricBenchmark(metrics)
    
    # Run speed benchmark
    benchmark.run_speed_benchmark(shapes, repeat=3)
    
    # Plot results
    benchmark.plot_speed_results()
    
    # Generate and print report
    report = benchmark.generate_report()
    print(textwrap.dedent(report))


if __name__ == "__main__":
    demonstrate_benchmark()
