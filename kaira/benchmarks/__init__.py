"""Kaira Benchmarking System.

This module provides standardized benchmarks for evaluating communication system components and
deep learning models in Kaira.
"""

from . import ecc_benchmark  # Import ECC benchmarks to register them  # noqa: F401
from . import standard  # Import standard benchmarks to register them  # noqa: F401
from .base import BaseBenchmark, BenchmarkResult, BenchmarkSuite
from .config import BenchmarkConfig, get_config, list_configs
from .metrics import StandardMetrics
from .registry import (
    BenchmarkRegistry,
    create_benchmark,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from .results_manager import BenchmarkResultsManager
from .runners import ComparisonRunner, ParallelRunner, ParametricRunner, StandardRunner
from .visualization import BenchmarkVisualizer

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkRegistry",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "create_benchmark",
    "StandardMetrics",
    "StandardRunner",
    "ParallelRunner",
    "ComparisonRunner",
    "ParametricRunner",
    "BenchmarkConfig",
    "get_config",
    "list_configs",
    "BenchmarkResultsManager",
    "BenchmarkVisualizer",
]
