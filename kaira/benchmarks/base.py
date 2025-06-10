"""Base classes for the Kaira benchmarking system."""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    benchmark_id: str
    name: str
    description: str
    metrics: Dict[str, Any]
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save result to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, name: str, description: str = ""):
        """Initialize base benchmark.

        Args:
            name: Name of the benchmark
            description: Description of what the benchmark tests
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        self._setup_called = False
        self._teardown_called = False

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """Setup benchmark environment."""
        self._setup_called = True

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the benchmark and return metrics."""
        pass

    def teardown(self) -> None:
        """Clean up after benchmark."""
        self._teardown_called = True

    def execute(self, **kwargs) -> BenchmarkResult:
        """Execute the full benchmark pipeline."""
        if not self._setup_called:
            self.setup(**kwargs)

        start_time = time.time()
        try:
            metrics = self.run(**kwargs)
        except Exception as e:
            metrics = {"error": str(e), "success": False}
        finally:
            execution_time = time.time() - start_time

        if not self._teardown_called:
            self.teardown()

        return BenchmarkResult(benchmark_id=self.id, name=self.name, description=self.description, metrics=metrics, execution_time=execution_time, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), metadata=kwargs)


class BenchmarkSuite:
    """Collection of benchmarks that can be run together."""

    def __init__(self, name: str, description: str = ""):
        """Initialize benchmark suite.

        Args:
            name: Name of the benchmark suite
            description: Description of the suite
        """
        self.name = name
        self.description = description
        self.benchmarks: List[BaseBenchmark] = []
        self.results: List[BenchmarkResult] = []

    def add_benchmark(self, benchmark: BaseBenchmark) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    def run_all(self, **kwargs) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        self.results = []
        for benchmark in self.benchmarks:
            result = benchmark.execute(**kwargs)
            self.results.append(result)
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all results."""
        if not self.results:
            return {}

        total_time = sum(r.execution_time for r in self.results)
        successful = sum(1 for r in self.results if r.metrics.get("success", True))

        return {"suite_name": self.name, "total_benchmarks": len(self.results), "successful": successful, "failed": len(self.results) - successful, "total_execution_time": total_time, "average_execution_time": total_time / len(self.results)}

    def save_results(self, directory: Union[str, Path]) -> None:
        """Save all results to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for result in self.results:
            filename = f"{result.name}_{result.benchmark_id[:8]}.json"
            result.save(directory / filename)

        # Save summary
        summary = self.get_summary()
        with open(directory / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


class CommunicationBenchmark(BaseBenchmark):
    """Base class for communication system benchmarks."""

    def __init__(self, name: str, description: str = "", snr_range: Optional[List[float]] = None):
        """Initialize communication benchmark.

        Args:
            name: Name of the benchmark
            description: Description of the benchmark
            snr_range: SNR range for testing (dB)
        """
        super().__init__(name, description)
        self.snr_range = snr_range or torch.arange(-10, 15, 1).tolist()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, **kwargs) -> None:
        """Setup communication benchmark environment."""
        super().setup(**kwargs)
        # Set random seeds for reproducibility
        torch.manual_seed(kwargs.get("seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kwargs.get("seed", 42))
