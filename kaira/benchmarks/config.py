"""Configuration management for benchmarks."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # General settings
    name: str = "default"
    description: str = ""
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"

    # Execution settings
    num_trials: int = 1
    timeout_seconds: Optional[float] = None
    verbose: bool = True
    save_results: bool = True

    # Output settings
    output_directory: str = "./benchmark_results"
    save_plots: bool = True
    save_raw_data: bool = False

    # Performance settings
    batch_size: int = 1000
    num_workers: int = 1
    memory_limit_mb: Optional[float] = None

    # Communication system specific
    snr_range: List[float] = field(default_factory=lambda: list(range(-10, 16)))
    block_length: int = 1000
    code_rate: float = 0.5

    # Model specific
    model_precision: str = "float32"  # "float16", "float32", "float64"
    compile_model: bool = False

    # Metrics settings
    calculate_confidence_intervals: bool = True
    confidence_level: float = 0.95

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkConfig":
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BenchmarkConfig":
        """Load configuration from file."""
        with open(filepath) as f:
            return cls.from_json(f.read())

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.custom_params.get(key, default)


# Predefined configurations for common scenarios
STANDARD_CONFIGS = {
    "fast": BenchmarkConfig(name="fast", description="Fast benchmark configuration for quick testing", num_trials=1, snr_range=[-5, 0, 5, 10], block_length=100, verbose=True),
    "accurate": BenchmarkConfig(name="accurate", description="High-accuracy configuration for publication results", num_trials=10, snr_range=list(range(-10, 16)), block_length=10000, calculate_confidence_intervals=True, save_raw_data=True),
    "comprehensive": BenchmarkConfig(name="comprehensive", description="Comprehensive benchmarking with all metrics", num_trials=5, snr_range=list(range(-15, 21)), block_length=5000, save_plots=True, save_raw_data=True, calculate_confidence_intervals=True),
    "gpu": BenchmarkConfig(name="gpu", description="GPU-optimized configuration", device="cuda", batch_size=10000, model_precision="float16", compile_model=True, num_trials=3),
    "minimal": BenchmarkConfig(name="minimal", description="Minimal configuration for CI/CD", num_trials=1, snr_range=[0, 10], block_length=100, verbose=False, save_plots=False),
}


def get_config(name: str) -> BenchmarkConfig:
    """Get a predefined configuration."""
    if name not in STANDARD_CONFIGS:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(STANDARD_CONFIGS.keys())}")
    return STANDARD_CONFIGS[name]


def list_configs() -> List[str]:
    """List available predefined configurations."""
    return list(STANDARD_CONFIGS.keys())
