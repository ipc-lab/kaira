"""Benchmark registry for managing and discovering benchmarks."""

from typing import Dict, List, Optional, Type

from .base import BaseBenchmark


class BenchmarkRegistry:
    """Registry for managing benchmark classes and instances."""

    _instance = None
    """Singleton instance of the registry."""
    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}
    """Dictionary storing registered benchmark classes."""

    def __new__(cls):
        """Ensure singleton pattern for the registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, benchmark_class: Type[BaseBenchmark]) -> None:
        """Register a benchmark class."""
        cls._benchmarks[name] = benchmark_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseBenchmark]]:
        """Get a registered benchmark class."""
        return cls._benchmarks.get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available benchmark names."""
        return list(cls._benchmarks.keys())

    @classmethod
    def create_benchmark(cls, name: str, **kwargs) -> Optional[BaseBenchmark]:
        """Create an instance of a registered benchmark."""
        benchmark_class = cls.get(name)
        if benchmark_class is None:
            return None
        return benchmark_class(**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered benchmarks."""
        cls._benchmarks.clear()


# Global registry instance
_registry = BenchmarkRegistry()


def register_benchmark(name: str):
    """Decorator to register a benchmark class."""

    def decorator(benchmark_class: Type[BaseBenchmark]):
        """Register the benchmark class with the given name."""
        _registry.register(name, benchmark_class)
        return benchmark_class

    return decorator


def get_benchmark(name: str) -> Optional[Type[BaseBenchmark]]:
    """Get a registered benchmark class."""
    return _registry.get(name)


def list_benchmarks() -> List[str]:
    """List all available benchmark names."""
    return _registry.list_available()


def create_benchmark(name: str, **kwargs) -> Optional[BaseBenchmark]:
    """Create an instance of a registered benchmark."""
    return _registry.create_benchmark(name, **kwargs)
