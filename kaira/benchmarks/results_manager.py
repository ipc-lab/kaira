"""Benchmark results management system for organizing and storing benchmark results."""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkResultsManager:
    """Manages benchmark results with improved directory structure and organization."""

    def __init__(self, base_dir: Union[str, Path] = "results"):
        """Initialize the results manager.

        Args:
            base_dir: Base directory for storing all benchmark results
        """
        self.base_dir = Path(base_dir)
        self._ensure_directory_structure()

    def _ensure_directory_structure(self) -> None:
        """Create the standardized directory structure for benchmark results."""
        directories = [
            self.base_dir,
            self.base_dir / "benchmarks",  # Individual benchmark results
            self.base_dir / "suites",  # Benchmark suite results
            self.base_dir / "experiments",  # Experimental runs
            self.base_dir / "comparisons",  # Comparative studies
            self.base_dir / "archives",  # Archived old results
            self.base_dir / "configs",  # Configuration files
            self.base_dir / "logs",  # Execution logs
            self.base_dir / "summaries",  # Summary reports
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def save_benchmark_result(self, result: BenchmarkResult, category: str = "benchmarks", experiment_name: Optional[str] = None, add_timestamp: bool = True) -> Path:
        """Save a single benchmark result with improved organization.

        Args:
            result: The benchmark result to save
            category: Category (benchmarks, suites, experiments, etc.)
            experiment_name: Optional experiment name for grouping
            add_timestamp: Whether to add timestamp to filename

        Returns:
            Path to the saved file
        """
        # Determine the directory structure
        if experiment_name:
            save_dir = self.base_dir / category / experiment_name
        else:
            save_dir = self.base_dir / category

        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        base_name = self._sanitize_filename(result.name)
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}_{result.benchmark_id[:8]}.json"
        else:
            filename = f"{base_name}_{result.benchmark_id[:8]}.json"

        filepath = save_dir / filename

        # Save the result
        result.save(filepath)
        logger.info(f"Saved benchmark result to {filepath}")

        return filepath

    def save_suite_results(self, results: List[BenchmarkResult], suite_name: str, experiment_name: Optional[str] = None) -> Dict[str, Path]:
        """Save multiple benchmark results from a suite.

        Args:
            results: List of benchmark results
            suite_name: Name of the benchmark suite
            experiment_name: Optional experiment name

        Returns:
            Dictionary mapping result names to file paths
        """
        saved_files = {}

        # Create suite-specific directory
        if experiment_name:
            suite_dir = self.base_dir / "suites" / experiment_name / suite_name
        else:
            suite_dir = self.base_dir / "suites" / suite_name

        suite_dir.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for result in results:
            filepath = self.save_benchmark_result(result, category=str(suite_dir), add_timestamp=False)
            saved_files[result.name] = filepath

        # Create suite summary
        summary_path = self._create_suite_summary(results, suite_dir, suite_name)
        saved_files["summary"] = summary_path

        return saved_files

    def _create_suite_summary(self, results: List[BenchmarkResult], suite_dir: Path, suite_name: str) -> Path:
        """Create a summary file for a benchmark suite."""
        summary_data: Dict[str, Any] = {
            "suite_name": suite_name,
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(results),
            "total_execution_time": sum(r.execution_time for r in results),
            "successful_benchmarks": len([r for r in results if r.metrics.get("success", True)]),
            "failed_benchmarks": len([r for r in results if not r.metrics.get("success", True)]),
            "benchmark_summaries": [],
        }

        for result in results:
            benchmark_summary = {"name": result.name, "benchmark_id": result.benchmark_id, "execution_time": result.execution_time, "success": result.metrics.get("success", True), "key_metrics": self._extract_key_metrics(result.metrics)}
            summary_data["benchmark_summaries"].append(benchmark_summary)

        summary_path = suite_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"Created suite summary at {summary_path}")
        return summary_path

    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for summary display."""
        key_metrics = {}

        # Common performance metrics to extract
        important_keys = ["throughput", "latency", "ber", "capacity", "snr", "mse", "psnr", "processing_time", "memory_usage", "accuracy", "error_rate"]

        for key in important_keys:
            if key in metrics:
                key_metrics[key] = metrics[key]

        return key_metrics

    def load_benchmark_result(self, filepath: Union[str, Path]) -> BenchmarkResult:
        """Load a benchmark result from file."""
        with open(filepath) as f:
            data = json.load(f)

        # Filter data to only include parameters that BenchmarkResult constructor accepts
        valid_params = {"benchmark_id", "name", "description", "metrics", "execution_time", "timestamp", "metadata"}
        filtered_data = {k: v for k, v in data.items() if k in valid_params}

        # Check if we have all required parameters
        required_params = {"benchmark_id", "name", "description", "metrics", "execution_time", "timestamp"}
        missing_params = required_params - set(filtered_data.keys())
        if missing_params:
            raise ValueError(f"File {filepath} is not a valid BenchmarkResult file (missing: {missing_params})")

        return BenchmarkResult(**filtered_data)

    def list_results(self, category: Optional[str] = None, experiment_name: Optional[str] = None) -> List[Path]:
        """List available benchmark result files.

        Args:
            category: Specific category to list (benchmarks, suites, etc.)
            experiment_name: Specific experiment to list

        Returns:
            List of result file paths (excludes summary files and comparison reports)
        """
        if category and experiment_name:
            search_dir = self.base_dir / category / experiment_name
        elif category:
            search_dir = self.base_dir / category
        else:
            search_dir = self.base_dir

        if not search_dir.exists():
            return []

        # Get all JSON files but exclude summary files and comparison reports
        all_json_files = list(search_dir.rglob("*.json"))
        excluded_files = {"summary.json"}
        excluded_dirs = {"comparisons", "archives"}

        valid_files = []
        for f in all_json_files:
            # Skip if filename is in excluded list
            if f.name in excluded_files:
                continue

            # Skip if file is in an excluded directory
            if any(excluded_dir in f.parts for excluded_dir in excluded_dirs):
                continue

            # Skip comparison report files (they end with _comparison.json)
            if f.name.endswith("_comparison.json"):
                continue

            valid_files.append(f)

        return valid_files

    def archive_old_results(self, days_old: int = 30) -> None:
        """Archive benchmark results older than specified days.

        Args:
            days_old: Number of days after which to archive results
        """
        import time

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)

        archived_count = 0
        for result_file in self.base_dir.rglob("*.json"):
            if result_file.parent.name == "archives":
                continue  # Skip already archived files

            if result_file.stat().st_mtime < cutoff_time:
                # Create archive path maintaining directory structure
                relative_path = result_file.relative_to(self.base_dir)
                archive_path = self.base_dir / "archives" / relative_path
                archive_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.move(str(result_file), str(archive_path))
                archived_count += 1
                logger.info(f"Archived {result_file} to {archive_path}")

        logger.info(f"Archived {archived_count} old result files")

    def cleanup_empty_directories(self) -> None:
        """Remove empty directories in the results structure."""
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            for directory in dirs:
                dir_path = Path(root) / directory
                try:
                    if not any(dir_path.iterdir()):  # Directory is empty
                        dir_path.rmdir()
                        logger.debug(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or permission issues

    def create_comparison_report(self, result_paths: List[Path], report_name: str) -> Path:
        """Create a comparison report from multiple benchmark results.

        Args:
            result_paths: List of paths to benchmark result files
            report_name: Name for the comparison report

        Returns:
            Path to the generated report
        """
        results = [self.load_benchmark_result(path) for path in result_paths]

        comparison_data: Dict[str, Any] = {"report_name": report_name, "timestamp": datetime.now().isoformat(), "compared_results": len(results), "results": []}

        for i, result in enumerate(results):
            result_summary = {"index": i, "name": result.name, "benchmark_id": result.benchmark_id, "execution_time": result.execution_time, "metrics": result.metrics, "timestamp": result.timestamp}
            comparison_data["results"].append(result_summary)

        # Save comparison report
        report_path = self.base_dir / "comparisons" / f"{report_name}_comparison.json"
        with open(report_path, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)

        logger.info(f"Created comparison report at {report_path}")
        return report_path

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a string to be safe for use as a filename."""
        # Replace problematic characters
        sanitized = name.replace(" ", "_").replace("(", "").replace(")", "")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-.")
        return sanitized[:100]  # Limit length
