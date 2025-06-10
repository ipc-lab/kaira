"""Configuration templates for Error Correction Codes benchmarks.

This module provides predefined configurations for comprehensive ECC evaluation, making it easy to
run standardized benchmarks across different code families.
"""

from typing import Any, Dict, List

from kaira.benchmarks.config import BenchmarkConfig

# Predefined ECC benchmark configurations
ECC_BENCHMARK_CONFIGS = {
    # Fast configuration for development and testing
    "fast": BenchmarkConfig(
        name="ecc_fast_evaluation", description="Fast ECC evaluation for development", snr_range=list(range(-2, 8, 2)), block_length=1000, num_trials=20, verbose=True, save_results=True, output_directory="./ecc_benchmark_results/fast", custom_params={"num_bits": 1000, "max_errors": 3}
    ),
    # Standard configuration for regular benchmarking
    "standard": BenchmarkConfig(
        name="ecc_standard_evaluation",
        description="Standard ECC evaluation configuration",
        snr_range=list(range(-5, 12, 1)),
        block_length=10000,
        num_trials=100,
        verbose=True,
        save_results=True,
        save_plots=True,
        output_directory="./ecc_benchmark_results/standard",
        custom_params={"num_bits": 10000, "max_errors": 8},
    ),
    # Comprehensive configuration for publication-quality results
    "comprehensive": BenchmarkConfig(
        name="ecc_comprehensive_evaluation",
        description="Comprehensive ECC evaluation for research",
        snr_range=list(range(-8, 15)),
        block_length=100000,
        num_trials=500,
        verbose=True,
        save_results=True,
        save_plots=True,
        save_raw_data=True,
        calculate_confidence_intervals=True,
        confidence_level=0.95,
        output_directory="./ecc_benchmark_results/comprehensive",
        custom_params={"num_bits": 100000, "max_errors": 15},
    ),
    # High SNR configuration for studying error floor behavior
    "high_snr": BenchmarkConfig(
        name="ecc_high_snr_evaluation",
        description="High SNR evaluation for error floor analysis",
        snr_range=list(range(8, 25, 1)),
        block_length=1000000,
        num_trials=100,
        verbose=True,
        save_results=True,
        save_plots=True,
        output_directory="./ecc_benchmark_results/high_snr",
        custom_params={"num_bits": 1000000, "max_errors": 20},
    ),
    # Low complexity configuration for embedded systems
    "low_complexity": BenchmarkConfig(
        name="ecc_low_complexity_evaluation",
        description="Low complexity ECC evaluation for embedded systems",
        snr_range=list(range(-3, 10, 2)),
        block_length=5000,
        num_trials=50,
        verbose=True,
        save_results=True,
        output_directory="./ecc_benchmark_results/low_complexity",
        custom_params={"num_bits": 5000, "max_errors": 5},
    ),
}


# Specific configurations for different ECC families
ECC_FAMILY_CONFIGS = {
    "hamming": {"codes_to_test": [{"mu": 3, "name": "Hamming(7,4)"}, {"mu": 4, "name": "Hamming(15,11)"}, {"mu": 5, "name": "Hamming(31,26)"}, {"mu": 6, "name": "Hamming(63,57)"}], "focus_metrics": ["ber_coded", "coding_gain_ber", "complexity"], "recommended_snr_range": list(range(-2, 12, 1))},
    "bch": {
        "codes_to_test": [{"mu": 4, "delta": 5, "name": "BCH(15,7)"}, {"mu": 5, "delta": 7, "name": "BCH(31,16)"}, {"mu": 6, "delta": 11, "name": "BCH(63,36)"}, {"mu": 7, "delta": 15, "name": "BCH(127,64)"}],
        "focus_metrics": ["ber_coded", "bler_coded", "coding_gain_ber", "error_correction_capability"],
        "recommended_snr_range": list(range(-1, 15, 1)),
    },
    "golay": {"codes_to_test": [{"extended": False, "name": "Golay(23,12)"}, {"extended": True, "name": "Extended Golay(24,12)"}], "focus_metrics": ["ber_coded", "bler_coded", "coding_gain_ber", "perfect_code_properties"], "recommended_snr_range": list(range(0, 18, 1))},
    "repetition": {
        "codes_to_test": [{"repetition_factor": 3, "name": "Repetition(3,1)"}, {"repetition_factor": 5, "name": "Repetition(5,1)"}, {"repetition_factor": 7, "name": "Repetition(7,1)"}, {"repetition_factor": 9, "name": "Repetition(9,1)"}],
        "focus_metrics": ["ber_coded", "coding_gain_ber", "simplicity"],
        "recommended_snr_range": list(range(-5, 8, 1)),
    },
    "reed_solomon": {
        "codes_to_test": [{"n": 15, "k": 11, "name": "Reed-Solomon(15,11)"}, {"n": 31, "k": 19, "name": "Reed-Solomon(31,19)"}, {"n": 63, "k": 39, "name": "Reed-Solomon(63,39)"}],
        "focus_metrics": ["ber_coded", "bler_coded", "burst_error_correction"],
        "recommended_snr_range": list(range(2, 20, 1)),
    },
}


# Benchmark suite configurations for different use cases
BENCHMARK_SUITE_CONFIGS = {
    "academic_comparison": {
        "name": "Academic ECC Comparison Suite",
        "description": "Comprehensive comparison of ECC families for academic research",
        "families": ["hamming", "bch", "golay", "repetition"],
        "base_config": "comprehensive",
        "additional_metrics": ["theoretical_bounds", "asymptotic_behavior"],
    },
    "industry_evaluation": {
        "name": "Industry ECC Evaluation Suite",
        "description": "Practical ECC evaluation for industry applications",
        "families": ["hamming", "bch", "reed_solomon"],
        "base_config": "standard",
        "additional_metrics": ["throughput", "power_consumption", "implementation_complexity"],
    },
    "satellite_communications": {
        "name": "Satellite Communications ECC Suite",
        "description": "ECC evaluation for satellite communication systems",
        "families": ["bch", "golay", "reed_solomon"],
        "base_config": "high_snr",
        "additional_metrics": ["burst_error_performance", "interleaving_compatibility"],
    },
    "iot_embedded": {"name": "IoT Embedded Systems ECC Suite", "description": "ECC evaluation for IoT and embedded applications", "families": ["hamming", "repetition"], "base_config": "low_complexity", "additional_metrics": ["energy_efficiency", "memory_requirements", "real_time_performance"]},
}


def get_ecc_config(config_name: str) -> BenchmarkConfig:
    """Get a predefined ECC benchmark configuration.

    Args:
        config_name: Name of the configuration ('fast', 'standard', 'comprehensive', etc.)

    Returns:
        BenchmarkConfig object with the specified configuration

    Raises:
        KeyError: If the configuration name is not found
    """
    if config_name not in ECC_BENCHMARK_CONFIGS:
        available_configs = list(ECC_BENCHMARK_CONFIGS.keys())
        raise KeyError(f"Configuration '{config_name}' not found. Available configurations: {available_configs}")

    return ECC_BENCHMARK_CONFIGS[config_name]


def get_family_config(family_name: str) -> Dict[str, Any]:
    """Get configuration specific to an ECC family.

    Args:
        family_name: Name of the ECC family ('hamming', 'bch', 'golay', etc.)

    Returns:
        Dictionary containing family-specific configuration

    Raises:
        KeyError: If the family name is not found
    """
    if family_name not in ECC_FAMILY_CONFIGS:
        available_families = list(ECC_FAMILY_CONFIGS.keys())
        raise KeyError(f"Family '{family_name}' not found. Available families: {available_families}")

    return ECC_FAMILY_CONFIGS[family_name]


def get_suite_config(suite_name: str) -> Dict[str, Any]:
    """Get configuration for a benchmark suite.

    Args:
        suite_name: Name of the benchmark suite

    Returns:
        Dictionary containing suite configuration

    Raises:
        KeyError: If the suite name is not found
    """
    if suite_name not in BENCHMARK_SUITE_CONFIGS:
        available_suites = list(BENCHMARK_SUITE_CONFIGS.keys())
        raise KeyError(f"Suite '{suite_name}' not found. Available suites: {available_suites}")

    return BENCHMARK_SUITE_CONFIGS[suite_name]


def create_custom_ecc_config(name: str, snr_range: List[float], num_bits: int = 10000, num_trials: int = 100, max_errors: int = 8, **kwargs) -> BenchmarkConfig:
    """Create a custom ECC benchmark configuration.

    Args:
        name: Name for the configuration
        snr_range: List of SNR values to test
        num_bits: Number of information bits to test
        num_trials: Number of Monte Carlo trials
        max_errors: Maximum number of errors to test in error correction evaluation
        **kwargs: Additional configuration parameters

    Returns:
        Custom BenchmarkConfig object
    """
    # Pack num_bits, num_trials, and max_errors into custom_params
    custom_params = {"num_bits": num_bits, "num_trials": num_trials, "max_errors": max_errors}

    # Add any additional custom parameters
    for k, v in kwargs.items():
        if k not in ["description", "verbose", "save_results", "save_plots", "output_directory"]:
            custom_params[k] = v

    return BenchmarkConfig(
        name=name,
        description=kwargs.get("description", f"Custom ECC configuration: {name}"),
        snr_range=snr_range,
        block_length=num_bits,  # Use block_length as the main parameter
        custom_params=custom_params,
        verbose=kwargs.get("verbose", True),
        save_results=kwargs.get("save_results", True),
        save_plots=kwargs.get("save_plots", True),
        output_directory=kwargs.get("output_directory", f"./ecc_benchmark_results/{name}"),
    )


# Utility function to list all available configurations
def list_all_configs() -> Dict[str, List[str]]:
    """List all available ECC benchmark configurations.

    Returns:
        Dictionary with categories of configurations and their names
    """
    return {"benchmark_configs": list(ECC_BENCHMARK_CONFIGS.keys()), "family_configs": list(ECC_FAMILY_CONFIGS.keys()), "suite_configs": list(BENCHMARK_SUITE_CONFIGS.keys())}
