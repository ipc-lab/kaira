"""
Utility modules for examples.

This package contains reusable utilities for examples, including
plotting functions that can be shared across different examples.
"""

from .plotting import (
    setup_plotting_style,
    plot_ldpc_matrix_comparison,
    plot_ber_performance,
    plot_ber_vs_snr_comparison,
    plot_complexity_comparison,
    plot_tanner_graph,
    plot_belief_propagation_iteration,
    plot_code_structure_comparison,
    plot_bit_error_visualization,
    plot_error_rate_comparison,
    plot_block_error_visualization,
    plot_qam_constellation_with_errors,
    plot_symbol_error_analysis,
    plot_multi_qam_ber_performance,
    plot_bler_vs_snr_analysis,
    plot_multiple_metrics_comparison,
    plot_signal_noise_comparison,
    plot_snr_psnr_comparison,
    plot_snr_vs_mse,
    plot_noise_level_analysis,
    plot_binary_channel_comparison,
    plot_channel_error_rates,
    plot_transition_matrices,
    plot_channel_capacity_analysis,
    BELIEF_CMAP,
    MODERN_PALETTE,
    MATRIX_CMAP
)

__all__ = [
    'setup_plotting_style',
    'plot_ldpc_matrix_comparison',
    'plot_ber_performance',
    'plot_ber_vs_snr_comparison',
    'plot_complexity_comparison',
    'plot_tanner_graph',
    'plot_belief_propagation_iteration',
    'plot_code_structure_comparison',
    'plot_bit_error_visualization',
    'plot_error_rate_comparison',
    'plot_block_error_visualization',
    'plot_qam_constellation_with_errors',
    'plot_symbol_error_analysis',
    'plot_multi_qam_ber_performance',
    'plot_bler_vs_snr_analysis',
    'plot_multiple_metrics_comparison',
    'plot_signal_noise_comparison',
    'plot_snr_psnr_comparison',
    'plot_snr_vs_mse',
    'plot_noise_level_analysis',
    'plot_binary_channel_comparison',
    'plot_channel_error_rates',
    'plot_transition_matrices',
    'plot_channel_capacity_analysis',
    'BELIEF_CMAP',
    'MODERN_PALETTE',
    'MATRIX_CMAP'
]
