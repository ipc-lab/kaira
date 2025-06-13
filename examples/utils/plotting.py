"""
Plotting utilities for LDPC and FEC examples.

This module provides reusable plotting functions to keep example files
focused on the core algorithm demonstrations while maintaining consistent
visualization across examples.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from typing import Optional, List, Tuple, Dict, Any


# Configure default plotting style
def setup_plotting_style():
    """Set up consistent plotting style for all examples."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.2)


# Color schemes and palettes
BELIEF_CMAP = LinearSegmentedColormap.from_list("belief", ["#d32f2f", "#ffeb3b", "#4caf50"], N=256)
MODERN_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MATRIX_CMAP = LinearSegmentedColormap.from_list("matrix", ["white", "#2c3e50"])


def plot_ldpc_matrix_comparison(H_matrices: List[torch.Tensor], 
                               titles: List[str], 
                               main_title: str = "LDPC Matrix Comparison") -> plt.Figure:
    """
    Plot comparison of LDPC code matrix structures.
    
    Parameters
    ----------
    H_matrices : List[torch.Tensor]
        List of parity check matrices to compare
    titles : List[str]
        Titles for each matrix
    main_title : str
        Overall plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    n_matrices = len(H_matrices)
    fig, axes = plt.subplots(1, n_matrices, figsize=(6*n_matrices, 5), constrained_layout=True)
    
    if n_matrices == 1:
        axes = [axes]
    
    fig.suptitle(main_title, fontsize=16, fontweight="bold")
    
    for i, (H, title) in enumerate(zip(H_matrices, titles)):
        ax = axes[i]
        H_np = H.numpy() if isinstance(H, torch.Tensor) else H
        m, n = H_np.shape
        
        # Plot matrix heatmap
        im = ax.imshow(H_np, cmap=MATRIX_CMAP, interpolation="nearest", aspect="auto")
        
        # Add text annotations for small matrices
        if m <= 8 and n <= 12:
            for row in range(m):
                for col in range(n):
                    color = "white" if H_np[row, col] == 1 else "black"
                    ax.text(col, row, int(H_np[row, col]), ha="center", va="center", 
                           color=color, fontsize=12, fontweight="bold")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Variable Nodes", fontsize=12)
        ax.set_ylabel("Check Nodes", fontsize=12)
        
        # Add sparsity information
        sparsity = np.sum(H_np) / (m * n)
        ax.text(0.02, 0.98, f"Sparsity: {sparsity:.3f}", 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
               verticalalignment="top")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    return fig


def plot_ber_performance(snr_range: np.ndarray, 
                        ber_values: List[np.ndarray], 
                        labels: List[str],
                        title: str = "BER vs SNR Performance",
                        ylabel: str = "Bit Error Rate") -> plt.Figure:
    """
    Plot BER vs SNR performance curves.
    
    Parameters
    ----------
    snr_range : np.ndarray
        SNR values in dB
    ber_values : List[np.ndarray]
        BER values for each configuration
    labels : List[str]
        Labels for each curve
    title : str
        Plot title
    ylabel : str
        Y-axis label
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    for i, (ber, label) in enumerate(zip(ber_values, labels)):
        # Convert to numpy array if it's a list
        ber_array = np.array(ber) if isinstance(ber, list) else ber
        color = MODERN_PALETTE[i % len(MODERN_PALETTE)]
        ax.semilogy(snr_range, ber_array, 'o-', color=color, linewidth=2, 
                   markersize=6, label=label, alpha=0.8)
    
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set reasonable y-axis limits
    all_ber_arrays = [np.array(ber) if isinstance(ber, list) else ber for ber in ber_values]
    non_zero_bers = [ber_arr[ber_arr > 0] for ber_arr in all_ber_arrays if len(ber_arr[ber_arr > 0]) > 0]
    if non_zero_bers:
        min_ber = min([np.min(ber_subset) for ber_subset in non_zero_bers])
        ax.set_ylim(min_ber / 10, 1)
    else:
        ax.set_ylim(1e-6, 1)
    
    return fig


def plot_complexity_comparison(code_types: List[str], 
                             metrics: Dict[str, List[float]],
                             title: str = "Complexity Comparison") -> plt.Figure:
    """
    Plot complexity comparison charts.
    
    Parameters
    ----------
    code_types : List[str]
        Names of different code types
    metrics : Dict[str, List[float]]
        Dictionary mapping metric names to values for each code type
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5), constrained_layout=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        bars = ax.bar(code_types, values, color=MODERN_PALETTE[:len(code_types)], 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=11)
        ax.tick_params(axis='x', rotation=45)
    
    return fig


def plot_tanner_graph(H: torch.Tensor, 
                     title: str = "LDPC Tanner Graph") -> plt.Figure:
    """
    Create enhanced Tanner graph visualization.
    
    Parameters
    ----------
    H : torch.Tensor
        Parity check matrix
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    H_np = H.numpy() if isinstance(H, torch.Tensor) else H
    m, n = H_np.shape  # m check nodes, n variable nodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Left plot: Bipartite graph representation
    ax1.set_title("Bipartite Graph", fontsize=14, fontweight="bold")

    # Position variable nodes in a circle (top)
    var_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    var_positions = [(2 * np.cos(angle), 2 * np.sin(angle) + 1) for angle in var_angles]

    # Position check nodes in a circle (bottom)
    check_angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    check_positions = [(1.5 * np.cos(angle), 1.5 * np.sin(angle) - 1) for angle in check_angles]

    # Draw connections
    connection_counts = np.sum(H_np, axis=0)  # variable node degrees
    max_degree = np.max(connection_counts)

    for i in range(m):
        for j in range(n):
            if H_np[i, j] == 1:
                thickness = 1 + 2 * (connection_counts[j] / max_degree)
                alpha = 0.6 + 0.4 * (connection_counts[j] / max_degree)
                line = FancyArrowPatch(check_positions[i], var_positions[j], 
                                     arrowstyle="-", color="gray", 
                                     linewidth=thickness, alpha=alpha, 
                                     connectionstyle="arc3,rad=0.1")
                ax1.add_patch(line)

    # Draw variable nodes
    for j, pos in enumerate(var_positions):
        size = 0.15 + 0.15 * (connection_counts[j] / max_degree)
        circle = Circle(pos, size, facecolor=MODERN_PALETTE[0], 
                       edgecolor="black", linewidth=2, zorder=10)
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], f"v{j}", ha="center", va="center", 
                fontsize=10, fontweight="bold", color="white", zorder=11)

    # Draw check nodes
    check_degrees = np.sum(H_np, axis=1)
    max_check_degree = np.max(check_degrees)

    for i, pos in enumerate(check_positions):
        size = 0.15 + 0.15 * (check_degrees[i] / max_check_degree)
        square = Rectangle((pos[0] - size, pos[1] - size), 2 * size, 2 * size, 
                         facecolor=MODERN_PALETTE[3], edgecolor="black", 
                         linewidth=2, zorder=10)
        ax1.add_patch(square)
        ax1.text(pos[0], pos[1], f"c{i}", ha="center", va="center", 
                fontsize=10, fontweight="bold", color="white", zorder=11)

    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODERN_PALETTE[0], 
                  markersize=10, label="Variable Nodes"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=MODERN_PALETTE[3], 
                  markersize=10, label="Check Nodes"),
        plt.Line2D([0], [0], color="gray", linewidth=2, label="Connections"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Right plot: Matrix heatmap
    ax2.set_title("Parity Check Matrix H", fontsize=14, fontweight="bold")
    im = ax2.imshow(H_np, cmap=MATRIX_CMAP, interpolation="nearest", aspect="auto")

    # Add text annotations for reasonable-sized matrices
    if m <= 10 and n <= 15:
        for i in range(m):
            for j in range(n):
                color = "black" if H_np[i, j] == 0 else "white"
                ax2.text(j, i, int(H_np[i, j]), ha="center", va="center", 
                        color=color, fontsize=12, fontweight="bold")

    ax2.set_xticks(range(n))
    ax2.set_yticks(range(m))
    ax2.set_xlabel("Variable Nodes", fontsize=12)
    ax2.set_ylabel("Check Nodes", fontsize=12)

    # Add colorbar and sparsity info
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    sparsity = np.sum(H_np) / (m * n)
    ax2.text(0.02, 0.98, f"Sparsity: {sparsity:.3f}\nDensity: {1-sparsity:.3f}", 
            transform=ax2.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), 
            verticalalignment="top")

    return fig


def plot_belief_propagation_iteration(beliefs: np.ndarray,
                                    var_to_check: np.ndarray,
                                    check_to_var: np.ndarray,
                                    iteration: int,
                                    belief_history: Optional[List[np.ndarray]] = None) -> plt.Figure:
    """
    Visualize a specific belief propagation iteration.
    
    Parameters
    ----------
    beliefs : np.ndarray
        Current variable node beliefs
    var_to_check : np.ndarray
        Variable to check node messages
    check_to_var : np.ndarray
        Check to variable node messages
    iteration : int
        Current iteration number
    belief_history : Optional[List[np.ndarray]]
        History of beliefs for convergence tracking
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(f"Belief Propagation - Iteration {iteration}", fontsize=16, fontweight="bold")

    # Current beliefs
    n = len(beliefs)
    hard_decisions = (beliefs > 0).astype(int)

    colors = [BELIEF_CMAP(0.5 + 0.5 * np.tanh(b / 4)) for b in beliefs]
    bars = ax1.bar(range(n), beliefs, color=colors, edgecolor="black", linewidth=1)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_title("Variable Node Beliefs (LLRs)")
    ax1.set_xlabel("Variable Node")
    ax1.set_ylabel("Log-Likelihood Ratio")

    # Add hard decision annotations
    for i, (belief, decision) in enumerate(zip(beliefs, hard_decisions)):
        y_pos = belief + 0.2 * np.sign(belief) if belief != 0 else 0.2
        ax1.text(i, y_pos, str(decision), ha="center", 
                va="bottom" if belief >= 0 else "top", 
                fontweight="bold", fontsize=12)

    # Variable to check messages
    im2 = ax2.imshow(var_to_check.T, cmap="RdBu_r", aspect="auto", vmin=-5, vmax=5)
    ax2.set_title("Variable → Check Messages")
    ax2.set_xlabel("Variable Node")
    ax2.set_ylabel("Check Node")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Check to variable messages
    im3 = ax3.imshow(check_to_var, cmap="RdBu_r", aspect="auto", vmin=-5, vmax=5)
    ax3.set_title("Check → Variable Messages")
    ax3.set_xlabel("Variable Node")
    ax3.set_ylabel("Check Node")
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    # Convergence tracking
    if belief_history and iteration > 0:
        belief_changes = [
            np.linalg.norm(belief_history[i + 1] - belief_history[i]) 
            for i in range(min(iteration, len(belief_history) - 1))
        ]
        ax4.plot(range(1, len(belief_changes) + 1), belief_changes, "o-", 
                color=MODERN_PALETTE[0], linewidth=2, markersize=8)
        ax4.set_title("Belief Convergence")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("L2 Norm of Belief Change")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Initial State\n(No convergence data)", 
                ha="center", va="center", transform=ax4.transAxes, 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.5))
        ax4.set_title("Belief Convergence")

    return fig


def plot_code_structure_comparison(structures: List[Dict[str, Any]], 
                                 title: str = "Code Structure Comparison") -> plt.Figure:
    """
    Plot comparison of different code structures and properties.
    
    Parameters
    ----------
    structures : List[Dict[str, Any]]
        List of dictionaries containing code properties
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    n_codes = len(structures)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    # Extract properties
    names = [struct['name'] for struct in structures]
    rates = [struct.get('rate', 0) for struct in structures]
    lengths = [struct.get('length', 0) for struct in structures]
    dimensions = [struct.get('dimension', 0) for struct in structures]
    
    # Code rates
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, rates, color=MODERN_PALETTE[:n_codes], alpha=0.8)
    ax1.set_title("Code Rates", fontweight="bold")
    ax1.set_ylabel("Rate (k/n)")
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars1, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Code lengths
    ax2 = axes[0, 1]
    bars2 = ax2.bar(names, lengths, color=MODERN_PALETTE[:n_codes], alpha=0.8)
    ax2.set_title("Code Lengths", fontweight="bold")
    ax2.set_ylabel("n (bits)")
    ax2.tick_params(axis='x', rotation=45)
    
    # Code dimensions
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, dimensions, color=MODERN_PALETTE[:n_codes], alpha=0.8)
    ax3.set_title("Code Dimensions", fontweight="bold")
    ax3.set_ylabel("k (bits)")
    ax3.tick_params(axis='x', rotation=45)
    
    # Redundancy (parity bits)
    redundancy = [length - dim for length, dim in zip(lengths, dimensions)]
    ax4 = axes[1, 1]
    bars4 = ax4.bar(names, redundancy, color=MODERN_PALETTE[:n_codes], alpha=0.8)
    ax4.set_title("Redundancy (Parity Bits)", fontweight="bold")
    ax4.set_ylabel("n - k (bits)")
    ax4.tick_params(axis='x', rotation=45)
    
    return fig


def plot_blockwise_operation(original_data: torch.Tensor, 
                           processed_data: torch.Tensor, 
                           block_size: int,
                           operation_name: str = "Block-wise Operation") -> plt.Figure:
    """
    Plot before and after visualization of blockwise operations.
    
    Parameters
    ----------
    original_data : torch.Tensor
        Original binary data
    processed_data : torch.Tensor
        Data after blockwise operation
    block_size : int
        Size of blocks used in operation
    operation_name : str
        Name of the operation for the title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4))
    
    # Plot original data
    ax1.bar(np.arange(len(original_data)), original_data.numpy(), color="#3498db")
    ax1.set_title("Original Binary Data")
    ax1.set_yticks([0, 1])
    ax1.grid(axis="y")
    
    # Add vertical lines to show block boundaries
    for i in range(1, len(original_data) // block_size):
        ax1.axvline(x=i * block_size - 0.5, color="gray", linestyle="--", alpha=0.7)
    
    # Plot processed data
    ax2.bar(np.arange(len(processed_data)), processed_data.numpy(), color="#f39c12")
    ax2.set_title(f"After {operation_name}")
    ax2.set_yticks([0, 1])
    ax2.grid(axis="y")
    
    # Add vertical lines to show block boundaries
    for i in range(1, len(processed_data) // block_size):
        ax2.axvline(x=i * block_size - 0.5, color="gray", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_parity_check_visualization(data: torch.Tensor,
                                  encoded_data: torch.Tensor,
                                  corrupted_data: torch.Tensor,
                                  block_size: int,
                                  error_positions: torch.Tensor,
                                  corrupted_blocks: torch.Tensor) -> plt.Figure:
    """
    Plot visualization of parity check error detection.
    
    Parameters
    ----------
    data : torch.Tensor
        Original data
    encoded_data : torch.Tensor
        Data with parity bits
    corrupted_data : torch.Tensor
        Data with introduced errors
    block_size : int
        Original block size
    error_positions : torch.Tensor
        Positions where errors were introduced
    corrupted_blocks : torch.Tensor
        Blocks after corruption for parity checking
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6))
    
    # Original data
    ax1.bar(np.arange(len(data)), data.numpy(), color="#3498db")
    ax1.set_title("Original Data")
    ax1.set_yticks([0, 1])
    ax1.grid(axis="y")
    
    # Encoded data with parity
    block_size_with_parity = block_size + 1
    num_blocks = len(encoded_data) // block_size_with_parity
    
    ax2.bar(np.arange(len(encoded_data)), encoded_data.numpy(), color="#2ecc71")
    ax2.set_title("Encoded Data with Parity Bits")
    ax2.set_yticks([0, 1])
    ax2.grid(axis="y")
    
    # Add vertical lines to show block boundaries
    for i in range(1, num_blocks):
        ax2.axvline(x=i * block_size_with_parity - 0.5, color="gray", linestyle="--", alpha=0.7)
    
    # Highlight parity bits
    for i in range(num_blocks):
        pos = (i + 1) * block_size_with_parity - 1
        ax2.scatter(pos, encoded_data[pos], color="#9b59b6", s=100, marker="o", zorder=3)
    
    # Corrupted data
    ax3.bar(np.arange(len(corrupted_data)), corrupted_data.numpy(), color="#f39c12")
    ax3.set_title("Corrupted Data with Errors")
    ax3.set_yticks([0, 1])
    ax3.grid(axis="y")
    
    # Add vertical lines to show block boundaries
    for i in range(1, num_blocks):
        ax3.axvline(x=i * block_size_with_parity - 0.5, color="gray", linestyle="--", alpha=0.7)
    
    # Highlight parity bits
    for i in range(num_blocks):
        pos = (i + 1) * block_size_with_parity - 1
        ax3.scatter(pos, corrupted_data[pos], color="#9b59b6", s=100, marker="o", zorder=3)
    
    # Highlight errors
    for pos in error_positions:
        ax3.scatter(pos, corrupted_data[pos], color="#e74c3c", s=150, marker="x", zorder=3)
    
    # Helper function to check parity
    def check_parity(block):
        return torch.sum(block) % 2 == 0
    
    # Highlight blocks with parity errors
    for i, block in enumerate(corrupted_blocks):
        if not check_parity(block):
            block_start = i * block_size_with_parity
            block_end = (i + 1) * block_size_with_parity - 1
            ax3.axvspan(block_start - 0.5, block_end + 0.5, alpha=0.2, color="#e74c3c")
    
    plt.tight_layout()
    return fig


def plot_hamming_code_visualization(original_data: torch.Tensor,
                                  encoded: torch.Tensor,
                                  corrupted: torch.Tensor,
                                  corrected: torch.Tensor,
                                  actual_error_pos: int) -> plt.Figure:
    """
    Create comprehensive visualization of Hamming code error correction.
    
    Parameters
    ----------
    original_data : torch.Tensor
        Original 4-bit data
    encoded : torch.Tensor
        7-bit Hamming encoded data
    corrupted : torch.Tensor
        Corrupted 7-bit data
    corrected : torch.Tensor
        Corrected 7-bit data
    actual_error_pos : int
        Position where error was introduced
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.patheffects as PathEffects
    
    # Define colors
    original_color = "#3498db"
    encoded_color = "#2ecc71"
    parity_color = "#e67e22"
    error_color = "#e74c3c"
    corrected_color = "#9b59b6"
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2], width_ratios=[4, 1], 
                  hspace=0.6, wspace=0.4, bottom=0.05, top=0.9, left=0.05, right=0.95)
    
    # 1. Original data visualization
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original 4-bit Data", fontsize=16, fontweight="bold")
    
    for i, bit in enumerate(original_data):
        color = original_color if bit == 1 else "white"
        ax1.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, 
                                     edgecolor="black", linewidth=1.5))
        ax1.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, 
                fontweight="bold", color="white" if bit == 1 else "black")
        ax1.text(i, -0.3, f"d{i+1}", ha="center", va="center", fontsize=12)
    
    ax1.set_xlim(-0.8, len(original_data) - 0.2)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # 2. Hamming code visualization
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("(7,4) Hamming Encoded Data", fontsize=16, fontweight="bold")
    
    for i, bit in enumerate(encoded):
        is_parity = i in [0, 1, 3]
        
        if is_parity:
            color = parity_color if bit == 1 else "white"
            label = f"p{[0,1,3].index(i)+1}"
        else:
            color = encoded_color if bit == 1 else "white"
            data_map = {2: 0, 4: 1, 5: 2, 6: 3}
            label = f"d{data_map[i]+1}"
        
        ax2.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, 
                                     edgecolor="black", linewidth=1.5))
        ax2.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, 
                fontweight="bold", color="white" if bit == 1 else "black")
        ax2.text(i, -0.3, label, ha="center", va="center", fontsize=12)
    
    # Add parity check circles
    parity_checks = [
        ([0, 2, 4, 6], "blue"),
        ([1, 2, 5, 6], "green"), 
        ([3, 4, 5, 6], "red")
    ]
    
    for positions, color in parity_checks:
        radius = 0.45 - 0.03 * parity_checks.index((positions, color))
        for pos in positions:
            if pos != positions[0]:  # Don't highlight parity bit itself
                circle = Circle((pos, 0.5), radius, fill=False, edgecolor=color, 
                              linestyle="--", linewidth=1.5, alpha=0.7)
                ax2.add_patch(circle)
    
    ax2.set_xlim(-0.8, len(encoded) - 0.2)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # 3. Corrupted and corrected data
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("Received Data (with error) and Correction", fontsize=16, fontweight="bold")
    
    # Plot corrupted data
    for i, bit in enumerate(corrupted):
        has_error = i == actual_error_pos
        is_parity = i in [0, 1, 3]
        
        if has_error:
            color = error_color
        elif is_parity:
            color = parity_color if bit == 1 else "white"
        else:
            color = encoded_color if bit == 1 else "white"
        
        ax3.add_patch(Rectangle((i - 0.4, 0), 0.8, 1, facecolor=color, 
                                     edgecolor="black", linewidth=1.5))
        ax3.text(i, 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, 
                fontweight="bold", color="white" if (bit == 1 or has_error) else "black")
        
        if has_error:
            ax3.text(i, 1.3, "✗", color=error_color, fontsize=20, ha="center", 
                    fontweight="bold", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
            ax3.text(i, 1.7, "Error", color=error_color, fontsize=12, ha="center", fontweight="bold")
    
    # Calculate and show syndrome
    s1 = (corrupted[0] + corrupted[2] + corrupted[4] + corrupted[6]) % 2
    s2 = (corrupted[1] + corrupted[2] + corrupted[5] + corrupted[6]) % 2
    s3 = (corrupted[3] + corrupted[4] + corrupted[5] + corrupted[6]) % 2
    syndrome = s1 + 2 * s2 + 4 * s3
    
    if syndrome > 0:
        ax3.text(-0.8, -1.0, f"Syndrome: (S3 S2 S1) = ({int(s3)}{int(s2)}{int(s1)}) = {syndrome}", 
                fontsize=12, fontweight="bold", 
                bbox=dict(facecolor="lightyellow", alpha=0.7, boxstyle="round,pad=0.5", edgecolor="orange"))
        ax3.text(-0.8, -1.5, f"Error position: {syndrome-1}", fontsize=12, fontweight="bold", 
                bbox=dict(facecolor="lightyellow", alpha=0.7, boxstyle="round,pad=0.5", edgecolor="orange"))
    
    # Show corrected codeword
    ax3.text(-0.8, -2.0, "Corrected codeword:", fontsize=12, fontweight="bold")
    y_offset = -2.3
    
    for i, bit in enumerate(corrected):
        was_corrected = i == actual_error_pos
        is_parity = i in [0, 1, 3]
        
        if was_corrected:
            color = corrected_color
        elif is_parity:
            color = parity_color if bit == 1 else "white"
        else:
            color = encoded_color if bit == 1 else "white"
        
        ax3.add_patch(Rectangle((i - 0.4, y_offset), 0.8, 1, facecolor=color, 
                                     edgecolor="black", linewidth=1.5))
        ax3.text(i, y_offset + 0.5, f"{int(bit)}", ha="center", va="center", fontsize=15, 
                fontweight="bold", color="white" if (bit == 1 or was_corrected) else "black")
        
        if was_corrected:
            ax3.text(i, y_offset + 1.3, "✓", color="green", fontsize=20, ha="center", 
                    fontweight="bold", path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])
    
    ax3.set_xlim(-0.8, len(corrupted) - 0.2)
    ax3.set_ylim(-2.8, 2.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    
    # 4. Information sidebar
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.set_title("Parity Check Matrix", fontsize=14, fontweight="bold")
    
    ax_info.text(0.1, 0.8, "Parity Checks:", fontsize=12, fontweight="bold")
    ax_info.text(0.1, 0.75, "P1 = d1 + d2 + d4", fontsize=10, color="blue")
    ax_info.text(0.1, 0.7, "P2 = d1 + d3 + d4", fontsize=10, color="green")
    ax_info.text(0.1, 0.65, "P3 = d2 + d3 + d4", fontsize=10, color="red")
    
    ax_info.text(0.1, 0.5, "Syndrome Lookup Table:", fontsize=12, fontweight="bold")
    syndrome_table = """000: No error
001: Error in P1
010: Error in P2
011: Error in d1
100: Error in P3
101: Error in d2
110: Error in d3
111: Error in d4"""
    ax_info.text(0.1, 0.45, syndrome_table, fontsize=9, fontfamily="monospace")
    
    ax_info.text(0.1, 0.15, "Hamming Code Properties:", fontsize=12, fontweight="bold")
    ax_info.text(0.1, 0.1, "• Can detect up to 2 errors", fontsize=10)
    ax_info.text(0.1, 0.05, "• Can correct 1 error", fontsize=10)
    ax_info.text(0.1, 0.0, "• Efficient: only 3 parity", fontsize=10)
    ax_info.text(0.1, -0.05, "  bits for 4 data bits", fontsize=10)
    
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(-0.1, 1)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_visible(False)
    
    fig.suptitle("Hamming (7,4) Code: Encoding, Error Detection, and Correction", 
                fontsize=18, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
    
    return fig


def plot_constellation_comparison(modulated_symbols: Dict[int, torch.Tensor], 
                                qam_orders: List[int],
                                title: str = "QAM Constellation Diagrams") -> plt.Figure:
    """
    Plot constellation diagrams for different QAM orders.
    
    Parameters
    ----------
    modulated_symbols : Dict[int, torch.Tensor]
        Dictionary mapping QAM order to modulated symbols
    qam_orders : List[int]
        List of QAM orders to plot
    title : str
        Main title for the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, axs = plt.subplots(1, len(qam_orders), figsize=(5 * len(qam_orders), 5))
    if len(qam_orders) == 1:
        axs = [axs]
    
    for i, order in enumerate(qam_orders):
        symbols = modulated_symbols[order].numpy().flatten()
        axs[i].scatter(symbols.real, symbols.imag, alpha=0.6, s=20)
        axs[i].set_title(f"{order}-QAM")
        axs[i].set_xlabel("In-phase")
        axs[i].set_ylabel("Quadrature")
        axs[i].grid(True, alpha=0.3)
        axs[i].set_aspect('equal')
        
        # Add constellation points for reference
        if order == 4:
            ref_points = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        elif order == 16:
            ref_points = []
            for i_val in [-3, -1, 1, 3]:
                for q_val in [-3, -1, 1, 3]:
                    ref_points.append(complex(i_val, q_val))
            ref_points = np.array(ref_points) / np.sqrt(10)
        elif order == 64:
            ref_points = []
            for i_val in range(-7, 8, 2):
                for q_val in range(-7, 8, 2):
                    ref_points.append(complex(i_val, q_val))
            ref_points = np.array(ref_points) / np.sqrt(42)
        else:
            ref_points = []
            
        if len(ref_points) > 0:
            axs[i].scatter(ref_points.real, ref_points.imag, 
                          color='red', marker='x', s=50, alpha=0.8, 
                          label='Ideal points')
            axs[i].legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_ber_vs_snr_comparison(snr_range: np.ndarray,
                              ber_results: Dict[int, List[float]],
                              qam_orders: List[int],
                              title: str = "BER vs SNR for Different QAM Orders") -> plt.Figure:
    """
    Plot BER vs SNR comparison for different QAM orders.
    
    Parameters
    ----------
    snr_range : np.ndarray
        Range of SNR values
    ber_results : Dict[int, List[float]]
        Dictionary mapping QAM order to BER results
    qam_orders : List[int]
        List of QAM orders
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(qam_orders)))
    
    for i, order in enumerate(qam_orders):
        ax.semilogy(snr_range, ber_results[order], 
                   marker='o', linewidth=2, markersize=6,
                   color=colors[i], label=f"{order}-QAM")
    
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(1e-6, 1)
    
    plt.tight_layout()
    return fig


def plot_constraint_comparison(signals: Dict[str, torch.Tensor], 
                             constrained_signals: Dict[str, np.ndarray],
                             t: np.ndarray,
                             constraint_name: str,
                             constraint_value: float) -> plt.Figure:
    """
    Plot comparison of signals before and after applying constraints.
    
    Parameters
    ----------
    signals : Dict[str, torch.Tensor]
        Original signals
    constrained_signals : Dict[str, np.ndarray]
        Signals after constraint application
    t : np.ndarray
        Time vector
    constraint_name : str
        Name of the constraint applied
    constraint_value : float
        Target value of the constraint
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    from kaira.constraints.utils import measure_signal_properties
    
    fig, axes = plt.subplots(len(signals), 2, figsize=(15, 10))
    if len(signals) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, signal) in enumerate(signals.items()):
        # Plot original signal
        axes[i, 0].plot(t, signal.squeeze().numpy(), "b-", linewidth=1.5)
        props = measure_signal_properties(signal)
        
        if constraint_name == "TotalPowerConstraint":
            axes[i, 0].set_title(f'Original {name}\nPower: {props["mean_power"]:.2f}, PAPR: {props["papr_db"]:.2f} dB')
            axes[i, 1].set_title(f"After {constraint_name}\nPower: {constraint_value:.2f}")
        elif constraint_name == "PAPRConstraint":
            axes[i, 0].set_title(f'Original {name}\nPAPR: {props["papr_db"]:.2f} dB')
            axes[i, 1].set_title(f"After {constraint_name}\nMax PAPR: {constraint_value:.2f}")
        else:
            axes[i, 0].set_title(f'Original {name}')
            axes[i, 1].set_title(f"After {constraint_name}")
            
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylabel("Amplitude")
        
        # Plot constrained signal
        axes[i, 1].plot(t, constrained_signals[name], "g-", linewidth=1.5)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylabel("Amplitude")
    
    # Set x-label only on bottom plots
    for j in range(2):
        axes[-1, j].set_xlabel("Time (s)")
    
    plt.tight_layout()
    fig.suptitle(f"Signal Comparison: {constraint_name}", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.94)
    
    return fig


def plot_signal_properties_comparison(original_signals: Dict[str, torch.Tensor],
                                    constrained_signals: Dict[str, torch.Tensor],
                                    constraint_types: List[str]) -> plt.Figure:
    """
    Plot comparison of signal properties before and after constraint application.
    
    Parameters
    ----------
    original_signals : Dict[str, torch.Tensor]
        Original signals
    constrained_signals : Dict[str, torch.Tensor]
        Constrained signals
    constraint_types : List[str]
        Types of constraints applied
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    from kaira.constraints.utils import measure_signal_properties
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Signal Properties Comparison", fontsize=16)
    
    signal_names = list(original_signals.keys())
    x_pos = np.arange(len(signal_names))
    
    # Collect properties
    orig_powers = []
    orig_paprs = []
    const_powers = []
    const_paprs = []
    
    for name in signal_names:
        orig_props = measure_signal_properties(original_signals[name])
        const_props = measure_signal_properties(constrained_signals[name])
        
        orig_powers.append(orig_props['mean_power'])
        orig_paprs.append(orig_props['papr_db'])
        const_powers.append(const_props['mean_power'])
        const_paprs.append(const_props['papr_db'])
    
    # Plot power comparison
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, orig_powers, width, label='Original', alpha=0.8, color='blue')
    axes[0, 0].bar(x_pos + width/2, const_powers, width, label='Constrained', alpha=0.8, color='green')
    axes[0, 0].set_title('Average Power Comparison')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(signal_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot PAPR comparison
    axes[0, 1].bar(x_pos - width/2, orig_paprs, width, label='Original', alpha=0.8, color='blue')
    axes[0, 1].bar(x_pos + width/2, const_paprs, width, label='Constrained', alpha=0.8, color='green')
    axes[0, 1].set_title('PAPR Comparison')
    axes[0, 1].set_ylabel('PAPR (dB)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(signal_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot power histogram
    axes[1, 0].hist([orig_powers, const_powers], bins=10, alpha=0.7, 
                    label=['Original', 'Constrained'], color=['blue', 'green'])
    axes[1, 0].set_title('Power Distribution')
    axes[1, 0].set_xlabel('Power')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot PAPR histogram
    axes[1, 1].hist([orig_paprs, const_paprs], bins=10, alpha=0.7,
                    label=['Original', 'Constrained'], color=['blue', 'green'])
    axes[1, 1].set_title('PAPR Distribution')
    axes[1, 1].set_xlabel('PAPR (dB)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_constraint_chain_effects(signals_list: List[Tuple[str, np.ndarray]], 
                                 properties_list: List[Dict],
                                 t: np.ndarray,
                                 title: str = "Constraint Chain Effects") -> plt.Figure:
    """
    Plot the effects of applying constraints in sequence.
    
    Parameters
    ----------
    signals_list : List[Tuple[str, np.ndarray]]
        List of (name, signal) tuples
    properties_list : List[Dict]
        List of signal properties for each signal
    t : np.ndarray
        Time vector
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(len(signals_list), 1, figsize=(15, 10))
    if len(signals_list) == 1:
        axes = [axes]
    
    # Plot a segment of the signal to see details
    plot_segment = slice(0, min(200, len(t)))
    
    for i, ((name, signal), props) in enumerate(zip(signals_list, properties_list)):
        axes[i].plot(t[plot_segment], signal[plot_segment], linewidth=1.5)
        axes[i].set_title(
            f"{name}\nPower: {props['mean_power']:.4f}, "
            f"PAPR: {props['papr_db']:.2f} dB, "
            f"Max Amplitude: {props['peak_amplitude']:.4f}"
        )
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel("Amplitude")
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_spectral_constraint_effects(original_spectrum: np.ndarray,
                                   constrained_spectrum: np.ndarray,
                                   mask: np.ndarray,
                                   freq: np.ndarray,
                                   title: str = "Spectral Constraint Effects") -> plt.Figure:
    """
    Plot the effects of spectral mask constraints.
    
    Parameters
    ----------
    original_spectrum : np.ndarray
        Original signal spectrum
    constrained_spectrum : np.ndarray  
        Constrained signal spectrum
    mask : np.ndarray
        Spectral mask
    freq : np.ndarray
        Frequency vector
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Scale mask for visualization
    mask_for_plot = mask * np.max(original_spectrum)
    
    ax1.semilogy(freq, original_spectrum, "b", label="Original", linewidth=1.5)
    ax1.semilogy(freq, mask_for_plot, "r--", label="Spectral Mask", linewidth=2)
    ax1.set_title("Original Signal Spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Power")
    ax1.legend()
    
    ax2.semilogy(freq, constrained_spectrum, "g", label="Constrained", linewidth=1.5)
    ax2.semilogy(freq, mask_for_plot, "r--", label="Spectral Mask", linewidth=2)
    ax2.set_title("Spectrum After Spectral Mask Constraint")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Normalized Frequency")
    ax2.set_ylabel("Power")
    ax2.legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_comprehensive_constraint_analysis(original_signal: np.ndarray,
                                         constrained_signal: np.ndarray,
                                         original_spectrum: np.ndarray,
                                         constrained_spectrum: np.ndarray,
                                         mask: np.ndarray,
                                         freq: np.ndarray,
                                         t: np.ndarray,
                                         props: Dict,
                                         plot_segment: slice) -> plt.Figure:
    """
    Create comprehensive analysis plot for constraint effects.
    
    Parameters
    ----------
    original_signal : np.ndarray
        Original signal in time domain
    constrained_signal : np.ndarray
        Constrained signal in time domain
    original_spectrum : np.ndarray
        Original signal spectrum
    constrained_spectrum : np.ndarray
        Constrained signal spectrum  
    mask : np.ndarray
        Spectral mask
    freq : np.ndarray
        Frequency vector
    t : np.ndarray
        Time vector
    props : Dict
        Signal properties
    plot_segment : slice
        Segment of signal to plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Time domain plots
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(t[plot_segment], original_signal[plot_segment], "b-", 
             label="Original", linewidth=1.5)
    ax1.plot(t[plot_segment], constrained_signal[plot_segment], "r-", 
             label="All Constraints", linewidth=1.5)
    ax1.set_title(
        f"Time Domain - Original vs. All Constraints\n"
        f"Power: {props['mean_power']:.2f}, "
        f"PAPR: {props['papr_db']:.2f} dB, "
        f"Max Amplitude: {props['peak_amplitude']:.2f}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylabel("Amplitude")
    
    # Frequency domain plots
    mask_for_plot = mask * np.max(original_spectrum)
    
    ax2 = plt.subplot(gs[1, 0])
    ax2.semilogy(freq, original_spectrum, "b", linewidth=1.5)
    ax2.set_title("Original Spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Power")
    
    ax3 = plt.subplot(gs[1, 1])
    ax3.semilogy(freq, constrained_spectrum, "r", linewidth=1.5)
    ax3.semilogy(freq, mask_for_plot, "k--", alpha=0.7, linewidth=2)
    ax3.set_title("Constrained Spectrum")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Normalized Frequency")
    
    # Amplitude distribution plot (histogram)
    ax4 = plt.subplot(gs[2, :])
    # Ensure we're using real values for histogram
    orig_real = original_signal.real if np.iscomplexobj(original_signal) else original_signal
    const_real = constrained_signal.real if np.iscomplexobj(constrained_signal) else constrained_signal
    
    ax4.hist(orig_real, bins=50, alpha=0.5, label="Original", density=True)
    ax4.hist(const_real, bins=50, alpha=0.5, label="Constrained", density=True)
    ax4.axvline(x=props["peak_amplitude"], color="r", linestyle="--", 
                label=f'Max Amplitude: {props["peak_amplitude"]:.2f}')
    ax4.set_title("Amplitude Distribution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlabel("Amplitude")
    ax4.set_ylabel("Density")
    
    plt.tight_layout()
    
    return fig


def plot_bit_error_visualization(bits: torch.Tensor, 
                               errors: torch.Tensor, 
                               received_bits: torch.Tensor,
                               title: str = "Bit Error Visualization") -> plt.Figure:
    """
    Plot visualization of bit errors.
    
    Parameters
    ----------
    bits : torch.Tensor
        Original bits
    errors : torch.Tensor  
        Error locations
    received_bits : torch.Tensor
        Received bits with errors
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    axes[0].imshow(bits.view(25, 40), cmap="binary", aspect="auto")
    axes[0].set_title("Original Bits")
    plt.colorbar(axes[0].get_images()[0], ax=axes[0])
    
    axes[1].imshow(errors.int().view(25, 40), cmap="binary", aspect="auto")
    axes[1].set_title("Error Locations") 
    plt.colorbar(axes[1].get_images()[0], ax=axes[1])
    
    axes[2].imshow(received_bits.view(25, 40), cmap="binary", aspect="auto")
    axes[2].set_title("Received Bits")
    plt.colorbar(axes[2].get_images()[0], ax=axes[2])
    
    plt.tight_layout()
    return fig


def plot_error_rate_comparison(metrics: Dict[str, float],
                             title: str = "Error Rate Comparison") -> plt.Figure:
    """
    Plot comparison of different error rate metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color=MODERN_PALETTE[:len(metrics)], alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Error Rate")
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.5f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_block_error_visualization(blocks_with_errors: torch.Tensor,
                                 error_rate: float,
                                 title: str = "Block Error Visualization") -> plt.Figure:
    """
    Plot visualization of block errors.
    
    Parameters
    ----------
    blocks_with_errors : torch.Tensor
        Binary tensor indicating blocks with errors
    error_rate : float
        Block error rate
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(blocks_with_errors.view(10, -1), cmap="binary", aspect="auto")
    ax.set_title(f"{title} (BLER = {error_rate:.3f})")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_qam_constellation_with_errors(qam_symbols: torch.Tensor,
                                     received_symbols: torch.Tensor,
                                     title: str = "QAM Constellation") -> plt.Figure:
    """
    Plot QAM constellation showing transmitted and received symbols.
    
    Parameters
    ----------
    qam_symbols : torch.Tensor
        Original transmitted symbols
    received_symbols : torch.Tensor
        Received symbols after channel
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(qam_symbols.real.flatten().numpy(), 
              qam_symbols.imag.flatten().numpy(), 
              label="Transmitted", alpha=0.7, s=30, color=MODERN_PALETTE[0])
    ax.scatter(received_symbols.real.flatten().numpy(), 
              received_symbols.imag.flatten().numpy(), 
              label="Received", alpha=0.3, s=10, color=MODERN_PALETTE[1])
    
    ax.grid(True)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_symbol_error_analysis(symbol_errors: torch.Tensor,
                             ber: float,
                             ser: float,
                             title: str = "Symbol Error Analysis") -> plt.Figure:
    """
    Plot symbol error analysis comparing BER and SER.
    
    Parameters
    ----------
    symbol_errors : torch.Tensor
        Binary tensor indicating symbol errors
    ber : float
        Bit error rate
    ser : float
        Symbol error rate
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error rate comparison
    ax1.bar(["BER", "SER"], [ber, ser], color=MODERN_PALETTE[:2], alpha=0.8)
    ax1.set_title("Error Rate Comparison")
    ax1.set_ylabel("Error Rate")
    ax1.set_ylim(0, max(ser * 1.2, 0.01))
    ax1.grid(axis="y", alpha=0.3)
    
    # Symbol error visualization
    im = ax2.imshow(symbol_errors.reshape(25, 40), cmap="binary", aspect="auto")
    ax2.set_title("Symbol Errors")
    plt.colorbar(im, ax=ax2)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_multi_qam_ber_performance(snr_range: np.ndarray,
                                 ber_results: Dict[int, List[float]],
                                 ser_results: Dict[int, List[float]],
                                 qam_orders: List[int]) -> plt.Figure:
    """
    Plot BER and SER performance for multiple QAM orders.
    
    Parameters
    ----------
    snr_range : np.ndarray
        SNR values in dB
    ber_results : Dict[int, List[float]]
        BER results for each QAM order
    ser_results : Dict[int, List[float]]
        SER results for each QAM order
    qam_orders : List[int]
        QAM modulation orders
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    markers = ["o-", "s-", "^-"]
    colors = MODERN_PALETTE[:len(qam_orders)]
    
    # BER vs SNR
    for i, (order, marker) in enumerate(zip(qam_orders, markers)):
        axes[0].semilogy(snr_range, ber_results[order], marker, 
                        color=colors[i], label=f"{order}-QAM")
    axes[0].grid(True)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[0].set_title("BER vs SNR")
    axes[0].legend()
    
    # SER vs SNR  
    for i, (order, marker) in enumerate(zip(qam_orders, markers)):
        axes[1].semilogy(snr_range, ser_results[order], marker,
                        color=colors[i], label=f"{order}-QAM")
    axes[1].grid(True)
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Symbol Error Rate (SER)")
    axes[1].set_title("SER vs SNR")
    axes[1].legend()
    
    # BER vs SER
    for i, (order, marker) in enumerate(zip(qam_orders, markers)):
        axes[2].loglog(ser_results[order], ber_results[order], marker,
                      color=colors[i], label=f"{order}-QAM")
    axes[2].grid(True)
    axes[2].set_xlabel("Symbol Error Rate (SER)")
    axes[2].set_ylabel("Bit Error Rate (BER)")
    axes[2].set_title("BER vs SER")
    axes[2].legend()
    
    plt.tight_layout()
    return fig


def plot_bler_vs_snr_analysis(snr_range: np.ndarray,
                             bler_vs_snr: Dict[int, List[float]],
                             block_sizes: List[int]) -> plt.Figure:
    """
    Plot BLER vs SNR for different block sizes.
    
    Parameters
    ----------
    snr_range : np.ndarray
        SNR values in dB
    bler_vs_snr : Dict[int, List[float]]
        BLER results for each block size
    block_sizes : List[int]
        Block sizes to compare
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = MODERN_PALETTE[:len(block_sizes)]
    
    for i, block_size in enumerate(block_sizes):
        ax.semilogy(snr_range, bler_vs_snr[block_size], "o-", 
                   color=colors[i], label=f"Block Size = {block_size}")
    
    ax.grid(True)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Block Error Rate (BLER)")
    ax.set_title("BLER vs SNR for Different Block Sizes")
    ax.legend()
    
    # Add threshold lines
    for threshold in [0.1, 0.01, 0.001]:
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.3)
        ax.text(0.5, threshold, f"BLER = {threshold}", 
               ha="left", va="bottom", alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_multiple_metrics_comparison(snr_range: np.ndarray,
                                   metrics: Dict[str, List[float]],
                                   title: str = "Error Rate Metrics vs SNR") -> plt.Figure:
    """
    Plot comparison of multiple error rate metrics vs SNR.
    
    Parameters
    ----------
    snr_range : np.ndarray
        SNR values in dB
    metrics : Dict[str, List[float]]
        Dictionary of metric names and their values vs SNR
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markers = ["o-", "s-", "^-", "x-"]
    colors = MODERN_PALETTE[:len(metrics)]
    line_styles = ["-", "--", "-.", ":"]
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        style_idx = i % len(line_styles)
        marker_style = markers[i % len(markers)]
        
        ax.semilogy(snr_range, values, marker_style[0] + line_styles[style_idx], 
                   color=colors[i], label=metric_name, linewidth=2, markersize=6)
    
    ax.grid(True)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Error Rate")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_signal_noise_comparison(t: np.ndarray,
                                original_signal: np.ndarray,
                                noisy_signals: List[Tuple[float, np.ndarray]],
                                measured_metrics: List[Dict[str, float]],
                                title: str = "Signal vs Noise Comparison") -> plt.Figure:
    """
    Plot comparison of original signal vs signals with different noise levels.
    
    Parameters
    ----------
    t : np.ndarray
        Time vector
    original_signal : np.ndarray
        Original clean signal
    noisy_signals : List[Tuple[float, np.ndarray]]
        List of (SNR_dB, noisy_signal) tuples
    measured_metrics : List[Dict[str, float]]
        List of dictionaries containing measured metrics
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Plot the original signal
    plt.subplot(len(noisy_signals) + 1, 1, 1)
    plt.plot(t, original_signal, "b-", linewidth=1.5, color=MODERN_PALETTE[0])
    plt.title("Original Signal", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylabel("Amplitude")
    plt.xlim([0, 1])
    
    # Plot each noisy signal
    for i, (snr_db, output) in enumerate(noisy_signals):
        plt.subplot(len(noisy_signals) + 1, 1, i + 2)
        plt.plot(t, output, "r-", alpha=0.8, color=MODERN_PALETTE[1])
        measured_snr = measured_metrics[i]["measured_snr_db"]
        plt.title(f"AWGN Channel (Target SNR = {snr_db} dB, Measured SNR = {measured_snr:.1f} dB)", 
                 fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylabel("Amplitude")
        if i == len(noisy_signals) - 1:
            plt.xlabel("Time (s)")
        plt.xlim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_snr_psnr_comparison(measured_metrics: List[Dict[str, float]],
                           title: str = "SNR and PSNR Analysis") -> plt.Figure:
    """
    Plot comparison of theoretical vs measured SNR and PSNR values.
    
    Parameters
    ----------
    measured_metrics : List[Dict[str, float]]
        List of dictionaries containing measured metrics
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    target_snrs = [metric["target_snr_db"] for metric in measured_metrics]
    measured_snrs = [metric["measured_snr_db"] for metric in measured_metrics]
    measured_psnrs = [metric["measured_psnr_db"] for metric in measured_metrics]
    
    # Plot SNR comparison
    ax1.plot(target_snrs, measured_snrs, "o-", linewidth=2, markersize=8, 
            color=MODERN_PALETTE[0], label="Measured SNR")
    ax1.plot(target_snrs, target_snrs, "--", linewidth=2, 
            color=MODERN_PALETTE[3], label="Theoretical (Target)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Target SNR (dB)", fontweight='bold')
    ax1.set_ylabel("Measured SNR (dB)", fontweight='bold')
    ax1.set_title("Theoretical vs. Measured SNR", fontweight='bold')
    ax1.legend()
    
    # Plot PSNR values
    ax2.plot(target_snrs, measured_psnrs, "o-", linewidth=2, markersize=8, 
            color=MODERN_PALETTE[1])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Target SNR (dB)", fontweight='bold')
    ax2.set_ylabel("PSNR (dB)", fontweight='bold')
    ax2.set_title("PSNR vs. Target SNR", fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_snr_vs_mse(snr_levels: List[float],
                   mse_values: List[float],
                   signal_power: float,
                   title: str = "SNR vs Mean Squared Error") -> plt.Figure:
    """
    Plot relationship between SNR and MSE with theoretical curve.
    
    Parameters
    ----------
    snr_levels : List[float]
        SNR values in dB
    mse_values : List[float]
        Measured MSE values
    signal_power : float
        Signal power for theoretical calculation
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(snr_levels, mse_values, "o-", linewidth=2, markersize=8, 
           color=MODERN_PALETTE[0], label="Measured MSE")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("SNR (dB)", fontweight='bold')
    ax.set_ylabel("Mean Squared Error", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_yscale("log")  # Use logarithmic scale for MSE
    
    # Add theoretical MSE curve: MSE = noise_power = signal_power / 10^(SNR/10)
    snr_range = np.linspace(-6, 21, 100)
    theoretical_mse = signal_power / np.power(10, snr_range / 10)
    ax.plot(snr_range, theoretical_mse, "--", linewidth=2, 
           color=MODERN_PALETTE[3], label="Theoretical")
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_noise_level_analysis(signal_powers: List[float],
                             noise_powers: List[float],
                             snr_levels: List[float],
                             title: str = "Noise Level Analysis") -> plt.Figure:
    """
    Plot analysis of signal power, noise power, and resulting SNR.
    
    Parameters
    ----------
    signal_powers : List[float]
        Signal power values
    noise_powers : List[float]
        Noise power values
    snr_levels : List[float]
        SNR levels in dB
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Power comparison
    x_pos = np.arange(len(snr_levels))
    width = 0.35
    
    ax1.bar(x_pos - width/2, signal_powers, width, label='Signal Power', 
           color=MODERN_PALETTE[0], alpha=0.8)
    ax1.bar(x_pos + width/2, noise_powers, width, label='Noise Power',
           color=MODERN_PALETTE[1], alpha=0.8)
    
    ax1.set_xlabel('SNR Level (dB)', fontweight='bold')
    ax1.set_ylabel('Power', fontweight='bold')
    ax1.set_title('Signal vs Noise Power', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{snr:.0f}' for snr in snr_levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # SNR vs Noise Power relationship
    ax2.plot(noise_powers, snr_levels, 'o-', linewidth=2, markersize=8,
            color=MODERN_PALETTE[2])
    ax2.set_xlabel('Noise Power', fontweight='bold')
    ax2.set_ylabel('SNR (dB)', fontweight='bold')
    ax2.set_title('SNR vs Noise Power', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    return fig


def plot_binary_channel_comparison(original_data: np.ndarray,
                                 channel_outputs: List[Tuple[str, np.ndarray, float]],
                                 segment_start: int = 0,
                                 segment_length: int = 50,
                                 title: str = "Binary Channel Effects") -> plt.Figure:
    """
    Plot comparison of binary data through different channels.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original binary data
    channel_outputs : List[Tuple[str, np.ndarray, float]]
        List of (channel_name, output_data, parameter_value) tuples
    segment_start : int
        Starting position of segment to visualize
    segment_length : int
        Length of segment to visualize  
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    
    def plot_binary_segment(ax, data, label, y_pos, erasures=None):
        """Helper function to plot binary data segment."""
        # Plot 0s and 1s
        colors = [MODERN_PALETTE[0] if b == 1 else MODERN_PALETTE[1] for b in data]
        ax.scatter(np.arange(len(data)), [y_pos] * len(data), 
                  c=colors, marker="o", s=50)
        
        # Mark erasures if provided
        if erasures is not None:
            erasure_indices = np.where(erasures)[0]
            if len(erasure_indices) > 0:
                ax.scatter(erasure_indices, [y_pos] * len(erasure_indices), 
                          facecolors="none", edgecolors="black", 
                          marker="o", s=80, linewidth=2)
        
        ax.set_ylabel(label, fontweight='bold')
        ax.set_ylim(y_pos - 0.5, y_pos + 0.5)
        ax.set_yticks([])
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        return ax
    
    # Extract segment
    segment_data = original_data[segment_start:segment_start + segment_length]
    
    fig, axes = plt.subplots(len(channel_outputs) + 1, 1, 
                           figsize=(12, (len(channel_outputs) + 1) * 1.5))
    
    # Plot original data
    plot_binary_segment(axes[0], segment_data, "Original", 0)
    
    # Plot each channel output
    for i, (channel_name, output_data, param) in enumerate(channel_outputs, 1):
        segment_output = output_data[segment_start:segment_start + segment_length]
        
        # Handle erasures for BEC
        if "BEC" in channel_name:
            erasures = segment_output == -1
            segment_output = np.where(erasures, 0.5, segment_output)
            plot_binary_segment(axes[i], segment_output, 
                              f"{channel_name} (p={param})", 0, erasures=erasures)
        else:
            plot_binary_segment(axes[i], segment_output, 
                              f"{channel_name} (p={param})", 0)
        
        axes[i].set_xlim(-1, segment_length)
        axes[i].set_xticks(np.arange(0, segment_length, 5))
    
    axes[-1].set_xlabel("Bit Position", fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_channel_error_rates(error_probs: List[float],
                            theoretical_rates: List[float],
                            observed_rates: List[float],
                            channel_names: List[str],
                            title: str = "Channel Error Rate Comparison") -> plt.Figure:
    """
    Plot comparison of theoretical vs observed error rates for multiple channels.
    
    Parameters
    ----------
    error_probs : List[float]
        Channel parameter values
    theoretical_rates : List[float]
        Theoretical error rates
    observed_rates : List[float]
        Observed error rates
    channel_names : List[str]
        Names of channels
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(10, len(channel_names) * 3))
    if len(channel_names) == 1:
        axes = [axes]
    
    colors = MODERN_PALETTE[:len(channel_names)]
    
    for i, (channel_name, color) in enumerate(zip(channel_names, colors)):
        ax = axes[i]
        
        # Plot theoretical and observed rates
        ax.plot(error_probs, theoretical_rates, "-", color=color, 
               linewidth=2, label="Theoretical")
        ax.plot(error_probs, observed_rates, "o--", color=color, 
               linewidth=2, markersize=6, label="Observed")
        
        ax.set_ylabel(f"{channel_name}\nError Rate", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if i == len(channel_names) - 1:
            ax.set_xlabel("Channel Parameter (p)", fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_transition_matrices(matrices: List[Tuple[str, np.ndarray, float]],
                           title: str = "Channel Transition Matrices") -> plt.Figure:
    """
    Plot transition matrices for binary channels.
    
    Parameters
    ----------
    matrices : List[Tuple[str, np.ndarray, float]]
        List of (channel_name, matrix, parameter) tuples
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    import seaborn as sns
    
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]
    
    for i, (channel_name, matrix, param) in enumerate(matrices):
        ax = axes[i]
        
        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", 
                   cbar=False, ax=ax, square=True)
        
        ax.set_title(f"{channel_name} (p={param})", fontweight='bold')
        ax.set_xlabel("Output", fontweight='bold')
        ax.set_ylabel("Input", fontweight='bold')
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["0", "1"])
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_channel_capacity_analysis(channel_parameters: np.ndarray,
                                  capacities: Dict[str, np.ndarray],
                                  title: str = "Binary Channel Capacity Analysis") -> plt.Figure:
    """
    Plot channel capacity vs parameter for different binary channels.
    
    Parameters
    ----------
    channel_parameters : np.ndarray
        Range of channel parameter values
    capacities : Dict[str, np.ndarray]
        Dictionary mapping channel names to capacity arrays
    title : str
        Plot title
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = MODERN_PALETTE[:len(capacities)]
    
    for i, (channel_name, capacity_values) in enumerate(capacities.items()):
        ax.plot(channel_parameters, capacity_values, "o-", 
               color=colors[i], linewidth=2, markersize=6, 
               label=channel_name)
    
    ax.set_xlabel("Channel Parameter (p)", fontweight='bold')
    ax.set_ylabel("Channel Capacity (bits)", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig
