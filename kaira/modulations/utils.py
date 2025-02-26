"""Utilities for modulation schemes."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List


def plot_constellation(
    symbols: Union[torch.Tensor, np.ndarray], 
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Constellation Diagram",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.7,
    grid: bool = True,
    show_axes: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot a constellation diagram.
    
    Args:
        symbols: Complex tensor or array of constellation points
        labels: Optional tensor or array of labels for each point
        title: Plot title
        figsize: Figure size as (width, height)
        alpha: Transparency of points
        grid: Whether to show grid
        show_axes: Whether to show axes
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    if torch.is_tensor(symbols):
        symbols = symbols.detach().cpu().numpy()
    
    if labels is not None and torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot constellation points
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                symbols.real[mask], symbols.imag[mask], 
                alpha=alpha, label=f"{label}"
            )
        ax.legend()
    else:
        ax.scatter(symbols.real, symbols.imag, alpha=alpha)
    
    # Add annotations for symbol values if there aren't too many
    if len(symbols) <= 64:
        for i, symbol in enumerate(symbols):
            label = f"{i}" if labels is None else f"{labels[i]}"
            ax.annotate(label, (symbol.real, symbol.imag), 
                        fontsize=8, ha='center', va='center')
    
    # Add formatting
    ax.set_title(title)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    
    # Make axes equal
    ax.axis("equal")
    
    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)
    
    if not show_axes:
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def binary_to_gray(value: int) -> int:
    """Convert binary to Gray code.
    
    Args:
        value: Binary value to convert
        
    Returns:
        Gray coded value
    """
    return value ^ (value >> 1)


def gray_to_binary(value: int) -> int:
    """Convert Gray code to binary.
    
    Args:
        value: Gray coded value to convert
        
    Returns:
        Binary value
    """
    result = value
    mask = result >> 1
    while mask != 0:
        result ^= mask
        mask >>= 1
    return result


def generate_gray_code_mapping(bits: int) -> torch.Tensor:
    """Generate a mapping table for Gray coding.
    
    Args:
        bits: Number of bits
        
    Returns:
        Tensor containing gray code mapping
    """
    n_symbols = 2**bits
    gray_map = torch.zeros(n_symbols, dtype=torch.long)
    
    for i in range(n_symbols):
        gray_map[i] = binary_to_gray(i)
        
    return gray_map


def calculate_theoretical_ber(snr_db: Union[float, List[float], np.ndarray], modulation: str) -> np.ndarray:
    """Calculate theoretical Bit Error Rate for common modulations.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        modulation: Modulation type ('bpsk', 'qpsk', '16qam', '64qam', etc.)
        
    Returns:
        Theoretical BER values
    """
    if isinstance(snr_db, (float, int)):
        snr_db = [snr_db]
    
    snr_db = np.array(snr_db)
    snr_linear = 10**(snr_db / 10.0)
    
    from scipy.special import erfc
    
    if modulation.lower() == 'bpsk':
        return 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation.lower() == 'qpsk':
        return 0.5 * erfc(np.sqrt(snr_linear/2))
    elif modulation.lower() == '16qam':
        return (3/8) * erfc(np.sqrt(snr_linear/10))
    elif modulation.lower() == '64qam':
        return (7/24) * erfc(np.sqrt(snr_linear/42))
    elif modulation.lower() == '256qam':
        return (15/64) * erfc(np.sqrt(snr_linear/170))
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
