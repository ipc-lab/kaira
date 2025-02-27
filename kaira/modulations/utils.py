"""Utility functions for digital modulation schemes."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import special


def binary_to_gray(num: int) -> int:
    """Convert binary number to Gray code.

    Args:
        num: Binary number to convert

    Returns:
        Gray-coded number
    """
    return num ^ (num >> 1)


def gray_to_binary(num: int) -> int:
    """Convert Gray code to binary number.

    Args:
        num: Gray-coded number to convert

    Returns:
        Binary number
    """
    mask = num
    while mask:
        mask >>= 1
        num ^= mask
    return num


def binary_array_to_gray(binary: Union[List[int], np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert binary array to Gray code.

    Args:
        binary: Binary array to convert

    Returns:
        Gray-coded array
    """
    if isinstance(binary, torch.Tensor):
        binary = binary.detach().cpu().numpy()
    elif isinstance(binary, list):
        binary = np.array(binary)

    # Convert to integers if the array contains decimals
    if binary.dtype == np.float32 or binary.dtype == np.float64:
        binary = binary.astype(np.int64)

    # Convert each number to Gray code
    gray = np.zeros_like(binary)
    for i, num in enumerate(binary):
        gray[i] = binary_to_gray(num)

    return gray


def gray_array_to_binary(gray: Union[List[int], np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert Gray-coded array to binary.

    Args:
        gray: Gray-coded array to convert

    Returns:
        Binary array
    """
    if isinstance(gray, torch.Tensor):
        gray = gray.detach().cpu().numpy()
    elif isinstance(gray, list):
        gray = np.array(gray)

    # Convert to integers if the array contains decimals
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray = gray.astype(np.int64)

    # Convert each number from Gray code to binary
    binary = np.zeros_like(gray)
    for i, num in enumerate(gray):
        binary[i] = gray_to_binary(num)

    return binary


def plot_constellation(
    constellation: torch.Tensor,
    labels: Optional[List[str]] = None,
    title: str = "Constellation Diagram",
    figsize: Tuple[int, int] = (8, 8),
    annotate: bool = True,
    grid: bool = True,
    axis_labels: bool = True,
    marker: str = "o",
    marker_size: int = 100,
    color: str = "blue",
    **kwargs,
) -> plt.Figure:
    """Plot a constellation diagram.

    Args:
        constellation: Complex-valued tensor of constellation points
        labels: Optional list of labels for each point
        title: Plot title
        figsize: Figure size (width, height) in inches
        annotate: Whether to annotate points with labels
        grid: Whether to show grid
        axis_labels: Whether to show axis labels
        marker: Marker style for constellation points
        marker_size: Marker size
        color: Marker color
        **kwargs: Additional arguments passed to matplotlib

    Returns:
        Matplotlib figure object
    """
    constellation = constellation.detach().cpu()
    fig, ax = plt.subplots(figsize=figsize, **kwargs)

    # Plot constellation points
    ax.scatter(constellation.real, constellation.imag, marker=marker, s=marker_size, color=color)

    # Add annotations if requested
    if annotate and labels is not None:
        for i, (x, y) in enumerate(zip(constellation.real, constellation.imag)):
            label = labels[i] if i < len(labels) else str(i)
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=12)

    # Add axis lines, grid, labels
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    if grid:
        ax.grid(True, alpha=0.3)

    if axis_labels:
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")

    ax.set_title(title)
    ax.set_aspect("equal")

    return fig


def calculate_theoretical_ber(
    modulation: str, snr_db: Union[float, List[float], np.ndarray, torch.Tensor]
) -> np.ndarray:
    """Calculate theoretical Bit Error Rate (BER) for common modulations.

    Args:
        modulation: Modulation scheme name ('bpsk', 'qpsk', '16qam', etc.)
        snr_db: Signal-to-noise ratio(s) in dB

    Returns:
        Theoretical BER values
    """
    # Convert to numpy array if needed
    if isinstance(snr_db, (list, torch.Tensor)):
        snr_db = np.array(snr_db)
    elif isinstance(snr_db, float):
        snr_db = np.array([snr_db])

    # Convert SNR from dB to linear scale
    snr = 10 ** (snr_db / 10)

    modulation = modulation.lower()

    if modulation == "bpsk":
        return 0.5 * special.erfc(np.sqrt(snr))
    elif modulation == "qpsk" or modulation == "4qam":
        # QPSK is equivalent to two BPSK systems
        return 0.5 * special.erfc(np.sqrt(snr / 2))
    elif modulation == "16qam":
        # Approximate BER for 16-QAM
        return 0.75 * special.erfc(np.sqrt(snr / 10))
    elif modulation == "64qam":
        # Approximate BER for 64-QAM
        return (7 / 12) * special.erfc(np.sqrt(snr / 42))
    elif modulation == "4pam":
        # BER for 4-PAM
        return 0.75 * special.erfc(np.sqrt(snr / 5))
    elif modulation == "8pam":
        # Approximate BER for 8-PAM
        return (7 / 12) * special.erfc(np.sqrt(snr / 21))
    elif modulation == "dpsk" or modulation == "dbpsk":
        # BER for DBPSK
        return 0.5 * np.exp(-snr)
    elif modulation == "dqpsk":
        # Approximate BER for DQPSK
        return special.erfc(np.sqrt(snr / 2)) - 0.25 * (special.erfc(np.sqrt(snr / 2))) ** 2
    else:
        raise ValueError(f"Modulation scheme '{modulation}' not supported for theoretical BER")


def calculate_spectral_efficiency(modulation: str) -> float:
    """Calculate spectral efficiency of a modulation scheme in bits/s/Hz.

    Args:
        modulation: Modulation scheme name

    Returns:
        Spectral efficiency in bits/s/Hz
    """
    modulation_lower = modulation.lower()

    if modulation_lower == "bpsk":
        return 1.0
    elif modulation_lower in ("qpsk", "4qam", "pi4qpsk", "oqpsk", "dqpsk"):
        return 2.0
    elif modulation_lower == "8psk":
        return 3.0
    elif modulation_lower == "16qam":
        return 4.0
    elif modulation_lower == "64qam":
        return 6.0
    elif modulation_lower == "256qam":
        return 8.0
    elif modulation_lower == "4pam":
        return 2.0
    elif modulation_lower == "8pam":
        return 3.0
    elif modulation_lower == "16pam":
        return 4.0
    else:
        # Try to extract order from name if it's a standard QAM/PSK/PAM
        for scheme in ("qam", "psk", "pam"):
            if scheme in modulation_lower:
                try:
                    order = int("".join(filter(str.isdigit, modulation_lower)))
                    return np.log2(order)
                except ValueError:
                    pass

        raise ValueError(f"Spectral efficiency for '{modulation}' not defined")
