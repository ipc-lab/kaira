"""Visualization Utilities for Channel Models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Dict, Any
from kaira.core import BaseChannel

def plot_channel_response(channel: BaseChannel, 
                          input_range: Tuple[float, float] = (-2, 2),
                          num_points: int = 100,
                          complex_input: bool = False,
                          phase_values: Optional[List[float]] = None,
                          ax: Optional[plt.Axes] = None,
                          **plot_kwargs) -> plt.Axes:
    """Plot the channel response for a range of input values.
    
    Args:
        channel (BaseChannel): Channel model to analyze
        input_range (tuple): Range of input values (min, max)
        num_points (int): Number of points to evaluate
        complex_input (bool): Whether to use complex-valued input
        phase_values (list): List of phases to use for complex inputs
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        **plot_kwargs: Additional keyword arguments for the plot
        
    Returns:
        matplotlib.axes.Axes: The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    x_values = torch.linspace(input_range[0], input_range[1], num_points)
    
    # Default plot styling
    defaults = {
        'marker': 'o',
        'markersize': 3,
        'alpha': 0.7,
        'linewidth': 1.5
    }
    # Override defaults with any provided kwargs
    for k, v in defaults.items():
        if k not in plot_kwargs:
            plot_kwargs[k] = v
            
    # Handle real-valued input
    if not complex_input:
        with torch.no_grad():
            y_values = channel(x_values).detach()
            
        ax.plot(x_values.numpy(), y_values.numpy(), **plot_kwargs)
        ax.set_xlabel('Input Amplitude')
        ax.set_ylabel('Output Amplitude')
        ax.set_title('Channel Response')
        ax.grid(True, alpha=0.3)
        
    # Handle complex-valued input
    else:
        if phase_values is None:
            phase_values = [0, np.pi/4, np.pi/2]
            
        with torch.no_grad():
            for phase in phase_values:
                complex_x = x_values * torch.exp(1j * torch.tensor(phase))
                complex_y = channel(complex_x).detach()
                
                # Plot magnitude response
                ax.plot(x_values.numpy(), 
                       torch.abs(complex_y).numpy(), 
                       label=f'Phase = {phase:.2f} rad',
                       **plot_kwargs)
        
        ax.set_xlabel('Input Magnitude')
        ax.set_ylabel('Output Magnitude')
        ax.set_title('Channel Magnitude Response')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    return ax


def plot_constellation(original: torch.Tensor, 
                       received: torch.Tensor,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot constellation diagram showing original and received symbols.
    
    Args:
        original (torch.Tensor): Original complex symbols
        received (torch.Tensor): Received complex symbols after channel
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        
    Returns:
        matplotlib.axes.Axes: The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # Convert to numpy for plotting
    orig_np = original.detach().view(-1).cpu().numpy()
    recv_np = received.detach().view(-1).cpu().numpy()
    
    # Plot original constellation points
    ax.scatter(orig_np.real, orig_np.imag, 
               c='blue', marker='o', s=50, alpha=0.7,
               label='Original')
    
    # Plot received constellation points
    ax.scatter(recv_np.real, recv_np.imag, 
               c='red', marker='x', s=30, alpha=0.5,
               label='Received')
    
    # Optional: draw lines between original and received
    for i in range(min(len(orig_np), 100)):  # Limit to 100 lines to avoid clutter
        ax.plot([orig_np[i].real, recv_np[i].real], 
                [orig_np[i].imag, recv_np[i].imag], 
                'k-', alpha=0.2)
    
    # Add labels and legend
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title('Constellation Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make the plot square with equal scales
    ax.axis('equal')
    
    return ax


def plot_impulse_response(channel: BaseChannel, 
                          length: int = 20, 
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot impulse response of a channel.
    
    Args:
        channel (BaseChannel): Channel to analyze
        length (int): Length of the impulse response to plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        
    Returns:
        matplotlib.axes.Axes: The axes containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create an impulse
    impulse = torch.zeros(length)
    impulse[0] = 1.0
    
    # Get channel response to impulse
    with torch.no_grad():
        response = channel(impulse).detach().cpu().numpy()
    
    # Plot the response
    ax.stem(np.arange(length), response, use_line_collection=True)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Channel Impulse Response')
    ax.grid(True, alpha=0.3)
    
    return ax
