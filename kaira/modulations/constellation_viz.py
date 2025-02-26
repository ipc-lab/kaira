"""Enhanced visualization tools for modulation constellations."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.animation as animation
from typing import Optional, Tuple, List, Dict, Union, Callable
from .base import Modulator, Demodulator
from .utils import plot_constellation


class ConstellationVisualizer:
    """Advanced visualization tool for modulation constellations.
    
    Provides enhanced visualization features for constellation diagrams
    including noise effects, decision regions, and animations.
    """
    
    def __init__(
        self, 
        modulator: Optional[Modulator] = None,
        constellation: Optional[torch.Tensor] = None,
        bit_labels: Optional[List[str]] = None,
        title: str = "Constellation Diagram",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Initialize the constellation visualizer.
        
        Args:
            modulator: Modulator object to extract constellation from
            constellation: Direct constellation points (if modulator not provided)
            bit_labels: Optional bit labels for each constellation point
            title: Plot title
            figsize: Figure size as (width, height)
        """
        if modulator is not None:
            # Extract constellation from modulator
            self.constellation = modulator.constellation.detach().cpu()
            self.bits_per_symbol = modulator.bits_per_symbol
            
            # Generate bit labels if not provided
            if hasattr(modulator, 'bit_patterns') and bit_labels is None:
                self.bit_labels = []
                for i in range(len(self.constellation)):
                    bit_str = ''.join(str(int(bit)) for bit in modulator.bit_patterns[i])
                    self.bit_labels.append(bit_str)
            else:
                self.bit_labels = bit_labels
        elif constellation is not None:
            # Use provided constellation
            if torch.is_tensor(constellation):
                self.constellation = constellation.detach().cpu()
            else:
                self.constellation = torch.tensor(constellation, dtype=torch.complex64)
            
            # Estimate bits per symbol from constellation size
            self.bits_per_symbol = int(np.log2(len(self.constellation)))
            self.bit_labels = bit_labels
        else:
            raise ValueError("Either modulator or constellation must be provided")
            
        self.title = title
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def plot_basic(self, show_grid: bool = True, show_labels: bool = True) -> plt.Figure:
        """Create a basic constellation diagram.
        
        Args:
            show_grid: Whether to show grid lines
            show_labels: Whether to show bit labels
            
        Returns:
            Matplotlib figure
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        constellation_np = self.constellation.numpy()
        
        # Plot constellation points
        self.ax.scatter(
            constellation_np.real, 
            constellation_np.imag, 
            s=100, 
            c='blue', 
            alpha=0.8, 
            edgecolors='black'
        )
        
        # Add bit labels if requested
        if show_labels and self.bit_labels is not None:
            for i, point in enumerate(constellation_np):
                label = self.bit_labels[i] if i < len(self.bit_labels) else f"{i}"
                self.ax.annotate(
                    label, 
                    (point.real, point.imag),
                    fontsize=10, 
                    ha='center', 
                    va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.3", 
                        fc="white", 
                        ec="gray", 
                        alpha=0.8
                    )
                )
        
        # Set equal aspect ratio and style
        self.ax.set_aspect('equal')
        self.ax.set_title(self.title, fontsize=14)
        self.ax.set_xlabel('In-Phase (I)', fontsize=12)
        self.ax.set_ylabel('Quadrature (Q)', fontsize=12)
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        if show_grid:
            self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust axis limits for better visualization
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        limit = max_val * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        plt.tight_layout()
        return self.fig
    
    def plot_decision_regions(self, resolution: int = 200) -> plt.Figure:
        """Plot decision regions for the constellation.
        
        Args:
            resolution: Grid resolution for decision region visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure if it doesn't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            
        constellation_np = self.constellation.numpy()
        
        # Find limits for the grid
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        limit = max_val * 1.5
        
        # Create grid of points
        x = np.linspace(-limit, limit, resolution)
        y = np.linspace(-limit, limit, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = X + 1j*Y
        
        # For each grid point, find the closest constellation point
        distances = np.abs(grid_points[:, :, np.newaxis] - constellation_np)
        closest_indices = np.argmin(distances, axis=2)
        
        # Create a colormap with distinct colors
        n_points = len(self.constellation)
        cmap = plt.cm.get_cmap('hsv', n_points)
        
        # Plot the decision regions
        im = self.ax.pcolormesh(
            X, Y, closest_indices, 
            cmap=cmap, 
            alpha=0.3, 
            shading='auto'
        )
        
        # Plot constellation points on top
        self.ax.scatter(
            constellation_np.real, 
            constellation_np.imag, 
            s=100, 
            c='black',
            alpha=1.0,
            marker='o',
            zorder=5
        )
        
        # Add bit labels
        if self.bit_labels is not None:
            for i, point in enumerate(constellation_np):
                label = self.bit_labels[i] if i < len(self.bit_labels) else f"{i}"
                self.ax.annotate(
                    label, 
                    (point.real, point.imag),
                    fontsize=10, 
                    ha='center', 
                    va='center',
                    color='white',
                    fontweight='bold',
                    bbox=dict(
                        boxstyle="round,pad=0.3", 
                        fc="black", 
                        ec="white", 
                        alpha=0.8
                    ),
                    zorder=6
                )
        
        # Style the plot
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.title} with Decision Regions", fontsize=14)
        self.ax.set_xlabel('In-Phase (I)', fontsize=12)
        self.ax.set_ylabel('Quadrature (Q)', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        
        # Set limits
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        plt.tight_layout()
        return self.fig
    
    def plot_with_noise(
        self, 
        snr_db: float = 15.0, 
        n_points: int = 1000, 
        show_density: bool = True
    ) -> plt.Figure:
        """Plot constellation with simulated noise.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            n_points: Number of noisy points to generate
            show_density: Whether to show 2D density of received points
            
        Returns:
            Matplotlib figure
        """
        # Create figure if it doesn't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            
        constellation_np = self.constellation.numpy()
        
        # Calculate noise standard deviation based on SNR
        signal_power = np.mean(np.abs(constellation_np)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)  # For complex noise
        
        # Generate random indices to pick constellation points
        random_indices = np.random.randint(0, len(constellation_np), n_points)
        original_points = constellation_np[random_indices]
        
        # Add complex Gaussian noise
        noise_real = np.random.normal(0, noise_std, n_points)
        noise_imag = np.random.normal(0, noise_std, n_points)
        noise = noise_real + 1j*noise_imag
        noisy_points = original_points + noise
        
        # Plot the original constellation
        self.ax.scatter(
            constellation_np.real, 
            constellation_np.imag, 
            s=120, 
            c='red',
            marker='*',
            alpha=1.0,
            label="Constellation Points",
            zorder=5
        )
        
        # Plot the received noisy points
        if show_density:
            # Use a hexbin plot to show density
            hb = self.ax.hexbin(
                noisy_points.real, 
                noisy_points.imag, 
                gridsize=50, 
                cmap='Blues',
                alpha=0.7,
                zorder=2
            )
            self.fig.colorbar(hb, label='Point Density')
        else:
            # Use scatter plot for individual points
            self.ax.scatter(
                noisy_points.real, 
                noisy_points.imag, 
                s=10, 
                c='blue',
                alpha=0.3,
                label="Received Points",
                zorder=2
            )
        
        # Add noise circles to visualize the noise standard deviation
        for point in constellation_np:
            circle = Circle(
                (point.real, point.imag), 
                noise_std*2, 
                fill=False, 
                linestyle='--',
                color='green',
                alpha=0.6,
                zorder=4
            )
            self.ax.add_patch(circle)
        
        # Style the plot
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.title} with Noise (SNR={snr_db} dB)", fontsize=14)
        self.ax.set_xlabel('In-Phase (I)', fontsize=12)
        self.ax.set_ylabel('Quadrature (Q)', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.legend(loc='upper right')
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        
        # Set appropriate limits
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        noise_margin = noise_std * 4
        limit = max_val + noise_margin
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        plt.tight_layout()
        return self.fig
    
    def plot_ber_estimation(self, snr_db_range: List[float] = None) -> plt.Figure:
        """Plot estimated BER for the constellation.
        
        Args:
            snr_db_range: List of SNR values in dB to evaluate
            
        Returns:
            Matplotlib figure with BER curve
        """
        if snr_db_range is None:
            snr_db_range = np.arange(0, 21, 2)
        
        # Create a new figure for BER curve
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate approximate BER for each SNR
        ber_values = []
        for snr_db in snr_db_range:
            # Monte Carlo simulation for BER estimation
            n_bits = 100000
            n_symbols = int(n_bits / self.bits_per_symbol)
            
            # Generate random symbols
            indices = np.random.randint(0, len(self.constellation), n_symbols)
            symbols = self.constellation.numpy()[indices]
            
            # Add noise
            signal_power = np.mean(np.abs(symbols)**2)
            snr_linear = 10**(snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            
            noise_real = np.random.normal(0, noise_std, n_symbols)
            noise_imag = np.random.normal(0, noise_std, n_symbols)
            noise = noise_real + 1j*noise_imag
            received = symbols + noise
            
            # Make decisions (find closest constellation point)
            decisions = np.argmin(
                np.abs(received[:, np.newaxis] - self.constellation.numpy()), 
                axis=1
            )
            
            # Count symbol errors
            symbol_errors = np.sum(decisions != indices)
            
            # Approximate bit errors (assuming Gray coding with ~1 bit error per symbol error)
            bit_errors = symbol_errors
            ber = bit_errors / n_bits
            ber_values.append(ber)
        
        # Plot BER curve
        ax.semilogy(snr_db_range, ber_values, 'o-', linewidth=2)
        ax.set_title(f"BER Estimation for {self.title}", fontsize=14)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=1e-6, top=1.0)
        
        plt.tight_layout()
        return fig
    
    def animate_phase_rotation(
        self, 
        n_frames: int = 100, 
        rotation_cycles: float = 1.0,
        interval: int = 50
    ) -> animation.Animation:
        """Create an animation showing phase rotation of the constellation.
        
        Args:
            n_frames: Number of animation frames
            rotation_cycles: Number of full rotation cycles
            interval: Frame interval in milliseconds
            
        Returns:
            Matplotlib animation object
        """
        # Create figure if it doesn't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            
        constellation_np = self.constellation.numpy()
        
        # Find limits for plot
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        limit = max_val * 1.2
        
        # Set up plot
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"{self.title} (Phase Rotation)", fontsize=14)
        self.ax.set_xlabel('In-Phase (I)', fontsize=12)
        self.ax.set_ylabel('Quadrature (Q)', fontsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.5)
        
        # Create initial scatter plot for animation
        scat = self.ax.scatter(
            constellation_np.real, 
            constellation_np.imag, 
            s=100, 
            c='blue',
            alpha=0.8, 
            edgecolors='black'
        )
        
        # Text for displaying phase angle
        phase_text = self.ax.text(
            0.02, 0.95, 
            'Phase: 0°', 
            transform=self.ax.transAxes,
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3", 
                fc="white", 
                ec="gray", 
                alpha=0.8
            )
        )
        
        # Animation update function
        def update(frame):
            # Calculate current phase rotation
            angle_rad = frame / n_frames * rotation_cycles * 2 * np.pi
            angle_deg = np.rad2deg(angle_rad) % 360
            
            # Apply phase rotation
            rotated = constellation_np * np.exp(1j * angle_rad)
            
            # Update scatter plot
            scat.set_offsets(np.column_stack([rotated.real, rotated.imag]))
            
            # Update phase text
            phase_text.set_text(f'Phase: {angle_deg:.1f}°')
            
            return scat, phase_text
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, 
            update, 
            frames=n_frames,
            interval=interval, 
            blit=True
        )
        
        plt.tight_layout()
        return ani
    
    def plot_bit_reliability(self, snr_db: float = 10.0, n_points: int = 10000) -> plt.Figure:
        """Visualize bit reliability with LLR heatmaps.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            n_points: Number of points to generate for visualization
            
        Returns:
            Matplotlib figure
        """
        constellation_np = self.constellation.numpy()
        
        # Calculate noise standard deviation
        signal_power = np.mean(np.abs(constellation_np)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        # Create a grid of points covering the constellation
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        limit = max_val * 1.5
        
        # Create grid coordinates
        resolution = 100
        x = np.linspace(-limit, limit, resolution)
        y = np.linspace(-limit, limit, resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = X + 1j*Y
        
        # Create a figure with subplots - one for each bit position
        n_bits = self.bits_per_symbol
        n_cols = min(3, n_bits)  # Maximum 3 columns
        n_rows = int(np.ceil(n_bits / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_bits == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For each bit position, calculate and plot LLR
        for bit_idx in range(n_bits):
            ax = axes[bit_idx]
            
            # Create bit value masks for constellation points
            bit_values = np.zeros(len(constellation_np))
            if hasattr(self, 'bit_labels'):
                for i, label in enumerate(self.bit_labels):
                    if i < len(constellation_np):
                        if len(label) > bit_idx:
                            bit_values[i] = int(label[bit_idx])
            else:
                # If no bit labels, use binary representation
                indices = np.arange(len(constellation_np))
                bit_values = (indices >> (n_bits - bit_idx - 1)) & 1
            
            # Points where this bit is 0 or 1
            points_bit0 = constellation_np[bit_values == 0]
            points_bit1 = constellation_np[bit_values == 1]
            
            # Calculate LLRs for each grid point
            # LLR = log(P(bit=0|y) / P(bit=1|y))
            llrs = np.zeros_like(X)
            
            for i in range(resolution):
                for j in range(resolution):
                    y_point = grid_points[i, j]
                    
                    # Calculate min distance to points where bit is 0
                    min_dist0 = np.min(np.abs(y_point - points_bit0)**2)
                    
                    # Calculate min distance to points where bit is 1
                    min_dist1 = np.min(np.abs(y_point - points_bit1)**2)
                    
                    # LLR approximation using max-log
                    llrs[i, j] = (min_dist1 - min_dist0) / noise_power
            
            # Plot LLR heatmap
            im = ax.imshow(
                llrs, 
                extent=[-limit, limit, -limit, limit], 
                origin='lower',
                cmap='coolwarm', 
                vmin=-5, 
                vmax=5
            )
            
            # Add constellation points
            ax.scatter(
                points_bit0.real, 
                points_bit0.imag, 
                c='blue', 
                marker='o', 
                s=80, 
                label='Bit=0', 
                edgecolors='black'
            )
            ax.scatter(
                points_bit1.real, 
                points_bit1.imag, 
                c='red', 
                marker='o', 
                s=80, 
                label='Bit=1', 
                edgecolors='black'
            )
            
            # Style the plot
            ax.set_title(f"Bit {bit_idx} Reliability (LLR)", fontsize=12)
            ax.set_xlabel('In-Phase (I)')
            ax.set_ylabel('Quadrature (Q)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend()
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='LLR')
        
        # Remove any unused subplots
        for i in range(n_bits, len(axes)):
            fig.delaxes(axes[i])
        
        fig.suptitle(
            f"{self.title} Bit Reliability Analysis (SNR={snr_db}dB)", 
            fontsize=16
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        return fig
    
    def plot_eye_diagram(
        self, 
        snr_db: float = 20.0,
        n_symbols: int = 1000,
        samples_per_symbol: int = 8,
        pulse_type: str = 'rrc',
        beta: float = 0.35
    ) -> plt.Figure:
        """Generate an eye diagram for the modulation.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            n_symbols: Number of symbols to simulate
            samples_per_symbol: Samples per symbol for pulse shaping
            pulse_type: Pulse shaping type ('rect', 'rrc', 'rc')
            beta: Roll-off factor for RRC/RC filters
            
        Returns:
            Matplotlib figure with eye diagram
        """
        # Generate random symbols
        constellation_np = self.constellation.numpy()
        indices = np.random.randint(0, len(constellation_np), n_symbols)
        symbols = constellation_np[indices]
        
        # Apply pulse shaping
        if pulse_type.lower() == 'rect':
            # Rectangular pulse (zeroth-order hold)
            signal = np.repeat(symbols, samples_per_symbol)
        else:
            # Create RRC or RC filter
            filter_span = 8  # Filter span in symbols
            L = filter_span * samples_per_symbol + 1
            t = np.arange(-filter_span/2, filter_span/2 + 1/samples_per_symbol, 1/samples_per_symbol)
            
            if pulse_type.lower() == 'rrc':
                # Root raised cosine filter
                h = np.zeros_like(t)
                for i, t_i in enumerate(t):
                    if abs(t_i) == 0:
                        h[i] = 1 - beta + (4 * beta / np.pi)
                    elif abs(t_i) == 1/(4*beta):
                        h[i] = (beta/np.sqrt(2)) * ((1+2/np.pi) * np.sin(np.pi/(4*beta)) + 
                                                   (1-2/np.pi) * np.cos(np.pi/(4*beta)))
                    else:
                        h[i] = (np.sin(np.pi*t_i*(1-beta)) + 
                              4*beta*t_i*np.cos(np.pi*t_i*(1+beta))) / (
                              np.pi*t_i * (1-(4*beta*t_i)**2))
            elif pulse_type.lower() == 'rc':
                # Raised cosine filter
                h = np.zeros_like(t)
                for i, t_i in enumerate(t):
                    if abs(t_i) == 0:
                        h[i] = 1
                    elif abs(t_i) == 1/(2*beta):
                        h[i] = (np.pi/(4*beta)) * np.sin(np.pi/(2*beta))
                    else:
                        h[i] = (np.sin(np.pi*t_i) / (np.pi*t_i)) * (
                              np.cos(np.pi*beta*t_i) / (1-(2*beta*t_i)**2))
            else:
                raise ValueError(f"Unsupported pulse type: {pulse_type}")
                
            # Normalize filter
            h = h / np.sqrt(np.sum(h**2))
                
            # Upsample symbols
            symbols_up = np.zeros(n_symbols * samples_per_symbol, dtype=complex)
            symbols_up[::samples_per_symbol] = symbols
            
            # Apply filter
            signal_real = np.convolve(symbols_up.real, h, 'same')
            signal_imag = np.convolve(symbols_up.imag, h, 'same')
            signal = signal_real + 1j*signal_imag
        
        # Add noise
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        noise_real = np.random.normal(0, noise_std, len(signal))
        noise_imag = np.random.normal(0, noise_std, len(signal))
        noise = noise_real + 1j*noise_imag
        
        noisy_signal = signal + noise
        
        # Create eye diagram figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Number of symbols to display in each eye trace
        symbols_per_trace = 2
        points_per_trace = symbols_per_trace * samples_per_symbol
        
        # Create eye traces
        for i in range(n_symbols - symbols_per_trace):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + points_per_trace
            
            # In-phase eye
            ax1.plot(
                np.arange(points_per_trace) / samples_per_symbol,
                noisy_signal[start_idx:end_idx].real,
                color='blue',
                alpha=0.1
            )
            
            # Quadrature eye
            ax2.plot(
                np.arange(points_per_trace) / samples_per_symbol,
                noisy_signal[start_idx:end_idx].imag,
                color='red',
                alpha=0.1
            )
        
        # Style the plots
        for ax, title, color in [(ax1, 'In-Phase', 'blue'), (ax2, 'Quadrature', 'red')]:
            ax.set_title(f"{title} Eye Diagram", fontsize=12)
            ax.set_xlabel('Symbol Time')
            ax.set_ylabel('Amplitude')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add markers for symbol timing
            for j in range(symbols_per_trace + 1):
                ax.axvline(x=j, color='green', linestyle='--', alpha=0.7)
        
        fig.suptitle(
            f"{self.title} Eye Diagram (SNR={snr_db}dB, β={beta})", 
            fontsize=14
        )
        plt.tight_layout()
        
        return fig
    
    def plot_trajectory(
        self, 
        n_symbols: int = 100, 
        snr_db: Optional[float] = None,
        show_arrows: bool = True,
        alpha: float = 0.8
    ) -> plt.Figure:
        """Plot the trajectory between consecutive symbols.
        
        Args:
            n_symbols: Number of symbols to simulate
            snr_db: Optional SNR in dB (if provided, adds noise)
            show_arrows: Whether to show directional arrows
            alpha: Transparency of trajectories
            
        Returns:
            Matplotlib figure
        """
        # Create figure if it doesn't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            
        constellation_np = self.constellation.numpy()
        
        # Generate random symbol indices
        indices = np.random.randint(0, len(constellation_np), n_symbols)
        symbols = constellation_np[indices]
        
        # Add noise if requested
        if snr_db is not None:
            signal_power = np.mean(np.abs(symbols)**2)
            snr_linear = 10**(snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power / 2)
            
            noise_real = np.random.normal(0, noise_std, n_symbols)
            noise_imag = np.random.normal(0, noise_std, n_symbols)
            noise = noise_real + 1j*noise_imag
            symbols = symbols + noise
        
        # Plot the constellation points
        self.ax.scatter(
            constellation_np.real, 
            constellation_np.imag, 
            s=100, 
            c='blue',
            marker='o',
            label='Constellation Points',
            zorder=5
        )
        
        # Plot trajectories between consecutive symbols
        for i in range(n_symbols - 1):
            start = symbols[i]
            end = symbols[i + 1]
            
            # Use different colors for trajectories
            color = plt.cm.jet(i / n_symbols)
            
            if show_arrows:
                self.ax.arrow(
                    start.real, start.imag,
                    end.real - start.real, end.imag - start.imag,
                    head_width=0.05, head_length=0.1, fc=color, ec=color,
                    alpha=alpha, zorder=2
                )
            else:
                self.ax.plot(
                    [start.real, end.real], 
                    [start.imag, end.imag],
                    color=color, 
                    alpha=alpha, 
                    zorder=2
                )
        
        # Style the plot
        self.ax.set_aspect('equal')
        title = f"{self.title} Symbol Trajectory"
        if snr_db is not None:
            title += f" (SNR={snr_db}dB)"
        self.ax.set_title(title, fontsize=14)
        self.ax.set_xlabel('In-Phase (I)', fontsize=12)
        self.ax.set_ylabel('Quadrature (Q)', fontsize=12)
        
        # Add coordinate axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate limits
        max_val = max(
            np.max(np.abs(constellation_np.real)), 
            np.max(np.abs(constellation_np.imag))
        )
        limit = max_val * 1.2
        if snr_db is not None:
            limit += np.sqrt(noise_power) * 2
            
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        plt.tight_layout()
        return self.fig
    
    def plot_spectral_efficiency(self, modulation_names: List[str] = None) -> plt.Figure:
        """Plot spectral efficiency comparison with other modulation schemes.
        
        Args:
            modulation_names: List of modulation schemes to compare against
            
        Returns:
            Matplotlib figure with spectral efficiency comparison
        """
        if modulation_names is None:
            modulation_names = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM']
            
        # Add current modulation to the list if not already there
        current = f"{2**self.bits_per_symbol}-point"
        if current not in modulation_names:
            modulation_names.append(current)
        
        # Calculate spectral efficiencies
        spectral_efficiencies = {}
        for name in modulation_names:
            if name.lower().endswith('psk') or name.lower().endswith('qam') or name.lower().endswith('point'):
                # Extract number (e.g., 16 from 16QAM)
                import re
                match = re.search(r'\d+', name)
                if match:
                    order = int(match.group())
                    bits_per_symbol = int(np.log2(order))
                    spectral_efficiencies[name] = bits_per_symbol
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        modulations = list(spectral_efficiencies.keys())
        efficiencies = [spectral_efficiencies[m] for m in modulations]
        
        colors = ['blue' if m != current else 'red' for m in modulations]
        ax.bar(modulations, efficiencies, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(efficiencies):
            ax.text(i, v+0.1, str(v), ha='center')
        
        # Style the plot
        ax.set_title("Spectral Efficiency Comparison", fontsize=14)
        ax.set_xlabel("Modulation Scheme", fontsize=12)
        ax.set_ylabel("Spectral Efficiency (bits/symbol)", fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig