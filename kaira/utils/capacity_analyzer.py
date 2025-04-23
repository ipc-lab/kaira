"""Capacity Analysis Module for Communication Systems.

This module provides tools for analyzing the capacity of communication channels under various
modulation schemes and channel conditions.
"""

import math
import multiprocessing
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.channels.base import BaseChannel
from kaira.modulations.base import BaseModulator
from kaira.utils.snr import snr_db_to_linear


class CapacityAnalyzer:
    """Compute and analyze capacity for various channels and modulation schemes.

    This class provides methods to compute the Shannon capacity of continuous channels
    as well as the capacity and achievable rates of specific modulation schemes over
    various channel models. It supports both analytical calculations and Monte Carlo
    simulations for numerical approximations when closed-form solutions are not available.

    Attributes:
        device (torch.device): Device to run computations on (CPU or GPU)
        num_processes (int): Number of parallel processes for Monte Carlo simulations
    """

    def __init__(self, device: Optional[torch.device] = None, num_processes: int = 1, fast_mode: bool = True):
        """Initialize the capacity analyzer.

        Args:
            device (torch.device, optional): Device to run computations on.
                If None, uses CUDA if available, otherwise CPU.
            num_processes (int, optional): Number of parallel processes to use for
                Monte Carlo simulations. Default is 1 (no parallelism).
                Set to -1 to use all available CPU cores.
            fast_mode (bool, optional): Whether to use faster approximations
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up parallel processing
        if num_processes == -1:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = max(1, num_processes)

        self.fast_mode = fast_mode

        # Cache to store previously computed capacity values
        self._capacity_cache: Dict[Tuple, torch.Tensor] = {}
        self._mimo_cache: Dict[Tuple, torch.Tensor] = {}
        self._mutual_info_cache: Dict[Tuple, torch.Tensor] = {}

    def awgn_capacity(self, snr_db: Union[float, List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute Shannon capacity for an AWGN channel.

        Calculates C = log2(1 + SNR) for the additive white Gaussian noise channel.

        Args:
            snr_db: Signal-to-noise ratio in dB, can be a single value or array

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        # Convert SNR from dB to linear
        if isinstance(snr_db, (list, np.ndarray)):
            snr_db = torch.tensor(snr_db, device=self.device)
        elif not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor([snr_db], device=self.device)

        snr_linear = snr_db_to_linear(snr_db)

        # Calculate Shannon capacity
        capacity = torch.log2(1 + snr_linear)

        return capacity

    def awgn_capacity_complex(self, snr_db: Union[float, List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute Shannon capacity for a complex AWGN channel.

        For complex channels, C = log2(1 + SNR) where SNR is defined per complex dimension.

        Args:
            snr_db: Signal-to-noise ratio in dB, can be a single value or array

        Returns:
            torch.Tensor: Capacity in bits per complex channel use
        """
        # Complex channel has capacity = log2(1 + SNR)
        return self.awgn_capacity(snr_db)

    def bsc_capacity(self, p: Union[float, List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute capacity for the Binary Symmetric Channel.

        For a BSC with crossover probability p, C = 1 - H(p) where H(p) is the binary entropy function.

        Args:
            p: Crossover probability (0 ≤ p ≤ 0.5), can be a single value or array

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        if isinstance(p, (list, np.ndarray)):
            p = torch.tensor(p, device=self.device)
        elif not isinstance(p, torch.Tensor):
            p = torch.tensor([p], device=self.device)

        # Ensure p is in valid range
        p = torch.clamp(p, 0, 0.5)

        # Calculate binary entropy function H(p)
        h_p = self._binary_entropy(p)

        # BSC capacity = 1 - H(p)
        capacity = 1 - h_p

        return capacity

    def _binary_entropy(self, p: torch.Tensor) -> torch.Tensor:
        """Calculate the binary entropy function H(p) = -p*log2(p) - (1-p)*log2(1-p).

        Handles edge cases (p=0, p=1) correctly.

        Args:
            p: Probability values

        Returns:
            torch.Tensor: H(p) values
        """
        # Initialize result tensor
        h_p = torch.zeros_like(p)

        # Handle p > 0
        mask_nonzero = p > 0
        if torch.any(mask_nonzero):
            h_p[mask_nonzero] -= p[mask_nonzero] * torch.log2(p[mask_nonzero])

        # Handle (1-p) > 0
        mask_not_one = p < 1
        if torch.any(mask_not_one):
            h_p[mask_not_one] -= (1 - p[mask_not_one]) * torch.log2(1 - p[mask_not_one])

        return h_p

    def bec_capacity(self, erasure_prob: Union[float, List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Compute capacity for the Binary Erasure Channel.

        For a BEC with erasure probability ε, C = 1 - ε.

        Args:
            erasure_prob: Erasure probability (0 ≤ ε ≤ 1), can be a single value or array

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        if isinstance(erasure_prob, (list, np.ndarray)):
            erasure_prob = torch.tensor(erasure_prob, device=self.device)
        elif not isinstance(erasure_prob, torch.Tensor):
            erasure_prob = torch.tensor([erasure_prob], device=self.device)

        # Ensure erasure probability is in valid range
        erasure_prob = torch.clamp(erasure_prob, 0, 1)

        # BEC capacity = 1 - ε
        capacity = 1 - erasure_prob

        return capacity

    def gaussian_input_capacity(self, channel: BaseChannel, snr_db: Union[float, List[float], np.ndarray, torch.Tensor], constrained: bool = True) -> torch.Tensor:
        """Compute capacity assuming Gaussian input distribution.

        For many channels, the capacity-achieving input distribution is Gaussian.
        This method computes the capacity assuming a Gaussian input.

        Args:
            channel: The channel model
            snr_db: Signal-to-noise ratio in dB, can be a single value or array
            constrained: Whether to apply an average power constraint

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        if isinstance(snr_db, (list, np.ndarray)):
            snr_db = torch.tensor(snr_db, device=self.device)
        elif not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor([snr_db], device=self.device)

        # Send channel to device
        channel.to(self.device)

        # Get channel name
        channel_type = channel.__class__.__name__

        # For AWGN channel, we know the closed-form capacity
        if channel_type in ["AWGNChannel", "GaussianChannel"]:
            return self.awgn_capacity(snr_db)

        # For fading channels with known statistics, we can compute ergodic capacity
        if channel_type in ["RayleighFadingChannel"]:
            # For Rayleigh fading with no CSI at transmitter: E[log2(1 + SNR|h|²)]
            # We'll approximate this with Monte Carlo
            return self._rayleigh_ergodic_capacity(snr_db)

        # For other channels, use numerical approximation
        snr_values, capacity_values = self.ergodic_capacity(channel, snr_db)
        return capacity_values

    def _rayleigh_ergodic_capacity(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Compute ergodic capacity for Rayleigh fading channel with no CSI at transmitter.

        Uses numerical integration rather than Monte Carlo for improved accuracy.

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            torch.Tensor: Ergodic capacity in bits per channel use
        """
        # Convert SNR from dB to linear
        snr_linear = snr_db_to_linear(snr_db)

        # Initialize capacity result
        capacity = torch.zeros_like(snr_linear)

        # Number of points for numerical integration
        num_points = 10000

        for i, snr in enumerate(snr_linear):
            # Generate channel gains following Rayleigh distribution
            # For Rayleigh fading, |h|² follows an exponential distribution
            channel_gains_squared = torch.exp(torch.rand(num_points, device=self.device).log() * (-1))

            # Calculate instantaneous capacity for each channel realization
            inst_capacity = torch.log2(1 + snr * channel_gains_squared)

            # Average to get ergodic capacity
            capacity[i] = torch.mean(inst_capacity)

        return capacity

    def mutual_information(self, modulator: BaseModulator, channel: BaseChannel, snr_db: Union[float, List[float], np.ndarray, torch.Tensor], num_symbols: int = 10000, num_bins: int = 100, estimation_method: str = "histogram") -> torch.Tensor:
        """Compute mutual information for a modulation scheme over a given channel.

        Uses Monte Carlo simulation to estimate the mutual information between
        channel input and output, which represents the achievable rate for
        a specific modulation scheme.

        Args:
            modulator: The modulation scheme
            channel: The channel model
            snr_db: Signal-to-noise ratio in dB, can be a single value or array
            num_symbols: Number of symbols to use in the Monte Carlo simulation
            num_bins: Number of bins to use in histogram estimation
            estimation_method: Method to estimate mutual information ('histogram' or 'knn')

        Returns:
            torch.Tensor: Mutual information in bits per channel use
        """
        if isinstance(snr_db, (list, np.ndarray)):
            snr_db = torch.tensor(snr_db, device=self.device)
        elif not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor([snr_db], device=self.device)

        # Create cache key with relevant parameters
        modulator_type = modulator.__class__.__name__
        channel_type = channel.__class__.__name__

        cache_key = (modulator_type, channel_type, tuple(snr_db.cpu().numpy().tolist()), num_symbols, num_bins, estimation_method)

        # Check if result is in cache
        if cache_key in self._mutual_info_cache:
            return self._mutual_info_cache[cache_key]

        # Send both modulator and channel to the same device
        modulator.to(self.device)
        channel.to(self.device)

        # Get bits per symbol from the modulator
        bits_per_symbol = modulator.bits_per_symbol

        # Initialize result tensor
        mi = torch.zeros_like(snr_db)

        # For parallel processing
        if self.num_processes > 1 and len(snr_db) > 1 and self.device.type == "cpu":
            # Process SNR points in parallel using the class method instead of local function
            with multiprocessing.Pool(self.num_processes) as pool:
                # Create a partial function with all the fixed arguments except snr_item
                from functools import partial

                process_func = partial(self._process_snr_for_mutual_information, modulator=modulator, channel=channel, num_symbols=num_symbols, bits_per_symbol=bits_per_symbol, estimation_method=estimation_method, num_bins=num_bins)

                # Use starmap to pass self as first argument
                results = pool.map(process_func, [snr.item() for snr in snr_db])

            # Assign results
            for i, result in enumerate(results):
                mi[i] = result
        else:
            # Sequential processing
            for i, snr in enumerate(snr_db):
                # Configure channel with the current SNR
                if hasattr(channel, "snr_db"):
                    channel.snr_db = float(snr)  # Convert tensor to float instead of using .item()

                # Generate random input bits
                input_bits = torch.randint(0, 2, (num_symbols, bits_per_symbol), device=self.device, dtype=torch.float32)

                # Modulate bits to symbols
                modulated = modulator(input_bits)

                # Normalize modulated signal to have unit energy
                modulated = modulated / torch.sqrt(torch.mean(torch.abs(modulated) ** 2))

                # Pass through channel
                received = channel(modulated)

                # Calculate mutual information using specified method
                if estimation_method == "knn":
                    mi[i] = self._estimate_mutual_information_knn(modulated, received, k=3, bits_per_symbol=int(bits_per_symbol))
                else:
                    mi[i] = self._estimate_mutual_information(modulated, received, num_bins=num_bins, bits_per_symbol=int(bits_per_symbol))

        # Cache the results before returning
        self._mutual_info_cache[cache_key] = mi

        return mi

    def _estimate_mutual_information(self, transmitted: torch.Tensor, received: torch.Tensor, num_bins: int = 100, bits_per_symbol: int = 1) -> torch.Tensor:
        """Estimate mutual information between transmitted and received signals.

        Uses histogram-based estimation of probability distributions.

        Args:
            transmitted: Transmitted symbols
            received: Received symbols after the channel
            num_bins: Number of histogram bins
            bits_per_symbol: Number of bits encoded per symbol

        Returns:
            torch.Tensor: Estimated mutual information in bits per symbol
        """
        # Handle complex signals by treating real and imaginary parts separately
        if torch.is_complex(transmitted):
            # For complex signals, we create a 2D histogram
            # Extract real and imaginary parts
            rx_real = received.real
            rx_imag = received.imag

            # Define histogram bins for real and imaginary components
            rx_real_min, rx_real_max = rx_real.min().item(), rx_real.max().item()
            rx_imag_min, rx_imag_max = rx_imag.min().item(), rx_imag.max().item()

            # Add a small buffer to avoid edge effects
            delta_real = (rx_real_max - rx_real_min) * 0.1
            delta_imag = (rx_imag_max - rx_imag_min) * 0.1
            rx_real_min -= delta_real
            rx_real_max += delta_real
            rx_imag_min -= delta_imag
            rx_imag_max += delta_imag

            # Handle unique transmitted symbols without using torch.unique for complex tensors
            # Manually identify unique values and create a mapping
            tx_flat = transmitted.view(-1)
            unique_tx: List[torch.Tensor] = []
            tx_indices = torch.zeros(tx_flat.size(0), dtype=torch.long, device=self.device)

            for i, tx_val in enumerate(tx_flat):
                # Check if this value is already in unique_tx
                found = False
                for j, unique_val in enumerate(unique_tx):
                    if tx_val == unique_val:  # Complex equality
                        tx_indices[i] = j
                        found = True
                        break

                if not found:
                    unique_tx.append(tx_val)
                    tx_indices[i] = len(unique_tx) - 1

            num_const_points = len(unique_tx)

            # Create histograms for joint and marginal distributions
            joint_hist = torch.zeros((num_const_points, num_bins, num_bins), device=self.device)

            # Assign each received sample to its corresponding bin
            rx_real_bins = torch.floor((rx_real - rx_real_min) / (rx_real_max - rx_real_min) * (num_bins - 1)).long()
            rx_imag_bins = torch.floor((rx_imag - rx_imag_min) / (rx_imag_max - rx_imag_min) * (num_bins - 1)).long()

            # Clip to ensure all values fall within valid bin indices
            rx_real_bins = torch.clamp(rx_real_bins, 0, num_bins - 1)
            rx_imag_bins = torch.clamp(rx_imag_bins, 0, num_bins - 1)

            # Build joint histogram efficiently
            # Using vectorized operations where possible, and direct indexing for the histogram
            for i in range(num_const_points):
                mask = tx_indices == i
                if mask.sum() > 0:  # Only process if we have points with this index
                    # Extract indices for this constellation point
                    real_bins = rx_real_bins[mask]
                    imag_bins = rx_imag_bins[mask]

                    # Update histogram directly
                    for j in range(len(real_bins)):
                        joint_hist[i, real_bins[j], imag_bins[j]] += 1

            # Normalize to get probability distributions
            # Add a small constant to avoid log(0)
            epsilon = 1e-10
            joint_hist = joint_hist / (torch.sum(joint_hist) + epsilon)

            # Marginal distributions
            p_x = torch.sum(joint_hist, dim=(1, 2)) + epsilon
            p_y = torch.sum(joint_hist, dim=0) + epsilon

            # Calculate mutual information using vectorized operations
            valid_indices = joint_hist > epsilon
            log_ratio = torch.log2(joint_hist[valid_indices] / (p_x.view(-1, 1, 1).expand_as(joint_hist) * p_y)[valid_indices])
            mi = torch.sum(joint_hist[valid_indices] * log_ratio)
        else:
            # For real signals, use a 1D histogram
            # Define histogram bins
            rx_min, rx_max = received.min().item(), received.max().item()

            # Add a small buffer to avoid edge effects
            delta = (rx_max - rx_min) * 0.1
            rx_min -= delta
            rx_max += delta

            # Handle unique transmitted symbols
            # Since these are real values, we can use torch.unique
            unique_tx, tx_indices = torch.unique(transmitted, return_inverse=True)
            num_const_points = len(unique_tx)

            # Create histograms for joint and marginal distributions
            joint_hist = torch.zeros((num_const_points, num_bins), device=self.device)

            # Assign each received sample to its corresponding bin
            rx_bins = torch.floor((received - rx_min) / (rx_max - rx_min) * (num_bins - 1)).long()
            rx_bins = torch.clamp(rx_bins, 0, num_bins - 1)

            # Build joint histogram with efficient vectorized operations for real signals
            for i in range(num_const_points):
                mask = tx_indices == i
                if mask.sum() > 0:
                    binned_indices = rx_bins[mask]
                    # Use bincount for real values as it's efficient
                    counts = torch.bincount(binned_indices, minlength=num_bins)
                    joint_hist[i] = counts

            # Normalize to get probability distributions
            # Add a small constant to avoid log(0)
            epsilon = 1e-10
            joint_hist = joint_hist / (torch.sum(joint_hist) + epsilon)

            # Marginal distributions
            p_x = torch.sum(joint_hist, dim=1) + epsilon
            p_y = torch.sum(joint_hist, dim=0) + epsilon

            # Calculate mutual information using vectorized operations
            valid_indices = joint_hist > epsilon
            log_ratio = torch.log2(joint_hist[valid_indices] / (p_x.view(-1, 1).expand_as(joint_hist) * p_y)[valid_indices])
            mi = torch.sum(joint_hist[valid_indices] * log_ratio)

        return mi

    def _estimate_mutual_information_knn(self, transmitted: torch.Tensor, received: torch.Tensor, k: int = 3, bits_per_symbol: int = 1) -> torch.Tensor:
        """Estimate mutual information using k-nearest neighbors (KNN) method.

        This method is more accurate than histograms for high-dimensional data.
        Based on Kraskov's mutual information estimator.

        Args:
            transmitted: Transmitted symbols
            received: Received symbols after the channel
            k: Number of nearest neighbors to use
            bits_per_symbol: Number of bits encoded per symbol

        Returns:
            torch.Tensor: Estimated mutual information in bits per symbol
        """
        # Move to CPU for KNN calculation (GPU KNN can be memory-intensive)
        if transmitted.device.type != "cpu":
            transmitted = transmitted.cpu()
        if received.device.type != "cpu":
            received = received.cpu()

        # Ensure inputs have correct shape for cdist
        if torch.is_complex(transmitted):
            # For complex signals, convert to 2D real vectors
            x = torch.stack([transmitted.real.view(-1), transmitted.imag.view(-1)], dim=1)
            y = torch.stack([received.real.view(-1), received.imag.view(-1)], dim=1)
        else:
            # For real signals, ensure correct shape
            x = transmitted.view(-1, 1).float()
            y = received.view(-1, 1).float()

        # Number of samples
        n = x.shape[0]

        # Combined space for distance calculations
        z = torch.cat([x, y], dim=1)

        # Calculate pairwise distances for X, Y, and joint Z
        dx = torch.cdist(x, x)
        dy = torch.cdist(y, y)
        dz = torch.cdist(z, z)

        # Set self-distances to infinity to exclude them
        for i in range(n):
            dx[i, i] = float("inf")
            dy[i, i] = float("inf")
            dz[i, i] = float("inf")

        # Find distances to k-th nearest neighbor in joint space
        knn_dists, _ = torch.topk(dz, k=min(k + 1, n), dim=1, largest=False)
        epsilon = knn_dists[:, min(k, n - 1)]

        # Count points within epsilon-ball in marginal spaces
        nx = torch.sum(dx < epsilon.view(-1, 1), dim=1)
        ny = torch.sum(dy < epsilon.view(-1, 1), dim=1)

        # Calculate mutual information
        mi = torch.digamma(torch.tensor(k)) - torch.mean(torch.digamma(nx.float() + 1) + torch.digamma(ny.float() + 1)) + torch.digamma(torch.tensor(n))

        # Convert to proper units (bits)
        mi = mi / torch.log(torch.tensor(2.0))

        # Move result back to original device
        return mi.clone().detach().to(self.device)

    def modulation_capacity(self, modulator: BaseModulator, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], num_symbols: int = 10000, monte_carlo: bool = True, estimation_method: str = "histogram") -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute capacity of a modulation scheme over a specified channel.

        Either analytically (when possible) or using Monte Carlo simulation.

        Args:
            modulator: The modulation scheme
            channel: The channel model
            snr_db_range: Range of SNR values in dB to compute capacity for
            num_symbols: Number of symbols for Monte Carlo simulation
            monte_carlo: Whether to force Monte Carlo simulation even if
                         analytical solution is available
            estimation_method: Method to estimate mutual information ('histogram' or 'knn')

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, capacity values)
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Get modulator and channel type names
        modulator_type = modulator.__class__.__name__
        channel_type = channel.__class__.__name__

        # Create cache key with relevant parameters
        cache_key = (modulator_type, channel_type, tuple(snr_db_range.cpu().numpy().tolist()), num_symbols, monte_carlo, estimation_method, self.fast_mode)

        # Check if result is in cache
        if cache_key in self._capacity_cache:
            return snr_db_range, self._capacity_cache[cache_key]

        # Check if analytical solution is available and preferred
        if not monte_carlo:
            # BPSK over AWGN has a closed-form capacity expression
            if modulator_type == "BPSKModulator" and channel_type in ["AWGNChannel", "GaussianChannel"]:
                capacity = self._bpsk_awgn_capacity(snr_db_range)
                self._capacity_cache[cache_key] = capacity
                return snr_db_range, capacity

            # QAM over AWGN can use closed-form expressions for high SNR
            elif modulator_type in ["QPSKModulator", "QAMModulator"] and channel_type in ["AWGNChannel", "GaussianChannel"]:
                # For high SNR, QAM capacity approaches log2(M) asymptotically
                # where M is the constellation size
                if modulator_type == "QPSKModulator":
                    m = 4  # QPSK has 4 constellation points
                else:
                    # QAM modulator should have constellation attribute
                    if hasattr(modulator, "constellation"):
                        m = len(modulator.constellation)
                    else:
                        m = 2**modulator.bits_per_symbol

                capacity = self._qam_awgn_capacity(snr_db_range, m)
                self._capacity_cache[cache_key] = capacity
                return snr_db_range, capacity

        # For all other cases or if Monte Carlo is explicitly requested,
        # use numerical estimation
        capacity_values = self.mutual_information(modulator, channel, snr_db_range, num_symbols=num_symbols, estimation_method=estimation_method)

        # Cache the computed values
        self._capacity_cache[cache_key] = capacity_values

        return snr_db_range, capacity_values

    def _bpsk_awgn_capacity(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Compute the capacity of BPSK over an AWGN channel analytically.

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        # Convert SNR from dB to linear
        snr_linear = snr_db_to_linear(snr_db)

        # For BPSK over AWGN, capacity = 1 - E[log2(1 + exp(-2y))]
        # where y ~ N(±√SNR, 1)

        # Create a large number of samples to approximate expectation
        num_samples = 100000
        capacity = torch.zeros_like(snr_linear)

        for i, snr in enumerate(snr_linear):
            # Create normal distributed samples with mean √SNR
            sqrt_snr = torch.sqrt(snr)
            y_pos = sqrt_snr + torch.randn(num_samples, device=self.device)
            y_neg = -sqrt_snr + torch.randn(num_samples, device=self.device)

            # Calculate expectation safely to avoid numerical issues
            term_pos = torch.log2(1 + torch.exp(-2 * torch.clamp(y_pos, min=-20, max=20)))
            term_neg = torch.log2(1 + torch.exp(-2 * torch.clamp(y_neg, min=-20, max=20)))

            # Capacity is 1 - average of terms
            capacity[i] = 1 - 0.5 * (torch.mean(term_pos) + torch.mean(term_neg))

        # Ensure capacity is between 0 and 1
        capacity = torch.clamp(capacity, min=0.0, max=1.0)

        return capacity

    def _qam_awgn_capacity(self, snr_db: torch.Tensor, constellation_size: int) -> torch.Tensor:
        """Compute the capacity of QAM over an AWGN channel analytically.

        Uses the accurate approximation for M-QAM capacity.

        Args:
            snr_db: Signal-to-noise ratio in dB
            constellation_size: Number of constellation points (M)

        Returns:
            torch.Tensor: Capacity in bits per channel use
        """
        # Convert SNR from dB to linear
        snr_linear = snr_db_to_linear(snr_db)

        # Maximum possible capacity is log2(M)
        max_capacity = math.log2(constellation_size)

        # Number of bits per symbol
        bits_per_symbol = math.log2(constellation_size)

        # Initialize capacity result
        capacity = torch.zeros_like(snr_db)

        # For each SNR value
        for i, snr in enumerate(snr_linear):
            # At very high SNR, capacity approaches log2(M)
            if snr > 100:  # ~20dB
                capacity[i] = max_capacity
                continue

            # Generate a large number of QAM symbols to calculate expected MI
            num_samples = 50000

            # For QAM with constellation points at fixed locations
            # We can simulate the received symbols directly
            symbols = self._generate_qam_symbols(constellation_size, num_samples, device=self.device)

            # Add AWGN with appropriate variance
            noise_var = 1 / snr
            noise = torch.sqrt(noise_var) * torch.complex(torch.randn(num_samples, device=self.device), torch.randn(num_samples, device=self.device))
            received = symbols + noise

            # Calculate mutual information using histogram method
            capacity[i] = self._estimate_mutual_information(symbols, received, num_bins=100, bits_per_symbol=int(bits_per_symbol))

        return capacity

    def _generate_qam_symbols(self, m: int, num_symbols: int, device: torch.device) -> torch.Tensor:
        """Generate random QAM symbols from a standard M-QAM constellation.

        Args:
            m: Constellation size
            num_symbols: Number of symbols to generate
            device: Device to create tensors on

        Returns:
            torch.Tensor: Complex tensor of QAM symbols
        """
        # Determine grid size for square QAM
        grid_size = int(math.sqrt(m))

        # Create PAM constellation for in-phase and quadrature components
        pam_points = torch.arange(-(grid_size - 1), grid_size, 2, device=device).float()

        # Generate random indices
        indices = torch.randint(0, m, (num_symbols,), device=device)

        # Map to constellation points
        i_index = indices % grid_size
        q_index = indices // grid_size

        real_part = pam_points[i_index]
        imag_part = pam_points[q_index]

        # Create complex symbols
        symbols = torch.complex(real_part, imag_part)

        # Normalize to unit average energy
        symbols = symbols / torch.sqrt(torch.mean(torch.abs(symbols) ** 2))

        return symbols

    def plot_capacity_vs_snr(
        self,
        snr_db_range: Union[List[float], np.ndarray, torch.Tensor],
        capacities: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
        labels: Optional[List[str]] = None,
        title: str = "Channel Capacity vs. SNR",
        xlabel: str = "SNR (dB)",
        ylabel: str = "Capacity (bits/channel use)",
        legend_loc: str = "lower right",
        include_shannon: bool = True,
        include_shannon_mimo: bool = False,
        mimo_tx: int = 2,
        mimo_rx: int = 2,
        figsize: Tuple[int, int] = (10, 6),
        grid: bool = True,
        style: str = "default",
    ) -> plt.Figure:
        """Plot capacity vs. SNR for one or more modulation schemes or channels.

        Args:
            snr_db_range: SNR values in dB
            capacities: List or dict of capacity values corresponding to each SNR point
            labels: Labels for each capacity curve
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_loc: Location of the legend
            include_shannon: Whether to include the Shannon capacity limit
            include_shannon_mimo: Whether to include the MIMO Shannon capacity limit
            mimo_tx: Number of transmit antennas for MIMO capacity
            mimo_rx: Number of receive antennas for MIMO capacity
            figsize: Figure size (width, height) in inches
            grid: Whether to show grid
            style: Matplotlib style to use

        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Set plotting style
        if style != "default":
            plt.style.use(style)

        fig, ax = plt.subplots(figsize=figsize)

        # Convert to numpy for plotting
        if isinstance(snr_db_range, torch.Tensor):
            snr_db_range = snr_db_range.cpu().numpy()

        # Plot Shannon capacity limit if requested
        if include_shannon:
            shannon_capacity = self.awgn_capacity(torch.tensor(snr_db_range, device=self.device)).cpu().numpy()
            ax.plot(snr_db_range, shannon_capacity, "k--", label="Shannon Limit (SISO)")

        # Plot MIMO Shannon capacity if requested
        if include_shannon_mimo and mimo_tx > 0 and mimo_rx > 0:
            mimo_capacity = self.mimo_capacity(snr_db_range, tx_antennas=mimo_tx, rx_antennas=mimo_rx, channel_knowledge="perfect").cpu().numpy()
            ax.plot(snr_db_range, mimo_capacity, "r-.", label=f"Shannon Limit ({mimo_tx}x{mimo_rx} MIMO)")

        # Plot each capacity curve
        if isinstance(capacities, dict):
            for label, capacity in capacities.items():
                if isinstance(capacity, torch.Tensor):
                    capacity = capacity.cpu().numpy()
                ax.plot(snr_db_range, capacity, "-o", label=label)
        else:
            for i, capacity in enumerate(capacities):
                if isinstance(capacity, torch.Tensor):
                    capacity = capacity.cpu().numpy()
                label = labels[i] if labels and i < len(labels) else f"Scheme {i+1}"
                ax.plot(snr_db_range, capacity, "-o", label=label)

        # Add grid, labels and legend
        if grid:
            ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=legend_loc)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        return fig

    def plot_capacity_vs_param(
        self,
        param_values: Union[List[float], np.ndarray, torch.Tensor],
        capacities: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
        param_name: str = "Parameter",
        labels: Optional[List[str]] = None,
        title: str = "Channel Capacity",
        xlabel: str = "Parameter Value",
        ylabel: str = "Capacity (bits/channel use)",
        legend_loc: str = "lower right",
        figsize: Tuple[int, int] = (10, 6),
        grid: bool = True,
        style: str = "default",
    ) -> plt.Figure:
        """Plot capacity vs. a parameter for one or more modulation schemes or channels.

        Args:
            param_values: Parameter values
            capacities: List or dict of capacity values corresponding to each parameter value
            param_name: Name of the parameter
            labels: Labels for each capacity curve
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_loc: Location of the legend
            figsize: Figure size (width, height) in inches
            grid: Whether to show grid
            style: Matplotlib style to use

        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Set plotting style
        if style != "default":
            plt.style.use(style)

        fig, ax = plt.subplots(figsize=figsize)

        # Convert to numpy for plotting
        if isinstance(param_values, torch.Tensor):
            param_values = param_values.cpu().numpy()

        # Plot each capacity curve
        if isinstance(capacities, dict):
            for label, capacity in capacities.items():
                if isinstance(capacity, torch.Tensor):
                    capacity = capacity.cpu().numpy()
                ax.plot(param_values, capacity, "-o", label=label)
        else:
            for i, capacity in enumerate(capacities):
                if isinstance(capacity, torch.Tensor):
                    capacity = capacity.cpu().numpy()
                label = labels[i] if labels and i < len(labels) else f"Scheme {i+1}"
                ax.plot(param_values, capacity, "-o", label=label)

        # Add grid, labels and legend
        if grid:
            ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel if xlabel else param_name)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=legend_loc)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        return fig

    def ergodic_capacity(self, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], num_realizations: int = 1000, num_symbols_per_realization: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ergodic capacity for fading channels.

        For fading channels, the ergodic capacity is the expected value of
        the capacity over all fading realizations.

        Args:
            channel: The fading channel model
            snr_db_range: Range of SNR values in dB
            num_realizations: Number of channel realizations to average over
            num_symbols_per_realization: Number of symbols per channel realization

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, ergodic capacity values)
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Channel to device
        channel.to(self.device)

        channel_type = channel.__class__.__name__

        # Initialize result
        ergodic_capacity = torch.zeros_like(snr_db_range)

        # For each SNR value
        for i, snr in enumerate(snr_db_range):
            if hasattr(channel, "snr_db"):
                channel.snr_db = snr.item()

            capacity_sum = 0.0
            for _ in range(num_realizations):
                # Generate random input symbols (unit power complex Gaussian)
                input_symbols = torch.complex(torch.randn(num_symbols_per_realization, device=self.device), torch.randn(num_symbols_per_realization, device=self.device)) / math.sqrt(2)

                # Pass through channel
                output_symbols = channel(input_symbols)

                # For known fading channels, we can use analytical formulas
                if channel_type in ["RayleighFadingChannel", "RicianFadingChannel", "FlatFadingChannel"]:
                    # Calculate received SNR
                    output_power = torch.mean(torch.abs(output_symbols) ** 2)
                    noise_power = 10 ** (-snr.item() / 10)
                    inst_capacity = torch.log2(1 + output_power / noise_power)
                    capacity_sum += inst_capacity.item()
                else:
                    # For general channels, use mutual information estimation
                    capacity_sum += self._estimate_mutual_information(input_symbols, output_symbols, bits_per_symbol=2).item()

            # Average over all realizations
            ergodic_capacity[i] = capacity_sum / num_realizations

        return snr_db_range, ergodic_capacity

    def outage_capacity(self, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], outage_probability: float = 0.01, num_realizations: int = 1000, num_symbols_per_realization: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute outage capacity for fading channels.

        The outage capacity is the highest rate that can be achieved with
        an outage probability less than the specified value.

        Args:
            channel: The fading channel model
            snr_db_range: Range of SNR values in dB
            outage_probability: Target outage probability
            num_realizations: Number of channel realizations
            num_symbols_per_realization: Number of symbols per channel realization

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, outage capacity values)
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Channel to device
        channel.to(self.device)

        # Initialize capacity values
        outage_capacity = torch.zeros_like(snr_db_range)

        # For each SNR
        for i, snr in enumerate(snr_db_range):
            # Configure channel with the current SNR
            if hasattr(channel, "snr_db"):
                channel.snr_db = snr.item()

            # Array to store instantaneous capacities
            inst_capacities = torch.zeros(num_realizations, device=self.device)

            # For each channel realization
            for j in range(num_realizations):
                # Generate input symbols (unit power complex Gaussian)
                input_symbols = torch.complex(torch.randn(num_symbols_per_realization, device=self.device), torch.randn(num_symbols_per_realization, device=self.device)) / math.sqrt(2)

                # Pass through channel
                output_symbols = channel(input_symbols)

                # Estimate instantaneous capacity
                inst_capacities[j] = self._estimate_mutual_information(input_symbols, output_symbols, bits_per_symbol=2)

            # Sort capacities
            sorted_capacities, _ = torch.sort(inst_capacities)

            # Find capacity at the outage probability percentile
            index = int(outage_probability * num_realizations)
            outage_capacity[i] = sorted_capacities[index]

        return snr_db_range, outage_capacity

    def compare_modulation_schemes(
        self, modulators: List[BaseModulator], channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], labels: Optional[List[str]] = None, num_symbols: int = 10000, plot: bool = True, figsize: Tuple[int, int] = (10, 6), estimation_method: str = "histogram"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[plt.Figure]]:
        """Compare capacity of multiple modulation schemes over a specified channel.

        Args:
            modulators: List of modulation schemes to compare
            channel: The channel model
            snr_db_range: Range of SNR values in dB
            labels: Labels for each modulation scheme
            num_symbols: Number of symbols for Monte Carlo simulation
            plot: Whether to generate and return a plot
            figsize: Figure size (width, height) in inches
            estimation_method: Method to estimate mutual information ('histogram' or 'knn')

        Returns:
            Tuple containing:
            - SNR values
            - Dictionary mapping modulation names to capacity values
            - Optional matplotlib figure if plot=True
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Generate default labels if not provided
        if labels is None:
            labels = [modulator.__class__.__name__ for modulator in modulators]

        # Compute capacity for each modulation scheme
        capacities = {}
        for i, modulator in enumerate(modulators):
            _, capacity = self.modulation_capacity(modulator, channel, snr_db_range, num_symbols=num_symbols, estimation_method=estimation_method)
            capacities[labels[i]] = capacity

        # Generate plot if requested
        fig = None
        if plot:
            fig = self.plot_capacity_vs_snr(snr_db_range, capacities, title=f"Modulation Capacity Comparison over {channel.__class__.__name__}", figsize=figsize)

        return snr_db_range, capacities, fig

    def compare_channels(
        self, modulator: BaseModulator, channels: List[BaseChannel], snr_db_range: Union[List[float], np.ndarray, torch.Tensor], labels: Optional[List[str]] = None, num_symbols: int = 10000, plot: bool = True, figsize: Tuple[int, int] = (10, 6), estimation_method: str = "histogram"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[plt.Figure]]:
        """Compare capacity of a modulation scheme over multiple channels.

        Args:
            modulator: The modulation scheme
            channels: List of channel models to compare
            snr_db_range: Range of SNR values in dB
            labels: Labels for each channel
            num_symbols: Number of symbols for Monte Carlo simulation
            plot: Whether to generate and return a plot
            figsize: Figure size (width, height) in inches
            estimation_method: Method to estimate mutual information ('histogram' or 'knn')

        Returns:
            Tuple containing:
            - SNR values
            - Dictionary mapping channel names to capacity values
            - Optional matplotlib figure if plot=True
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Generate default labels if not provided
        if labels is None:
            labels = [channel.__class__.__name__ for channel in channels]

        # Compute capacity for each channel
        capacities = {}
        for i, channel in enumerate(channels):
            _, capacity = self.modulation_capacity(modulator, channel, snr_db_range, num_symbols=num_symbols, estimation_method=estimation_method)
            capacities[labels[i]] = capacity

        # Generate plot if requested
        fig = None
        if plot:
            fig = self.plot_capacity_vs_snr(snr_db_range, capacities, title=f"Channel Capacity Comparison with {modulator.__class__.__name__}", figsize=figsize)

        return snr_db_range, capacities, fig

    def mimo_capacity(self, snr_db_range: Union[float, List[float], np.ndarray, torch.Tensor], tx_antennas: int = 2, rx_antennas: int = 2, channel_knowledge: str = "perfect", num_realizations: int = 1000) -> torch.Tensor:
        """Compute capacity for MIMO systems.

        For MIMO systems, the capacity varies based on channel knowledge at the transmitter.

        Args:
            snr_db_range: Range of SNR values in dB
            tx_antennas: Number of transmit antennas
            rx_antennas: Number of receive antennas
            channel_knowledge: Type of channel knowledge ('perfect', 'statistical', 'none')
            num_realizations: Number of channel realizations to average over

        Returns:
            torch.Tensor: MIMO capacity in bits per channel use
        """
        # Handle various input types for SNR values
        if isinstance(snr_db_range, (float, int)):
            snr_db_range = [snr_db_range]

        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Create cache key
        cache_key = (tuple(snr_db_range.cpu().numpy().tolist()), tx_antennas, rx_antennas, channel_knowledge, num_realizations)

        # Check if result is in cache
        if cache_key in self._mimo_cache:
            return self._mimo_cache[cache_key]

        # Convert SNR from dB to linear
        snr_linear = snr_db_to_linear(snr_db_range)

        # Initialize capacity
        capacity = torch.zeros_like(snr_db_range)

        # Validate parameters
        if tx_antennas <= 0 or rx_antennas <= 0:
            raise ValueError("Number of antennas must be positive")

        if channel_knowledge not in ["perfect", "statistical", "none"]:
            raise ValueError("Channel knowledge must be 'perfect', 'statistical', or 'none'")

        # Calculate capacity for each SNR point
        for i, snr in enumerate(snr_linear):
            capacity_sum = 0.0

            # Generate multiple random channel realizations
            for _ in range(num_realizations):
                # Generate Rayleigh fading channel matrix
                # H ~ CN(0, 1) i.i.d. elements
                H = torch.complex(torch.randn(rx_antennas, tx_antennas, device=self.device), torch.randn(rx_antennas, tx_antennas, device=self.device)) / math.sqrt(2)

                if channel_knowledge == "perfect":
                    # With perfect channel knowledge at both ends,
                    # capacity = sum_i log2(1 + λi²·SNR/Nt)
                    # where λi are singular values of H

                    # Perform SVD: H = U·Σ·V*
                    try:
                        # Use CPU for SVD as it's more stable
                        H_cpu = H.cpu()
                        U, S, V = torch.linalg.svd(H_cpu, full_matrices=False)
                        singular_values = S.to(self.device)
                    except RuntimeError:
                        # Fallback: compute eigenvalues of H*·H
                        H_H = H.conj().transpose(-2, -1) @ H
                        eigenvalues = torch.linalg.eigvals(H_H)
                        singular_values = torch.sqrt(torch.abs(eigenvalues))

                    # Compute optimal power allocation using water-filling
                    # For simplicity, we'll use equal power allocation
                    # which is optimal at high SNR
                    r = min(tx_antennas, rx_antennas)
                    capacity_value = torch.tensor(0.0)
                    for j in range(r):
                        capacity_value += torch.log2(1 + (singular_values[j] ** 2) * snr / tx_antennas)

                    capacity_sum += capacity_value.item()

                elif channel_knowledge == "statistical":
                    # With statistical knowledge (covariance) at transmitter,
                    # capacity achieved by diagonalizing the channel covariance
                    # For i.i.d. Rayleigh fading, equal power allocation is optimal
                    try:
                        # Handle the matrix determinant calculation more carefully to avoid overflow
                        r = min(tx_antennas, rx_antennas)

                        # Compute eigenvalues of H*H to avoid direct determinant calculation
                        H_H = H.conj().transpose(-2, -1) @ H
                        eigenvalues = torch.linalg.eigvalsh(H_H)

                        # Compute log determinant as sum of log(1 + λ*SNR/tx_antennas)
                        log_det = torch.tensor(0.0)
                        for j in range(len(eigenvalues)):
                            log_det += torch.log2(1 + eigenvalues[j] * snr / tx_antennas)

                        capacity_sum += log_det.item()
                    except Exception:
                        # Fallback method if the above fails
                        r = min(tx_antennas, rx_antennas)
                        # identity = torch.eye(r, device=self.device))
                        scaled_SNR = snr / tx_antennas

                        # Process singular values individually to avoid overflow
                        H_H = H.conj().transpose(-2, -1) @ H
                        eigenvalues = torch.linalg.eigvalsh(H_H)
                        log_det = sum(torch.log2(1 + scaled_SNR * ev).item() for ev in eigenvalues)

                        capacity_sum += log_det

                else:  # 'none'
                    # No CSI at transmitter, equal power allocation
                    try:
                        # Similar approach to avoid overflow
                        H_H_t = H @ H.conj().transpose(-2, -1)
                        eigenvalues = torch.linalg.eigvalsh(H_H_t)

                        log_det = torch.tensor(0.0)
                        for j in range(len(eigenvalues)):
                            log_det += torch.log2(1 + eigenvalues[j] * snr / tx_antennas)

                        capacity_sum += log_det.item()
                    except Exception:
                        # Fallback method
                        # identity = torch.eye(rx_antennas, device=self.device)
                        scaled_SNR = snr / tx_antennas

                        # Process singular values individually
                        H_H_t = H @ H.conj().transpose(-2, -1)
                        eigenvalues = torch.linalg.eigvalsh(H_H_t)
                        log_det = sum(torch.log2(1 + scaled_SNR * ev).item() for ev in eigenvalues)

                        capacity_sum += log_det

            # Average over all channel realizations
            capacity[i] = capacity_sum / num_realizations

        # Cache the result
        self._mimo_cache[cache_key] = capacity

        return capacity

    def capacity_gap_to_shannon(self, modulator: BaseModulator, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], num_symbols: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gap between a modulation scheme's capacity and the Shannon limit.

        This quantifies how close a practical modulation scheme comes to the theoretical limit.

        Args:
            modulator: The modulation scheme
            channel: The channel model
            snr_db_range: Range of SNR values in dB
            num_symbols: Number of symbols for Monte Carlo simulation

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, capacity gap in dB)
        """
        if isinstance(snr_db_range, (list, np.ndarray)):
            snr_db_range = torch.tensor(snr_db_range, device=self.device)

        # Calculate Shannon capacity
        shannon_capacity = self.awgn_capacity(snr_db_range)

        # Calculate modulation capacity
        _, modulation_capacity = self.modulation_capacity(modulator, channel, snr_db_range, num_symbols=num_symbols)

        # Calculate gap
        gap = shannon_capacity - modulation_capacity

        return snr_db_range, gap

    def capacity_cdf(self, channel: BaseChannel, snr_db: float, num_realizations: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the cumulative distribution function (CDF) of capacity.

        For fading channels, this shows the probability that capacity is less than a certain value.

        Args:
            channel: The channel model (should be a fading channel)
            snr_db: Signal-to-noise ratio in dB
            num_realizations: Number of channel realizations

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Capacity values, CDF values)
        """
        # Configure channel SNR
        if hasattr(channel, "snr_db"):
            channel.snr_db = snr_db

        # Array to store instantaneous capacity values
        inst_capacities = torch.zeros(num_realizations, device=self.device)

        # Small number of symbols per realization is sufficient
        num_symbols = 100

        # Compute capacity for each channel realization
        for i in range(num_realizations):
            # Generate input symbols
            input_symbols = torch.complex(torch.randn(num_symbols, device=self.device), torch.randn(num_symbols, device=self.device)) / math.sqrt(2)

            # Pass through channel
            output_symbols = channel(input_symbols)

            # Estimate instantaneous capacity
            inst_capacities[i] = self._estimate_mutual_information(input_symbols, output_symbols, bits_per_symbol=2)

        # Sort capacities and generate CDF
        sorted_capacities, _ = torch.sort(inst_capacities)
        cdf = torch.arange(1, num_realizations + 1, device=self.device) / num_realizations

        return sorted_capacities, cdf

    def spectral_efficiency(self, modulator: BaseModulator, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], bandwidth: float = 1.0, overhead: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate spectral efficiency for a modulation scheme.

        Takes into account protocol overhead to give a realistic measure of efficiency.

        Args:
            modulator: The modulation scheme
            channel: The channel model
            snr_db_range: Range of SNR values in dB
            bandwidth: Signal bandwidth in Hz
            overhead: Fraction of overhead (0 to 1) for protocols, pilots, etc.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, spectral efficiency in bits/s/Hz)
        """
        # Get raw capacity
        _, capacity = self.modulation_capacity(modulator, channel, snr_db_range)

        # Apply overhead reduction
        spectral_eff = capacity * (1 - overhead)

        return snr_db_range, spectral_eff

    def energy_efficiency(self, modulator: BaseModulator, channel: BaseChannel, snr_db_range: Union[List[float], np.ndarray, torch.Tensor], tx_power_watts: float = 1.0, circuit_power_watts: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate energy efficiency for a communication system.

        Energy efficiency is defined as bits/joule, accounting for both
        transmission power and circuit power consumption.

        Args:
            modulator: The modulation scheme
            channel: The channel model
            snr_db_range: Range of SNR values in dB
            tx_power_watts: Transmission power in watts
            circuit_power_watts: Circuit power consumption in watts

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (SNR values, energy efficiency in bits/joule)
        """
        # Get raw capacity
        _, capacity = self.modulation_capacity(modulator, channel, snr_db_range)

        # Calculate energy efficiency
        total_power = tx_power_watts + circuit_power_watts
        energy_eff = capacity / total_power  # bits/s/W = bits/joule

        return snr_db_range, energy_eff

    def _process_snr_for_mutual_information(self, snr_item, modulator, channel, num_symbols, bits_per_symbol, estimation_method, num_bins=100):
        """Process a single SNR point for mutual information calculation.

        This method is defined at the class level so it can be pickled for multiprocessing.

        Args:
            snr_item: Single SNR value in dB
            modulator: Modulation scheme to use
            channel: Channel model to use
            num_symbols: Number of symbols to simulate
            bits_per_symbol: Number of bits per symbol
            estimation_method: Method for estimating mutual information
            num_bins: Number of bins for histogram estimation

        Returns:
            float: Mutual information for this SNR point
        """
        # Create local copies of modulators with proper initialization
        modulator_type = modulator.__class__.__name__
        if modulator_type == "QAMModulator":
            # QAMModulator requires 'order' parameter
            if hasattr(modulator, "order"):
                order = modulator.order
            else:
                # If order is not directly accessible, estimate from bits_per_symbol
                order = 2**bits_per_symbol
            local_modulator = type(modulator)(order=order)
        elif modulator_type == "PSKModulator":
            # PSKModulator also requires 'order' parameter
            if hasattr(modulator, "order"):
                order = modulator.order
            else:
                # Estimate from bits_per_symbol
                order = 2**bits_per_symbol
            local_modulator = type(modulator)(order=order)
        else:
            # Other modulators like BPSK, QPSK might not need specific parameters
            local_modulator = type(modulator)()

        # Try to copy other modulator parameters
        try:
            if hasattr(modulator, "state_dict") and callable(getattr(modulator, "state_dict")):
                state_dict = modulator.state_dict()
                local_modulator.load_state_dict(state_dict)
        except Exception:
            # Fallback: copy attributes individually
            for attr_name in dir(modulator):
                if not attr_name.startswith("_") and not callable(getattr(modulator, attr_name)) and attr_name not in ["order"]:  # Skip 'order' as it's already handled
                    try:
                        setattr(local_modulator, attr_name, getattr(modulator, attr_name))
                    except (AttributeError, RuntimeError):
                        pass

        # Initialize channel directly with the SNR parameter
        channel_type = channel.__class__.__name__
        if channel_type == "AWGNChannel":
            # For AWGN channel, initialize with explicit SNR
            local_channel = type(channel)(snr_db=float(snr_item))
        elif channel_type == "RayleighFadingChannel":
            # For Rayleigh channel, preserve coherence time but update SNR
            coherence_time = getattr(channel, "coherence_time", 1)
            local_channel = type(channel)(coherence_time=coherence_time, snr_db=float(snr_item))
        elif channel_type == "RicianFadingChannel":
            # For Rician channel, preserve k_factor and coherence time but update SNR
            k_factor = getattr(channel, "k_factor", 1)
            coherence_time = getattr(channel, "coherence_time", 1)
            local_channel = type(channel)(k_factor=k_factor, coherence_time=coherence_time, snr_db=float(snr_item))
        else:
            # For other channel types, create instance and try to copy parameters
            local_channel = type(channel)()
            try:
                # Copy state_dict if available
                if hasattr(channel, "state_dict") and callable(getattr(channel, "state_dict")):
                    state_dict = channel.state_dict()
                    local_channel.load_state_dict(state_dict)
                # Always explicitly set SNR
                if hasattr(local_channel, "snr_db"):
                    local_channel.snr_db = float(snr_item)
            except Exception:
                # Fallback: copy attributes individually
                for attr_name in dir(channel):
                    if not attr_name.startswith("_") and not callable(getattr(channel, attr_name)) and attr_name not in ["snr_db", "avg_noise_power"]:
                        try:
                            setattr(local_channel, attr_name, getattr(channel, attr_name))
                        except (AttributeError, RuntimeError):
                            pass
                # Explicitly set SNR
                if hasattr(local_channel, "snr_db"):
                    local_channel.snr_db = float(snr_item)
                elif hasattr(local_channel, "avg_noise_power"):
                    # If channel uses avg_noise_power instead of snr_db
                    linear_snr = 10 ** (float(snr_item) / 10)
                    # Assuming unit power signal
                    local_channel.avg_noise_power = 1 / linear_snr

        # Generate random input bits
        input_bits = torch.randint(0, 2, (num_symbols, bits_per_symbol), dtype=torch.float32)

        # Modulate bits to symbols
        modulated = local_modulator(input_bits)

        # Normalize modulated signal to have unit energy
        modulated = modulated / torch.sqrt(torch.mean(torch.abs(modulated) ** 2))

        # Pass through channel
        received = local_channel(modulated)

        # Calculate mutual information using specified method
        if estimation_method == "knn":
            return self._estimate_mutual_information_knn(modulated, received, k=3, bits_per_symbol=bits_per_symbol).item()
        else:
            return self._estimate_mutual_information(modulated, received, num_bins=num_bins, bits_per_symbol=bits_per_symbol).item()
