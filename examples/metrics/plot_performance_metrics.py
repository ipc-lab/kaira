"""
=================================
Performance Metrics Visualization
=================================

This example demonstrates how to create visually appealing visualizations of
performance metrics used in communication systems using the Kaira library.
We'll visualize various metrics like BER, SNR, capacity, and other important
measures of system performance.
"""

# %%
# Imports and Setup
# --------------------------
#
# First, let's import the necessary libraries and set up our environment.

import numpy as np
import torch
from scipy import special

from examples.utils.plotting import (
    setup_plotting_style,
    plot_ber_performance,
    plot_constellation_comparison,
    plot_ber_vs_snr_comparison
)

from kaira.channels import AWGNChannel, BinarySymmetricChannel, RayleighFadingChannel
from kaira.utils import seed_everything

# Set seeds for reproducibility
seed_everything(42)

# Configure plotting style
setup_plotting_style()

# %%
# Theoretical vs. Simulated BER Curves
# ------------------------------------------------------------------------
#
# Let's compare theoretical and simulated Bit Error Rate (BER) curves for different
# modulation schemes in an AWGN channel.

# SNR range in dB
snr_db_range = np.linspace(0, 20, 11)


# Theoretical BER calculations
def q_function(x):
    """Compute the Q-function (tail probability of the standard normal distribution).

    The Q-function is commonly used in communications to calculate error probabilities.
    It represents the probability that a standard normal random variable exceeds
    the value x, defined as Q(x) = (1/√(2π)) ∫_x^∞ exp(-t²/2) dt.

    Parameters
    -------------------
    x : float or numpy.ndarray
        Input value(s) at which to evaluate the Q-function.

    Returns
    -------
    float or numpy.ndarray
        The Q-function evaluated at the input value(s).
    """
    return 0.5 * special.erfc(x / np.sqrt(2))


# Theoretical BER for BPSK in AWGN
ber_bpsk_theory = [q_function(np.sqrt(2 * 10 ** (snr / 10))) for snr in snr_db_range]

# Theoretical BER for QPSK in AWGN (same as BPSK for gray coding)
ber_qpsk_theory = ber_bpsk_theory

# Theoretical BER for 16-QAM in AWGN (approximation)
ber_16qam_theory = [0.75 * q_function(np.sqrt((4 / 5) * 10 ** (snr / 10) / 4)) for snr in snr_db_range]


# Simulated BER calculations
def simulate_ber(snr_db_range, modulation_type):
    """Simulate Bit Error Rate (BER) for different modulation schemes over AWGN channel.

    This function performs Monte Carlo simulation to calculate the BER for different
    modulation schemes (BPSK, QPSK, 16-QAM) across a range of SNR values using an
    AWGN channel model.

    Parameters
    -------------------
    snr_db_range : array_like
        Array of SNR values in dB to simulate over.
    modulation_type : str
        Type of modulation to simulate. Must be one of "BPSK", "QPSK", or "16-QAM".

    Returns
    -------
    list
        List of BER values corresponding to each SNR value in the input range.
    """
    bers = []
    for snr in snr_db_range:
        # Set up channel
        channel = AWGNChannel(snr_db=snr)

        # Generate input data based on modulation type
        if modulation_type == "BPSK":
            # BPSK: {0, 1} -> {-1, 1}
            input_bits = torch.randint(0, 2, (10000, 1), dtype=torch.float32)
            modulated = 2 * input_bits - 1
        elif modulation_type == "QPSK":
            # QPSK: 2 bits per symbol, represented as real-valued tensor with 2 channels
            input_bits = torch.randint(0, 2, (10000, 2), dtype=torch.float32)
            modulated = 2 * input_bits - 1
        elif modulation_type == "16-QAM":
            # 16-QAM: 4 bits per symbol, simplified representation
            input_bits = torch.randint(0, 2, (10000, 4), dtype=torch.float32)
            # Convert to appropriate constellation points (simplified)
            modulated = torch.zeros((10000, 2), dtype=torch.float32)
            for i in range(10000):
                # Convert 2 bits to constellation point for I and Q
                i_bits = input_bits[i, 0:2]
                q_bits = input_bits[i, 2:4]

                # Map bits to constellation points {-3, -1, 1, 3}
                i_val = 2 * (2 * i_bits[0] - 1) - (2 * i_bits[1] - 1)
                q_val = 2 * (2 * q_bits[0] - 1) - (2 * q_bits[1] - 1)

                modulated[i, 0] = i_val
                modulated[i, 1] = q_val

            # Normalize by average symbol energy
            modulated = modulated / np.sqrt(10)

        # Transmit through channel
        received = channel(modulated)

        # Demodulate (hard decision)
        if modulation_type == "BPSK":
            demodulated = (received > 0).float()
            ber = torch.mean((demodulated != input_bits).float()).item()
        elif modulation_type == "QPSK":
            demodulated = (received > 0).float()
            ber = torch.mean((demodulated != input_bits).float()).item()
        elif modulation_type == "16-QAM":
            # Simplified 16-QAM demodulation
            demodulated = torch.zeros_like(input_bits)
            for i in range(10000):
                rx_i, rx_q = received[i]

                # Demodulate I component
                i_bits = torch.zeros(2, dtype=torch.float32)
                i_bits[0] = (rx_i > 0).float()
                i_bits[1] = 1.0 - ((rx_i > -2) & (rx_i < 2)).float()  # Negate for correct mapping

                # Demodulate Q component
                q_bits = torch.zeros(2, dtype=torch.float32)
                q_bits[0] = (rx_q > 0).float()
                q_bits[1] = 1.0 - ((rx_q > -2) & (rx_q < 2)).float()  # Negate for correct mapping

                demodulated[i, 0:2] = i_bits
                demodulated[i, 2:4] = q_bits

            ber = torch.mean((demodulated != input_bits).float()).item()

        bers.append(ber)

    return bers


# Simulate BER for different modulation schemes
ber_bpsk_sim = simulate_ber(snr_db_range, "BPSK")
ber_qpsk_sim = simulate_ber(snr_db_range, "QPSK")
ber_16qam_sim = simulate_ber(snr_db_range, "16-QAM")

# Plotting
plt.figure(figsize=(12, 8))

# Plot theoretical curves
plt.semilogy(snr_db_range, ber_bpsk_theory, "b-", linewidth=2, label="BPSK (Theory)")
plt.semilogy(snr_db_range, ber_qpsk_theory, "g-", linewidth=2, label="QPSK (Theory)")
plt.semilogy(snr_db_range, ber_16qam_theory, "r-", linewidth=2, label="16-QAM (Theory)")

# Plot simulated results
plt.semilogy(snr_db_range, ber_bpsk_sim, "bo", markersize=8, label="BPSK (Simulated)")
plt.semilogy(snr_db_range, ber_qpsk_sim, "g^", markersize=8, label="QPSK (Simulated)")
plt.semilogy(snr_db_range, ber_16qam_sim, "rs", markersize=8, label="16-QAM (Simulated)")

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Bit Error Rate (BER)", fontsize=14)
plt.title("Theoretical vs. Simulated BER for Different Modulation Schemes", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 20)
plt.ylim(1e-6, 1)
plt.tight_layout()
plt.show()

# %%
# Channel Capacity Visualization
# --------------------------------------------------------
#
# Let's visualize the theoretical channel capacity for different channels.

# Calculate Shannon capacity for AWGN channel: C = log2(1 + SNR)
snr_linear = 10 ** (snr_db_range / 10)
capacity_awgn = np.log2(1 + snr_linear)


# Calculate capacity for Binary Symmetric Channel (BSC): C = 1 - H(p)
# where H(p) is the binary entropy function
def binary_entropy(p):
    """Calculate the binary entropy H(p) in bits.

    The binary entropy function quantifies the information content of a Bernoulli(p)
    random variable:
    H(p) = -p * log2(p) - (1 - p) * log2(1 - p).

    Parameters
    -------------------
    p : float
        Probability value between 0 and 1.

    Returns
    -------
    float
        The binary entropy in bits.
    """
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Calculate BSC capacity for different crossover probabilities
# Relate SNR to crossover probability using an approximate formula
p_crossover = 0.5 * np.exp(-snr_linear / 2)
capacity_bsc = [1 - binary_entropy(p) for p in p_crossover]

# Calculate capacity for Binary Erasure Channel (BEC): C = 1 - p
# where p is the erasure probability (also related to SNR)
p_erasure = 0.5 * np.exp(-snr_linear / 2)  # Similar to BSC relation
capacity_bec = 1 - p_erasure

# Calculate capacity for Rayleigh fading channel (no CSI at Tx)
# C ≈ E[log2(1 + |h|^2 * SNR)] ≈ log2(e) * e^(1/SNR) * E1(1/SNR)
# Using a simpler approximation: C ≈ log2(1 + SNR) * exp(-1/SNR)
capacity_rayleigh = capacity_awgn * np.exp(-1 / snr_linear)

# Plotting
plt.figure(figsize=(12, 8))

plt.plot(snr_db_range, capacity_awgn, "b-", linewidth=3, label="AWGN Channel")
plt.plot(snr_db_range, capacity_bsc, "g-", linewidth=3, label="Binary Symmetric Channel")
plt.plot(snr_db_range, capacity_bec, "r-", linewidth=3, label="Binary Erasure Channel")
plt.plot(snr_db_range, capacity_rayleigh, "m-", linewidth=3, label="Rayleigh Fading Channel")

plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Channel Capacity (bits/channel use)", fontsize=14)
plt.title("Channel Capacity vs. SNR for Different Channel Models", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Power vs. Rate Trade-off Visualization
# -------------------------------------------------------------------------
#
# Let's visualize the trade-off between transmit power and achievable rate for
# different target error probabilities.


# Theoretical capacity function
def capacity_function(snr_db):
    """Calculate the theoretical Shannon capacity for an AWGN channel.

    This function computes the maximum achievable transmission rate for a given
    signal-to-noise ratio (SNR) according to Shannon's capacity formula:
    C = log₂(1 + SNR)

    Parameters
    -------------------
    snr_db : float or numpy.ndarray
        Signal-to-noise ratio in decibels (dB).

    Returns
    -------
    float or numpy.ndarray
        Channel capacity in bits per channel use.
    """
    return np.log2(1 + 10 ** (snr_db / 10))


# Calculate the required SNR for a given target rate and error probability
def required_snr(rate, error_prob):
    """Calculate the required SNR to achieve a target rate with given error probability.

    This function computes the approximate signal-to-noise ratio (SNR) required to
    achieve a specified transmission rate with a given target error probability.
    It uses a simplified approximation: SNR ≈ 2^R - 1 + gap, where the gap is
    related to the error probability.

    Parameters
    -------------------
    rate : float
        Target transmission rate in bits per channel use.
    error_prob : float
        Target error probability (between 0 and 1).

    Returns
    -------
    float
        Required SNR in decibels (dB) to achieve the target rate with the
        specified error probability.
    """
    # Using a simplified approximation: SNR ≈ 2^R - 1 + gap
    # where the gap is related to the error probability
    gap = -np.log(error_prob / 2)  # Approximate relation
    return 10 * np.log10(2**rate - 1 + gap / 10)


# Range of rates to consider
rates = np.linspace(0.1, 5, 50)

# Different target error probabilities
error_probs = [1e-2, 1e-3, 1e-4, 1e-5]

plt.figure(figsize=(12, 8))

for error_prob in error_probs:
    snr_values = [required_snr(r, error_prob) for r in rates]
    plt.plot(rates, snr_values, linewidth=3, label=f"Target Error Prob = {error_prob:.0e}")

# Add the theoretical channel capacity curve
capacity_snrs = np.linspace(-10, 20, 100)
capacity_rates = [capacity_function(snr) for snr in capacity_snrs]
plt.plot(capacity_rates, capacity_snrs, "k--", linewidth=2, label="Theoretical Capacity Limit")

plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("Achievable Rate (bits/channel use)", fontsize=14)
plt.ylabel("Required SNR (dB)", fontsize=14)
plt.title("Power vs. Rate Trade-off for Different Target Error Probabilities", fontsize=16)
plt.legend(fontsize=12)
plt.ylim(-10, 25)
plt.tight_layout()
plt.show()

# %%
# BER Performance Comparison Across Channels
# ----------------------------------------------------------------------------
#
# Let's compare the BER performance of BPSK modulation across different channel types.

# SNR range
snr_db_range = np.linspace(0, 20, 11)

# Theoretical BER for BPSK in AWGN
ber_awgn_theory = [q_function(np.sqrt(2 * 10 ** (snr / 10))) for snr in snr_db_range]

# Theoretical BER for BPSK in Rayleigh fading
ber_rayleigh_theory = [0.5 * (1 - np.sqrt(10 ** (snr / 10) / (1 + 10 ** (snr / 10)))) for snr in snr_db_range]

# Theoretical BER for BPSK with BSC (using approximation)
ber_bsc_theory = [0.5 * np.exp(-(10 ** (snr / 10)) / 2) for snr in snr_db_range]


# Simulated BER for different channels
def simulate_channel_ber(snr_db_range, channel_type):
    """Simulate Bit Error Rate (BER) for BPSK across different channel types.

    Perform Monte Carlo simulation to estimate BER of BPSK modulation over
    AWGN, Rayleigh fading, or Binary Symmetric Channel models for a range of SNRs.

    Parameters
    -------------------
    snr_db_range : array_like
        Sequence of SNR values in dB.
    channel_type : str
        Type of channel: "AWGN", "Rayleigh", or "BSC".

    Returns
    -------
    list of float
        Estimated BER for each SNR value.
    """
    bers = []
    for snr in snr_db_range:
        # Set up channel
        if channel_type == "AWGN":
            channel = AWGNChannel(snr_db=snr)
        elif channel_type == "Rayleigh":
            channel = RayleighFadingChannel(snr_db=snr)
        elif channel_type == "BSC":
            # Approximate BSC crossover probability based on SNR
            p = 0.5 * np.exp(-(10 ** (snr / 10)) / 2)
            channel = BinarySymmetricChannel(crossover_prob=p)

        # Generate BPSK input
        input_bits = torch.randint(0, 2, (10000, 1), dtype=torch.float32)
        modulated = 2 * input_bits - 1

        # Transmit through channel
        received = channel(modulated)

        # Demodulate (hard decision)
        if channel_type == "Rayleigh":
            # For complex signals, take the real part for demodulation
            demodulated = (received.real > 0).float()
        else:
            demodulated = (received > 0).float()
        ber = torch.mean((demodulated != input_bits).float()).item()

        bers.append(ber)

    return bers


# Simulate BER for different channels
ber_awgn_sim = simulate_channel_ber(snr_db_range, "AWGN")
ber_rayleigh_sim = simulate_channel_ber(snr_db_range, "Rayleigh")
ber_bsc_sim = simulate_channel_ber(snr_db_range, "BSC")

# Plotting
plt.figure(figsize=(12, 8))

# Plot theoretical curves
plt.semilogy(snr_db_range, ber_awgn_theory, "b-", linewidth=2, label="AWGN (Theory)")
plt.semilogy(snr_db_range, ber_rayleigh_theory, "g-", linewidth=2, label="Rayleigh Fading (Theory)")
plt.semilogy(snr_db_range, ber_bsc_theory, "r-", linewidth=2, label="BSC (Theory)")

# Plot simulated results
plt.semilogy(snr_db_range, ber_awgn_sim, "bo", markersize=8, label="AWGN (Simulated)")
plt.semilogy(snr_db_range, ber_rayleigh_sim, "g^", markersize=8, label="Rayleigh Fading (Simulated)")
plt.semilogy(snr_db_range, ber_bsc_sim, "rs", markersize=8, label="BSC (Simulated)")

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Bit Error Rate (BER)", fontsize=14)
plt.title("BER Performance Comparison of BPSK Across Different Channels", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 20)
plt.ylim(1e-6, 1)
plt.tight_layout()
plt.show()

# %% Radar Chart Visualization of Channel Characteristics
# -----------------------------------------------------------------------------------------------
#
# Let's create a radar chart to compare different aspects of various channel models.


def radar_factory(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes."""
    # Calculate angles for each axis
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        """Custom PolarAxes subclass for radar charts.

        Extends PolarAxes to support radar-specific features:
        - Closed polygons by default
        - Automatic closure on fill and plot
        - Convenient variable label placement
        """

        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default."""
            closed = kwargs.pop("closed", True)
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            """Close a plotted line to form a closed polygon.

            Appends the first data point to the end so that the radar plot is drawn as a closed
            shape.
            """
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            """Set variable labels at the corresponding theta grid points.

            Places each label around the radar chart at its associated angle.

            Parameters
            -------------------
            labels : list of str
                Text labels for each axis.
            """
            self.set_thetagrids(np.degrees(theta), labels)

    # Register custom projection
    register_projection(RadarAxes)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection="radar")

    return fig, ax, theta


# Define channel characteristics to compare
characteristics = ["BER Performance", "Capacity", "Implementation\nComplexity", "Model\nAccuracy", "Computational\nEfficiency", "Adaptability"]

# Define scores for each channel (0-10 scale, 10 being best)
# These are illustrative scores, not precise measurements
channel_data = {"AWGN Channel": [8, 9, 10, 9, 10, 7], "BSC": [6, 7, 10, 7, 10, 6], "Rayleigh Fading": [5, 6, 7, 9, 8, 9], "Rician Fading": [4, 5, 6, 9, 7, 8]}

# Create radar chart
fig, ax, theta = radar_factory(len(characteristics))

# Plot each channel
for i, (channel, scores) in enumerate(channel_data.items()):
    color = colors[i]
    ax.plot(theta, scores, color=color, linewidth=2.5, label=channel)
    ax.fill(theta, scores, color=color, alpha=0.2)

# Set chart properties
ax.set_varlabels(characteristics)
plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
plt.title("Comparison of Channel Models", fontsize=16, y=1.05)

plt.tight_layout()
plt.show()

# %%
# Diversity Gain Visualization
# --------------------------------------------
#
# Let's visualize the benefits of diversity techniques in fading channels.

# SNR range
snr_db_range = np.linspace(0, 20, 21)
snr_linear = 10 ** (snr_db_range / 10)


# Theoretical BER curves for different diversity orders in Rayleigh fading
def ber_rayleigh_diversity(snr_db, diversity_order):
    """Calculate theoretical BER for BPSK with diversity in Rayleigh fading."""
    snr_linear = 10 ** (snr_db / 10)

    if diversity_order == 1:
        # No diversity
        return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
    else:
        # With diversity (simplified approximation)
        # Based on asymptotic behavior: BER ≈ (1/(4*SNR))^L * binomial(2L-1, L)
        coef = special.comb(2 * diversity_order - 1, diversity_order)
        return coef * (0.25 / snr_linear) ** diversity_order


# Calculate BER for different diversity orders
diversity_orders = [1, 2, 4, 8]
ber_curves = []

for order in diversity_orders:
    ber = [ber_rayleigh_diversity(snr, order) for snr in snr_db_range]
    ber_curves.append(ber)

# Plotting
plt.figure(figsize=(12, 8))

for i, order in enumerate(diversity_orders):
    plt.semilogy(snr_db_range, ber_curves[i], linewidth=3, label=f"Diversity Order = {order}")

# Add asymptotic slopes for reference
for i, order in enumerate(diversity_orders):
    asymptotic_ber = [(0.25 / snr) ** order for snr in snr_linear]
    plt.semilogy(snr_db_range, asymptotic_ber, "k--", linewidth=1, alpha=0.5)

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Bit Error Rate (BER)", fontsize=14)
plt.title("Impact of Diversity Order on BER Performance in Rayleigh Fading", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 20)
plt.ylim(1e-7, 1)
plt.tight_layout()
plt.show()

# %%
# Outage Probability Visualization
# ---------------------------------------------------------
#
# Let's visualize the outage probability for different channel models and
# transmission rates.

# SNR range
snr_db_range = np.linspace(0, 30, 31)
snr_linear = 10 ** (snr_db_range / 10)


# Calculate outage probability for Rayleigh fading channel
# P_out = 1 - exp(-2^R-1/SNR)
def rayleigh_outage_prob(snr_db, rate):
    """Calculate outage probability for Rayleigh fading channel."""
    snr_linear = 10 ** (snr_db / 10)
    threshold = 2**rate - 1
    return 1 - np.exp(-threshold / snr_linear)


# Calculate outage probability for Rician fading channel
# Simplified approximation based on K-factor
def rician_outage_prob(snr_db, rate, K_factor):
    """Calculate outage probability for Rician fading channel."""
    snr_linear = 10 ** (snr_db / 10)
    threshold = 2**rate - 1
    # Simplified approximation
    # Higher K factor means stronger line-of-sight component
    return 1 - special.marcumq(np.sqrt(2 * K_factor), np.sqrt(2 * (K_factor + 1) * threshold / snr_linear))


# Rates to compare
rates = [1, 2, 4]  # bits/channel use

# K-factor for Rician fading
K_factor = 5  # dB

# Calculate outage probabilities
rayleigh_outage = []
for rate in rates:
    rayleigh_outage.append([rayleigh_outage_prob(snr, rate) for snr in snr_db_range])

# Plotting
plt.figure(figsize=(12, 8))

for i, rate in enumerate(rates):
    plt.semilogy(snr_db_range, rayleigh_outage[i], linewidth=3, label=f"Rate = {rate} bits/use (Rayleigh)")

# Add outage capacity curve (when rate = log2(1+SNR))
outage_capacity = [rayleigh_outage_prob(snr, np.log2(1 + 10 ** (snr / 10))) for snr in snr_db_range]
plt.semilogy(snr_db_range, outage_capacity, "k--", linewidth=2, label="Outage at Capacity (Rayleigh)")

plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Outage Probability", fontsize=14)
plt.title("Outage Probability vs. SNR for Different Transmission Rates", fontsize=16)
plt.legend(fontsize=12)
plt.xlim(0, 30)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.show()

# %%
# Conclusion
# -------------------
#
# In this visualization example, we have created various attractive and informative
# visualizations of performance metrics in communication systems using the Kaira library.
# These visualizations help in understanding the theoretical limits, practical performance,
# and trade-offs in different communication scenarios.
#
# The key insights from these visualizations are:
#
# 1. Higher-order modulation schemes offer higher spectral efficiency but require higher SNR
#    to achieve the same BER performance.
#
# 2. Different channel models have different capacities and BER characteristics:
#    - AWGN channels have the highest capacity for a given SNR
#    - Fading channels significantly degrade performance compared to AWGN
#    - Binary channels have limited capacity that saturates at high SNR
#
# 3. There is a fundamental trade-off between power (SNR) and rate, which becomes
#    more pronounced as the target error probability decreases.
#
# 4. Diversity techniques can significantly improve performance in fading channels,
#    with diminishing returns as the diversity order increases.
#
# These visualizations serve as valuable tools for understanding and designing
# communication systems using the Kaira library.
