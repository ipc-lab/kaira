"""
==========================================================================================================================================================================
Practical Applications of Constraints in Wireless Communication Systems
==========================================================================================================================================================================

This example demonstrates practical applications of Kaira's constraints in realistic
wireless communication scenarios, focusing on OFDM and MIMO systems. We'll explore
how to configure and apply appropriate constraints for these systems.
"""

# %%
# Imports and Setup
# ----------------------------------------------------------
# We start by importing the necessary modules and setting up the environment.

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.constraints import (
    PAPRConstraint,
    PeakAmplitudeConstraint,
    SpectralMaskConstraint,
    TotalPowerConstraint,
)
from kaira.constraints.utils import (
    apply_constraint_chain,
    combine_constraints,
    create_mimo_constraints,
    create_ofdm_constraints,
    measure_signal_properties,
    verify_constraint,
)
from kaira.utils.plotting import PlottingUtils

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
PlottingUtils.setup_plotting_style()

# %%
# Part 1: OFDM System Constraints
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# OFDM (Orthogonal Frequency Division Multiplexing) is widely used in modern communication
# systems like Wi-Fi, 5G, and digital broadcasting. OFDM signals typically require constraints
# to control their high PAPR (Peak-to-Average Power Ratio).

# Create an OFDM signal (simplified)
n_subcarriers = 1024
n_symbols = 10
CP_length = n_subcarriers // 4  # Cyclic prefix length

# Create random frequency-domain OFDM symbols
X_freq = torch.complex(torch.randn(1, n_symbols, n_subcarriers), torch.randn(1, n_symbols, n_subcarriers))

# Zero out DC and edge subcarriers (as in real OFDM systems)
X_freq[:, :, 0] = 0  # DC
guard_band = int(0.05 * n_subcarriers)  # 5% guard band on each edge
X_freq[:, :, -guard_band:] = 0
X_freq[:, :, :guard_band] = 0

# Convert to time domain with IFFT
X_time = torch.fft.ifft(X_freq, dim=2)

# Add cyclic prefix
cp_indices = torch.arange(n_subcarriers - CP_length, n_subcarriers)
with_cp = []
for i in range(n_symbols):
    symbol = X_time[:, i, :]
    cp = symbol[:, cp_indices]
    with_cp.append(torch.cat([cp, symbol], dim=1))

# Create final OFDM signal
ofdm_signal = torch.cat(with_cp, dim=1)

# Convert to real representation (I/Q components)
ofdm_iq = torch.cat([ofdm_signal.real, ofdm_signal.imag], dim=0)

# Display properties of the original OFDM signal
ofdm_props = measure_signal_properties(ofdm_iq)

# Original OFDM Signal Properties:
# - Shape: ofdm_iq.shape (2 components, samples)
# - Power: ofdm_props['mean_power']
# - PAPR: ofdm_props['papr'] (ofdm_props['papr_db'] dB)
# - Peak Amplitude: ofdm_props['peak_amplitude']
print("Original OFDM Signal Properties:")
print(f"  Shape: {ofdm_iq.shape} (2 components, {ofdm_iq.shape[1]} samples)")
print(f"  Power: {ofdm_props['mean_power']:.4f}")
print(f"  PAPR: {ofdm_props['papr']:.2f} ({ofdm_props['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {ofdm_props['peak_amplitude']:.4f}")

# %%
# OFDM Signal Analysis
# ------------------------------------------------------------------------
# First, let's analyze the original OFDM signal characteristics.

# Calculate time and frequency vectors for plotting
t = np.arange(ofdm_iq.shape[1])
signal_i = ofdm_iq[0].numpy()
signal_q = ofdm_iq[1].numpy()

# Generate comprehensive OFDM signal analysis visualization
# Time vector for plotting
plot_samples = min(1000, len(signal_i))
t_plot = np.arange(plot_samples)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Original OFDM Signal Analysis", fontsize=16, fontweight="bold")

# Plot I and Q components
ax1 = axes[0, 0]
ax1.plot(t_plot, signal_i[:plot_samples], color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, label="I component")
ax1.plot(t_plot, signal_q[:plot_samples], color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, label="Q component")
ax1.set_title("I/Q Components")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot instantaneous power
ax2 = axes[0, 1]
instantaneous_power = signal_i[:plot_samples] ** 2 + signal_q[:plot_samples] ** 2
ax2.plot(t_plot, instantaneous_power, color=PlottingUtils.MODERN_PALETTE[2], linewidth=1.5)
ax2.axhline(y=np.mean(instantaneous_power), color="red", linestyle="--", label=f"Average Power: {np.mean(instantaneous_power):.3f}")
ax2.set_title("Instantaneous Power")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Power")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot constellation
ax3 = axes[1, 0]
# Sample every nth point for cleaner constellation
n_samples_const = min(500, len(signal_i))
sample_step = max(1, len(signal_i) // n_samples_const)
ax3.scatter(signal_i[::sample_step], signal_q[::sample_step], s=15, alpha=0.6, color=PlottingUtils.MODERN_PALETTE[3])
ax3.set_title("I/Q Constellation")
ax3.set_xlabel("I Component")
ax3.set_ylabel("Q Component")
ax3.grid(True, alpha=0.3)
ax3.axis("equal")

# Plot power distribution histogram
ax4 = axes[1, 1]
ax4.hist(instantaneous_power, bins=50, density=True, alpha=0.7, color=PlottingUtils.MODERN_PALETTE[4], edgecolor="black")
ax4.axvline(x=np.mean(instantaneous_power), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(instantaneous_power):.3f}")
ax4.axvline(x=np.max(instantaneous_power), color="orange", linestyle="--", linewidth=2, label=f"Peak: {np.max(instantaneous_power):.3f}")
ax4.set_title("Power Distribution")
ax4.set_xlabel("Instantaneous Power")
ax4.set_ylabel("Density")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.show()

# %%
# Applying OFDM Constraints
# -------------------------------------------------------------------------------------------------------------------
# Let's configure and apply appropriate constraints for the OFDM signal.

# Create OFDM constraints using the factory function
ofdm_constraints = create_ofdm_constraints(total_power=1.0, max_papr=6.0, is_complex=True, peak_amplitude=2.5)  # Normalize total power to 1.0  # Limit PAPR to 6 (approximately 7.8 dB)  # Signal has I/Q components  # Limit maximum amplitude

# Apply constraints to the OFDM signal
constrained_ofdm = ofdm_constraints(ofdm_iq.clone())

# Measure properties of the constrained signal
constrained_props = measure_signal_properties(constrained_ofdm)

# Constrained OFDM Signal Properties:
# - Power: constrained_props['mean_power']
# - PAPR: constrained_props['papr'] (constrained_props['papr_db'] dB)
# - Peak Amplitude: constrained_props['peak_amplitude']
print("\nConstrained OFDM Signal Properties:")
print(f"  Power: {constrained_props['mean_power']:.4f}")
print(f"  PAPR: {constrained_props['papr']:.2f} ({constrained_props['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {constrained_props['peak_amplitude']:.4f}")

# Alternative approach: apply individual constraints sequentially with verbose output
# Sequential Application of OFDM Constraints: This demonstrates step-by-step constraint application
print("\nSequential Application of OFDM Constraints:")
constraints_list = [TotalPowerConstraint(total_power=1.0), PAPRConstraint(max_papr=6.0), PeakAmplitudeConstraint(max_amplitude=2.5)]

sequential_ofdm = apply_constraint_chain(constraints_list, ofdm_iq.clone())
sequential_props = measure_signal_properties(sequential_ofdm)

# %%
# Visualizing OFDM Constraint Effects
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Extract I/Q components for visualization
constrained_i = constrained_ofdm[0].numpy()
constrained_q = constrained_ofdm[1].numpy()
constrained_power = constrained_i**2 + constrained_q**2

# Generate OFDM constraint effects visualization
# Calculate signal segment for detailed analysis
plot_segment = slice(0, 1000)
original_power = signal_i**2 + signal_q**2

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("OFDM Constraint Effects Analysis", fontsize=16, fontweight="bold")

# Time domain comparison - I component
ax1 = axes[0, 0]
t_seg = np.arange(1000)
ax1.plot(t_seg, signal_i[:1000], color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label="Original I")
ax1.plot(t_seg, constrained_i[:1000], color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.7, label="Constrained I")
ax1.set_title("I Component: Before vs After Constraints")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power comparison
ax2 = axes[0, 1]
ax2.plot(t_seg, original_power[:1000], color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label="Original Power")
ax2.plot(t_seg, constrained_power[:1000], color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.7, label="Constrained Power")
ax2.axhline(y=np.mean(original_power), color=PlottingUtils.MODERN_PALETTE[0], linestyle="--", alpha=0.8, label=f"Orig Avg: {np.mean(original_power):.3f}")
ax2.axhline(y=np.mean(constrained_power), color=PlottingUtils.MODERN_PALETTE[1], linestyle="--", alpha=0.8, label=f"Const Avg: {np.mean(constrained_power):.3f}")
ax2.set_title("Instantaneous Power Comparison")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Power")
ax2.legend()
ax2.grid(True, alpha=0.3)

# PAPR comparison bar chart
ax3 = axes[1, 0]
papr_data = [ofdm_props["papr_db"], constrained_props["papr_db"]]
power_data = [ofdm_props["mean_power"], constrained_props["mean_power"]]
categories = ["Original", "Constrained"]

x_pos = np.arange(len(categories))
ax3_twin = ax3.twinx()

bars1 = ax3.bar(x_pos - 0.2, papr_data, 0.4, label="PAPR (dB)", color=PlottingUtils.MODERN_PALETTE[2], alpha=0.7)
bars2 = ax3_twin.bar(x_pos + 0.2, power_data, 0.4, label="Power", color=PlottingUtils.MODERN_PALETTE[3], alpha=0.7)

ax3.set_title("PAPR and Power Comparison")
ax3.set_xlabel("Signal Type")
ax3.set_ylabel("PAPR (dB)", color=PlottingUtils.MODERN_PALETTE[2])
ax3_twin.set_ylabel("Power", color=PlottingUtils.MODERN_PALETTE[3])
ax3.set_xticks(x_pos)
ax3.set_xticklabels(categories)
ax3.legend(loc="upper left")
ax3_twin.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# Constellation comparison
ax4 = axes[1, 1]
# Sample points for cleaner visualization
sample_step = max(1, len(signal_i) // 300)
ax4.scatter(signal_i[::sample_step], signal_q[::sample_step], s=20, alpha=0.6, color=PlottingUtils.MODERN_PALETTE[0], label="Original")
ax4.scatter(constrained_i[::sample_step], constrained_q[::sample_step], s=20, alpha=0.6, color=PlottingUtils.MODERN_PALETTE[1], label="Constrained")
ax4.set_title("Constellation Comparison")
ax4.set_xlabel("I Component")
ax4.set_ylabel("Q Component")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis("equal")

plt.show()

# %%
# Verify OFDM Constraint Effectiveness
# --------------------------------------------------------------------------------------------------------------------
# Let's verify that our constraints achieved their goals

# Verify power constraint
power_result = verify_constraint(TotalPowerConstraint(total_power=1.0), ofdm_iq.clone(), "power", 1.0, tolerance=1e-4)

# Power constraint verification results: power_result
print(f"Power constraint verification: {power_result}")

# Verify PAPR constraint
papr_result = verify_constraint(PAPRConstraint(max_papr=6.0), sequential_ofdm.clone(), "papr", 6.0, tolerance=1e-3)  # Use output from sequential application

# PAPR constraint verification results: papr_result
print(f"PAPR constraint verification: {papr_result}")

# Verify amplitude constraint
amplitude_result = verify_constraint(PeakAmplitudeConstraint(max_amplitude=2.5), constrained_ofdm.clone(), "amplitude", 2.5, tolerance=1e-4)

# Amplitude constraint verification results: amplitude_result
print(f"Amplitude constraint verification: {amplitude_result}")

# %%
# Part 2: MIMO System Constraints
# -------------------------------------------
# MIMO (Multiple-Input Multiple-Output) systems use multiple antennas to improve
# communication performance. Each antenna typically has its own power constraints.

# Define MIMO system parameters
n_antennas = 4
n_symbols = 50
n_subcarriers = 64

# Create a random MIMO signal
mimo_signal = torch.complex(torch.randn(n_antennas, n_symbols * n_subcarriers), torch.randn(n_antennas, n_symbols * n_subcarriers))

# Display properties of the original MIMO signal
# Original MIMO Signal Properties:
# - Shape: mimo_signal.shape (n_antennas antennas, samples per antenna)
# - Per-antenna Power: calculated for each antenna
# - Total Power: sum of all antenna powers
print("\nOriginal MIMO Signal Properties:")
print(f"  Shape: {mimo_signal.shape} ({n_antennas} antennas, {mimo_signal.shape[1]} samples per antenna)")

# Calculate per-antenna power
per_antenna_power = []
for i in range(n_antennas):
    antenna_power = torch.mean(torch.abs(mimo_signal[i]) ** 2).item()
    per_antenna_power.append(antenna_power)
    print(f"  Antenna {i+1} Power: {antenna_power:.4f}")

print(f"  Total Power: {sum(per_antenna_power):.4f}")

# %%
# Applying MIMO Constraints
# -----------------------------------------------------------------------------
# Let's configure and apply appropriate constraints for the MIMO system.

# Create MIMO constraints using the factory function
# Set uniform power distribution across antennas
uniform_power = 0.25  # Total power 1.0 divided by 4 antennas
mimo_constraints = create_mimo_constraints(num_antennas=n_antennas, uniform_power=uniform_power, max_papr=4.0)  # Limit PAPR to 4 (approximately 6 dB)

# Apply constraints to the MIMO signal (convert to real first)
mimo_real = torch.cat([mimo_signal.real, mimo_signal.imag], dim=0)
constrained_mimo = mimo_constraints(mimo_real.clone())

# Extract I/Q components after constraint application
constrained_mimo_i = constrained_mimo[:n_antennas]
constrained_mimo_q = constrained_mimo[n_antennas:]

# Calculate constrained per-antenna power
constrained_per_antenna_power = []
for i in range(n_antennas):
    antenna_power_i = torch.mean(constrained_mimo_i[i] ** 2).item()
    antenna_power_q = torch.mean(constrained_mimo_q[i] ** 2).item()
    total_antenna_power = antenna_power_i + antenna_power_q
    constrained_per_antenna_power.append(total_antenna_power)

# Constrained MIMO Signal Properties:
# - Per-antenna Power: constrained power for each antenna
# - Total Power: sum of constrained antenna powers
print("\nConstrained MIMO Signal Properties:")
for i in range(n_antennas):
    print(f"  Antenna {i+1} Power: {constrained_per_antenna_power[i]:.4f}")
print(f"  Total Power: {sum(constrained_per_antenna_power):.4f}")

# Generate MIMO constraint visualization
# Calculate per-antenna properties for visualization
antenna_powers_orig = []
antenna_powers_const = []
antenna_paprs_orig = []
antenna_paprs_const = []

for i in range(n_antennas):
    # Original antenna signal
    orig_ant = mimo_signal[i]
    orig_power = torch.mean(torch.abs(orig_ant) ** 2).item()
    orig_papr = (torch.max(torch.abs(orig_ant) ** 2) / torch.mean(torch.abs(orig_ant) ** 2)).item()

    antenna_powers_orig.append(orig_power)
    antenna_paprs_orig.append(orig_papr)

    # Constrained antenna signal
    const_ant = torch.complex(constrained_mimo_i[i], constrained_mimo_q[i])
    const_power = torch.mean(torch.abs(const_ant) ** 2).item()
    const_papr = (torch.max(torch.abs(const_ant) ** 2) / torch.mean(torch.abs(const_ant) ** 2)).item()

    antenna_powers_const.append(const_power)
    antenna_paprs_const.append(const_papr)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("MIMO System Constraint Analysis", fontsize=16, fontweight="bold")

# Per-antenna power comparison
ax1 = axes[0, 0]
x_pos = np.arange(n_antennas)
width = 0.35
ax1.bar(x_pos - width / 2, antenna_powers_orig, width, label="Original", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
ax1.bar(x_pos + width / 2, antenna_powers_const, width, label="Constrained", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
ax1.axhline(y=uniform_power, color="red", linestyle="--", label=f"Target: {uniform_power:.3f}")
ax1.set_title("Per-Antenna Power Distribution")
ax1.set_xlabel("Antenna Index")
ax1.set_ylabel("Power")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"Ant {i+1}" for i in range(n_antennas)])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Per-antenna PAPR comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - width / 2, antenna_paprs_orig, width, label="Original", color=PlottingUtils.MODERN_PALETTE[2], alpha=0.7)
ax2.bar(x_pos + width / 2, antenna_paprs_const, width, label="Constrained", color=PlottingUtils.MODERN_PALETTE[3], alpha=0.7)
ax2.axhline(y=4.0, color="red", linestyle="--", label="Max PAPR: 4.0")
ax2.set_title("Per-Antenna PAPR")
ax2.set_xlabel("Antenna Index")
ax2.set_ylabel("PAPR")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"Ant {i+1}" for i in range(n_antennas)])
ax2.legend()
ax2.grid(True, alpha=0.3)

# Time series comparison for one antenna
ax3 = axes[1, 0]
antenna_idx = 0
original_antenna = mimo_signal[antenna_idx]
constrained_antenna = torch.complex(constrained_mimo_i[antenna_idx], constrained_mimo_q[antenna_idx])

t_mimo = np.arange(min(200, len(original_antenna)))
ax3.plot(t_mimo, torch.abs(original_antenna[: len(t_mimo)]).numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label=f"Original Ant {antenna_idx+1}")
ax3.plot(t_mimo, torch.abs(constrained_antenna[: len(t_mimo)]).numpy(), color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.7, label=f"Constrained Ant {antenna_idx+1}")
ax3.set_title(f"Antenna {antenna_idx+1} Magnitude Comparison")
ax3.set_xlabel("Sample Index")
ax3.set_ylabel("Magnitude")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Total power evolution
ax4 = axes[1, 1]
total_power_orig = sum(antenna_powers_orig)
total_power_const = sum(antenna_powers_const)
categories = ["Original", "Constrained"]
total_powers = [total_power_orig, total_power_const]

bars = ax4.bar(categories, total_powers, color=[PlottingUtils.MODERN_PALETTE[0], PlottingUtils.MODERN_PALETTE[1]], alpha=0.7)
ax4.axhline(y=1.0, color="red", linestyle="--", label="Target Total: 1.0")
ax4.set_title("Total MIMO System Power")
ax4.set_ylabel("Total Power")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, power in zip(bars, total_powers):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{power:.3f}", ha="center", va="bottom")

plt.show()

# %%
# Adding Spectral Constraints to MIMO
# --------------------------------------------------------------------------------------------------------------------
# Let's add spectral mask constraints to our MIMO system to simulate regulatory requirements.

# Create a spectral mask (e.g., simulating regulatory band restrictions)
n_freq = n_symbols * n_subcarriers
spectral_mask = torch.ones(n_freq)

# Create a restricted band
restricted_start = int(0.3 * n_symbols * n_subcarriers)
restricted_end = int(0.4 * n_symbols * n_subcarriers)
spectral_mask[restricted_start:restricted_end] = 0.1  # Heavy attenuation

# Apply spectral mask to one antenna for demonstration
spectral_constraint = SpectralMaskConstraint(spectral_mask)
antenna_idx = 0
constrained_antenna_spectral = spectral_constraint(torch.stack([constrained_mimo_i[antenna_idx], constrained_mimo_q[antenna_idx]]))

# Spectral Constraint Application Results:
# - Applied spectral mask to reduce emissions in restricted frequency band
# - Antenna power maintained while meeting spectral requirements
print(f"\nSpectral constraint applied to Antenna {antenna_idx+1}")
print("- Restricted frequency band: 30-40% of total bandwidth")
print("- Attenuation factor: 0.1 (20 dB reduction)")

# %%
# Part 3: Real-world Application - Complete OFDM Transmitter Constraints
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's put everything together to simulate a complete OFDM transmitter with practical constraints.

# Create a more complex OFDM signal with pilot symbols and data
n_sym = 20
n_sub = 256
cp_len = 32

# Create realistic OFDM frequency domain symbols with pilot allocation
pilots_indices = np.arange(0, n_sub, 8)  # Every 8th subcarrier is a pilot
data_indices = np.setdiff1d(np.arange(n_sub), pilots_indices)

# Initialize frequency domain signal
X_full = torch.zeros(1, n_sym, n_sub, dtype=torch.complex64)

# Add pilot symbols (known reference signals)
X_full[:, :, pilots_indices] = torch.complex(torch.ones(1, n_sym, len(pilots_indices)), torch.zeros(1, n_sym, len(pilots_indices)))

# Add random data symbols
X_full[:, :, data_indices] = torch.complex(torch.randn(1, n_sym, len(data_indices)), torch.randn(1, n_sym, len(data_indices)))

# Convert to time domain and add cyclic prefix
X_full_time = torch.fft.ifft(X_full, dim=2)
ofdm_full = []
for i in range(n_sym):
    symbol = X_full_time[:, i, :]
    cp = symbol[:, -cp_len:]
    ofdm_full.append(torch.cat([cp, symbol], dim=1))

ofdm_full_signal = torch.cat(ofdm_full, dim=1)
ofdm_iq_full = torch.cat([ofdm_full_signal.real, ofdm_full_signal.imag], dim=0)

# Measure properties of full OFDM transmitter
ofdm_full_props = measure_signal_properties(ofdm_iq_full)

# Complete OFDM Transmitter Properties:
# - Total symbols: n_sym, Subcarriers: n_sub, CP length: cp_len
# - Pilot allocation: every 8th subcarrier
# - Power: ofdm_full_props['mean_power']
# - PAPR: ofdm_full_props['papr'] (ofdm_full_props['papr_db'] dB)
print("\nComplete OFDM Transmitter Properties:")
print(f"  Total symbols: {n_sym}, Subcarriers: {n_sub}, CP length: {cp_len}")
print("  Pilot allocation: every 8th subcarrier")
print(f"  Power: {ofdm_full_props['mean_power']:.4f}")
print(f"  PAPR: {ofdm_full_props['papr']:.2f} ({ofdm_full_props['papr_db']:.2f} dB)")

# Apply comprehensive transmitter constraints
tx_constraints = combine_constraints([TotalPowerConstraint(total_power=1.0), PAPRConstraint(max_papr=5.0), PeakAmplitudeConstraint(max_amplitude=2.0)])  # Stricter PAPR for practical systems  # Hardware amplifier limits

tx_constrained = tx_constraints(ofdm_iq_full.clone())
tx_constrained_props = measure_signal_properties(tx_constrained)

# Transmitter Constraint Results:
# - Power: tx_constrained_props['mean_power']
# - PAPR: tx_constrained_props['papr'] (tx_constrained_props['papr_db'] dB)
# - Peak Amplitude: tx_constrained_props['peak_amplitude']
print("\nTransmitter Constraint Results:")
print(f"  Power: {tx_constrained_props['mean_power']:.4f}")
print(f"  PAPR: {tx_constrained_props['papr']:.2f} ({tx_constrained_props['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {tx_constrained_props['peak_amplitude']:.4f}")

# Generate final comprehensive transmitter analysis
# Prepare data for comprehensive analysis
original_power = ofdm_iq_full[0].numpy() ** 2 + ofdm_iq_full[1].numpy() ** 2
constrained_power = tx_constrained[0].numpy() ** 2 + tx_constrained[1].numpy() ** 2

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Comprehensive Multi-Constraint OFDM Transmitter Analysis", fontsize=16, fontweight="bold")

# Time domain signal comparison
ax1 = axes[0, 0]
t_samples = min(2000, len(ofdm_iq_full[0]))
t_vec = np.arange(t_samples)
ax1.plot(t_vec, ofdm_iq_full[0][:t_samples].numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1, alpha=0.7, label="Original I")
ax1.plot(t_vec, tx_constrained[0][:t_samples].numpy(), color=PlottingUtils.MODERN_PALETTE[1], linewidth=1, alpha=0.7, label="Constrained I")
ax1.set_title("I Component: Complete Transmitter")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power envelope comparison
ax2 = axes[0, 1]
ax2.plot(t_vec, original_power[:t_samples], color=PlottingUtils.MODERN_PALETTE[0], linewidth=1, alpha=0.7, label="Original")
ax2.plot(t_vec, constrained_power[:t_samples], color=PlottingUtils.MODERN_PALETTE[1], linewidth=1, alpha=0.7, label="Constrained")
ax2.axhline(y=np.mean(original_power), color=PlottingUtils.MODERN_PALETTE[0], linestyle="--", alpha=0.8, label=f"Orig Avg: {np.mean(original_power):.3f}")
ax2.axhline(y=np.mean(constrained_power), color=PlottingUtils.MODERN_PALETTE[1], linestyle="--", alpha=0.8, label=f"Const Avg: {np.mean(constrained_power):.3f}")
ax2.set_title("Power Envelope")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Instantaneous Power")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Summary statistics
ax3 = axes[1, 0]
properties = ["Power", "PAPR (dB)", "Peak Amplitude"]
original_values = [ofdm_full_props["mean_power"], ofdm_full_props["papr_db"], ofdm_full_props["peak_amplitude"]]
constrained_values = [tx_constrained_props["mean_power"], tx_constrained_props["papr_db"], tx_constrained_props["peak_amplitude"]]
targets = [1.0, 10 * np.log10(5.0), 2.0]  # Target values

x_props = np.arange(len(properties))
width = 0.25
ax3.bar(x_props - width, original_values, width, label="Original", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
ax3.bar(x_props, constrained_values, width, label="Constrained", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
ax3.bar(x_props + width, targets, width, label="Target", color=PlottingUtils.MODERN_PALETTE[2], alpha=0.7)

ax3.set_title("Constraint Satisfaction Summary")
ax3.set_xlabel("Signal Property")
ax3.set_ylabel("Value")
ax3.set_xticks(x_props)
ax3.set_xticklabels(properties)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Frequency domain analysis
ax4 = axes[1, 1]
# Compute spectrum of original and constrained signals
orig_spectrum = np.abs(np.fft.fft(ofdm_iq_full[0].numpy() + 1j * ofdm_iq_full[1].numpy())) ** 2
const_spectrum = np.abs(np.fft.fft(tx_constrained[0].numpy() + 1j * tx_constrained[1].numpy())) ** 2
freqs = np.fft.fftfreq(len(orig_spectrum))

# Plot only positive frequencies
pos_mask = freqs >= 0
ax4.semilogy(freqs[pos_mask], orig_spectrum[pos_mask], color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label="Original")
ax4.semilogy(freqs[pos_mask], const_spectrum[pos_mask], color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.7, label="Constrained")
ax4.set_title("Power Spectral Density")
ax4.set_xlabel("Normalized Frequency")
ax4.set_ylabel("PSD")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.show()

# %%
# Conclusion
# ------------------------------------
# This example demonstrated practical applications of Kaira's constraints in wireless communication systems:
#
# - **OFDM Systems**: We applied appropriate constraints to control power, PAPR, and peak amplitude
#   for realistic OFDM signals with cyclic prefixes and guard bands
#
# - **MIMO Systems**: We demonstrated how to enforce per-antenna power constraints and spectral masks
#   for multi-antenna communication systems
#
# - **Complete Transmitter**: We simulated a full OFDM transmitter with pilot symbols, data allocation,
#   and comprehensive constraint application suitable for practical deployment
#
# Key insights:
# - Factory functions like `create_ofdm_constraints()` simplify constraint configuration for common scenarios
# - Sequential constraint application allows step-by-step analysis of constraint effects
# - Verification functions ensure constraints are properly satisfied within specified tolerances
# - Real-world systems require multiple simultaneous constraints to meet power, spectral, and hardware limitations
#
# These examples provide a foundation for applying Kaira constraints in practical wireless communication
# system design and optimization.
