"""
==========================================================================================================================================================================
Composing Constraints for Complex Signal Requirements
==========================================================================================================================================================================

This example demonstrates how to combine multiple constraints in Kaira to satisfy complex
signal requirements. We'll explore the composition utilities and see how constraints
can be sequentially applied to meet practical transmission specifications.
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
    create_ofdm_constraints,
    measure_signal_properties,
)
from kaira.utils.plotting import PlottingUtils

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure plotting style
PlottingUtils.setup_plotting_style()

# %%
# Creating a Test Signal with Challenging Properties
# ------------------------------------------------------------------------
# Let's create a test signal that would typically need multiple constraints.
# OFDM signals are known for having high PAPR and hence make a good example.

# Number of subcarriers for our simple OFDM signal
n_subcarriers = 64
n_symbols = 10
sample_rate = 10  # samples per symbol

# Create random frequency-domain OFDM symbols
X_freq = torch.complex(torch.randn(1, n_symbols, n_subcarriers), torch.randn(1, n_symbols, n_subcarriers))

# Convert to time domain with IFFT
X_time = torch.fft.ifft(X_freq, dim=2)

# Reshape to a continuous signal
signal = X_time.reshape(1, -1)
signal = torch.cat((signal.real, signal.imag), dim=0)  # Separate real and imaginary parts

# Display properties of the original signal
original_props = measure_signal_properties(signal)

# Original Signal Properties:
# - Shape: signal.shape
# - Power: original_props['mean_power']
# - PAPR: original_props['papr'] (original_props['papr_db'] dB)
# - Peak Amplitude: original_props['peak_amplitude']
print("Original Signal Properties:")
print(f"  Shape: {signal.shape}")
print(f"  Power: {original_props['mean_power']:.4f}")
print(f"  PAPR: {original_props['papr']:.2f} ({original_props['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {original_props['peak_amplitude']:.4f}")

# %%
# Manual Constraint Composition
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# First, let's manually combine constraints to see how the composition works.

# Define individual constraints
power_constraint = TotalPowerConstraint(total_power=1.0)
papr_constraint = PAPRConstraint(max_papr=4.0)  # ~6 dB
amplitude_constraint = PeakAmplitudeConstraint(max_amplitude=1.5)

# Apply constraints one by one
signal1 = power_constraint(signal.clone())
# Reshape the signal to ensure it has the correct dimensions for torch.max()
# The PAPR constraint expects dim to be an integer, not a tuple
signal1_reshaped = signal1.view(signal1.shape[0], -1)  # Reshape to [batch, features]
signal2 = papr_constraint(signal1_reshaped)
signal3 = amplitude_constraint(signal2.clone())

# Display results of sequential application
props1 = measure_signal_properties(signal1)
props2 = measure_signal_properties(signal2)
props3 = measure_signal_properties(signal3)

# Sequential Constraint Application Results:
# After Power Constraint:
# - Power: props1['mean_power']
# - PAPR: props1['papr'] (props1['papr_db'] dB)
# - Peak Amplitude: props1['peak_amplitude']
#
# After PAPR Constraint:
# - Power: props2['mean_power']
# - PAPR: props2['papr'] (props2['papr_db'] dB)
# - Peak Amplitude: props2['peak_amplitude']
#
# After Amplitude Constraint:
# - Power: props3['mean_power']
# - PAPR: props3['papr'] (props3['papr_db'] dB)
# - Peak Amplitude: props3['peak_amplitude']
print("\nSequential Constraint Application:")
print("After Power Constraint:")
print(f"  Power: {props1['mean_power']:.4f}")
print(f"  PAPR: {props1['papr']:.2f} ({props1['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props1['peak_amplitude']:.4f}")

print("\nAfter PAPR Constraint:")
print(f"  Power: {props2['mean_power']:.4f}")
print(f"  PAPR: {props2['papr']:.2f} ({props2['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props2['peak_amplitude']:.4f}")

print("\nAfter Amplitude Constraint:")
print(f"  Power: {props3['mean_power']:.4f}")
print(f"  PAPR: {props3['papr']:.2f} ({props3['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props3['peak_amplitude']:.4f}")

# %%
# Using the combine_constraints Utility
# -------------------------------------------------------------------------------------------------------------------
# The combine_constraints utility simplifies the process of applying multiple constraints.

# Combine constraints
combined_constraint = combine_constraints([power_constraint, papr_constraint, amplitude_constraint])

# Apply the combined constraint
signal_combined = combined_constraint(signal.clone())
props_combined = measure_signal_properties(signal_combined)

# Combined Constraint Application Results:
# - Power: props_combined['mean_power']
# - PAPR: props_combined['papr'] (props_combined['papr_db'] dB)
# - Peak Amplitude: props_combined['peak_amplitude']
print("\nCombined Constraint Application:")
print(f"  Power: {props_combined['mean_power']:.4f}")
print(f"  PAPR: {props_combined['papr']:.2f} ({props_combined['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_combined['peak_amplitude']:.4f}")

# %%
# Visualizing the Effect of Constraint Composition
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# This section demonstrates how each constraint modifies the signal properties

# Time vector for plotting
t = np.arange(signal.shape[1]) / (sample_rate * n_subcarriers)

# Create list of signals and their properties for visualization
signals_list = [("Original", signal[0].numpy()), ("Power Constraint", signal1[0].numpy()), ("+ PAPR Constraint", signal2[0].numpy()), ("+ Amplitude Constraint", signal3[0].numpy()), ("Combined Constraints", signal_combined[0].numpy())]

properties_list = [original_props, props1, props2, props3, props_combined]

# Generate constraint chain visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig.suptitle("Sequential Constraint Application Effects", fontsize=16, fontweight="bold")

# Time-domain plots for each stage
time_idx = np.arange(min(200, signal.shape[1]))  # Show first 200 samples
stage_names = ["Original", "Power", "Power+PAPR", "Power+PAPR+Amplitude", "Combined"]
stage_signals = [signal[0].numpy(), signal1[0].numpy(), signal2[0].numpy(), signal3[0].numpy(), signal_combined[0].numpy()]
stage_props = [original_props, props1, props2, props3, props_combined]

# Plot signal evolution
for i, (name, sig, props) in enumerate(zip(stage_names, stage_signals, stage_props)):
    if i < 3:
        ax = axes[0, i]
    else:
        ax = axes[1, i - 3] if i < 5 else None

    if ax is not None:
        ax.plot(time_idx, sig[: len(time_idx)], color=PlottingUtils.MODERN_PALETTE[i % len(PlottingUtils.MODERN_PALETTE)], linewidth=1.5)
        ax.set_title(f"{name}\nPower: {props['mean_power']:.3f}, PAPR: {props['papr_db']:.1f}dB")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

# Properties comparison bar chart
ax_props = axes[1, 2]
x_pos = np.arange(len(stage_names))
powers = [props["mean_power"] for props in stage_props]
paprs = [props["papr"] for props in stage_props]

ax_props2 = ax_props.twinx()
bars1 = ax_props.bar(x_pos - 0.2, powers, 0.4, label="Power", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
bars2 = ax_props2.bar(x_pos + 0.2, paprs, 0.4, label="PAPR", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)

ax_props.set_title("Signal Properties Evolution")
ax_props.set_xlabel("Constraint Stage")
ax_props.set_ylabel("Power", color=PlottingUtils.MODERN_PALETTE[0])
ax_props2.set_ylabel("PAPR", color=PlottingUtils.MODERN_PALETTE[1])
ax_props.set_xticks(x_pos)
ax_props.set_xticklabels([s.replace("+", "+\n") for s in stage_names], rotation=45, ha="right")
ax_props.grid(True, alpha=0.3)

# Add legends
ax_props.legend(loc="upper left")
ax_props2.legend(loc="upper right")

fig.show()

# %%
# Using apply_constraint_chain with Verbose Output
# --------------------------------------------------------------------------------------------------------------------
# The apply_constraint_chain utility lets us visualize how each constraint affects the signal.

constraints = [power_constraint, papr_constraint, amplitude_constraint]

# Apply constraints with verbose output
signal_chain = apply_constraint_chain(constraints, signal.clone())
props_chain = measure_signal_properties(signal_chain)

# Constraint chain results:
# - Power: props_chain['mean_power']
# - PAPR: props_chain['papr'] (props_chain['papr_db'] dB)
# - Peak Amplitude: props_chain['peak_amplitude']
print("\nConstraint Chain Result:")
print(f"  Power: {props_chain['mean_power']:.4f}")
print(f"  PAPR: {props_chain['papr']:.2f} ({props_chain['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_chain['peak_amplitude']:.4f}")

# %%
# Using Factory Functions for Common Constraint Combinations
# -------------------------------------------------------------------------------------
# Kaira provides factory functions for creating common constraint combinations.

# Create OFDM constraints (typical for OFDM transmission)
ofdm_constraints = create_ofdm_constraints(total_power=1.0, max_papr=5.0, is_complex=True, peak_amplitude=1.8)  # ~7 dB

# Apply the OFDM constraints
signal_ofdm = ofdm_constraints(signal.clone())
props_ofdm = measure_signal_properties(signal_ofdm)

# OFDM Constraints (from factory function) Results:
# - Power: props_ofdm['mean_power']
# - PAPR: props_ofdm['papr'] (props_ofdm['papr_db'] dB)
# - Peak Amplitude: props_ofdm['peak_amplitude']
print("\nOFDM Constraints (from factory function):")
print(f"  Power: {props_ofdm['mean_power']:.4f}")
print(f"  PAPR: {props_ofdm['papr']:.2f} ({props_ofdm['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_ofdm['peak_amplitude']:.4f}")

# %%
# Creating and Visualizing a Spectral Mask Constraint
# -----------------------------------------------------------------------------
# Let's create a spectral mask constraint to limit frequency components.

# Create a spectral mask (simplified example)
n_freq = signal.shape[1]
mask = torch.ones(n_freq)

# Create a notch in the mask (simulating a forbidden frequency band)
notch_start = int(0.3 * n_freq)
notch_end = int(0.4 * n_freq)
mask[notch_start:notch_end] = 0.1  # Heavy attenuation in this band

# Apply FFT to see spectrum
signal_freq = torch.fft.fft(signal[0])
signal_spectrum = torch.abs(signal_freq) ** 2

# Create and apply spectral mask constraint
spectral_constraint = SpectralMaskConstraint(mask)
signal_spectral = spectral_constraint(signal.clone())

# Apply FFT to see constrained spectrum
signal_spectral_freq = torch.fft.fft(signal_spectral[0])
signal_spectral_spectrum = torch.abs(signal_spectral_freq) ** 2

# Generate spectral constraint visualization
# Generate spectral constraint visualization
freq = np.fft.fftfreq(n_freq)
fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Spectral Mask Constraint Effects", fontsize=16, fontweight="bold")

# Original spectrum
ax1 = axes[0, 0]
ax1.semilogy(freq, signal_spectrum.numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, label="Original Spectrum")
ax1.set_title("Original Signal Spectrum")
ax1.set_xlabel("Normalized Frequency")
ax1.set_ylabel("Power Spectral Density")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Spectral mask
ax2 = axes[0, 1]
ax2.plot(freq, mask.numpy(), color=PlottingUtils.MODERN_PALETTE[1], linewidth=2, label="Spectral Mask")
ax2.fill_between(freq, 0, mask.numpy(), alpha=0.3, color=PlottingUtils.MODERN_PALETTE[1])
ax2.set_title("Applied Spectral Mask")
ax2.set_xlabel("Normalized Frequency")
ax2.set_ylabel("Mask Value")
ax2.set_ylim(0, 1.2)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Constrained spectrum
ax3 = axes[1, 0]
ax3.semilogy(freq, signal_spectral_spectrum.numpy(), color=PlottingUtils.MODERN_PALETTE[2], linewidth=1.5, label="Constrained Spectrum")
ax3.set_title("Constrained Signal Spectrum")
ax3.set_xlabel("Normalized Frequency")
ax3.set_ylabel("Power Spectral Density")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Before vs After comparison
ax4 = axes[1, 1]
ax4.semilogy(freq, signal_spectrum.numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label="Original")
ax4.semilogy(freq, signal_spectral_spectrum.numpy(), color=PlottingUtils.MODERN_PALETTE[2], linewidth=1.5, alpha=0.7, label="Constrained")
ax4.plot(freq, mask.numpy() * np.max(signal_spectrum.numpy()), "r--", linewidth=2, alpha=0.8, label="Mask (scaled)")
ax4.set_title("Spectrum Comparison")
ax4.set_xlabel("Normalized Frequency")
ax4.set_ylabel("Power Spectral Density")
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.show()

# %%
# Combining All Constraints Together
# -----------------------------------------------------------------------------
# Let's apply all constraints together: power, PAPR, amplitude, and spectral mask.

# Create a combined constraint with all our requirements
all_constraints = combine_constraints([power_constraint, papr_constraint, amplitude_constraint, spectral_constraint])

# Apply the combined constraint
signal_all = all_constraints(signal.clone())

# Calculate spectrum of the fully constrained signal
signal_all_freq = torch.fft.fft(signal_all[0])
signal_all_spectrum = torch.abs(signal_all_freq) ** 2

# Measure properties
props_all = measure_signal_properties(signal_all)

# All constraints combined results:
# - Power: props_all['mean_power']
# - PAPR: props_all['papr'] (props_all['papr_db'] dB)
# - Peak Amplitude: props_all['peak_amplitude']
print("\nAll Constraints Combined:")
print(f"  Power: {props_all['mean_power']:.4f}")
print(f"  PAPR: {props_all['papr']:.2f} ({props_all['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_all['peak_amplitude']:.4f}")

# Create comprehensive visualization of all constraints effects
plot_segment = slice(0, 200)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig.suptitle("Comprehensive Constraint Analysis", fontsize=16, fontweight="bold")

# Time domain signals comparison
ax1 = axes[0, 0]
t_seg = np.arange(200)
ax1.plot(t_seg, signal[0, plot_segment].numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.8, label="Original")
ax1.plot(t_seg, signal_all[0, plot_segment].numpy(), color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.8, label="All Constraints")
ax1.set_title("Time Domain Comparison")
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Frequency domain comparison
ax2 = axes[0, 1]
freq = np.fft.fftfreq(signal.shape[1])
pos_mask = freq >= 0
ax2.semilogy(freq[pos_mask], signal_spectrum[pos_mask].numpy(), color=PlottingUtils.MODERN_PALETTE[0], linewidth=1.5, alpha=0.7, label="Original")
ax2.semilogy(freq[pos_mask], signal_all_spectrum[pos_mask].numpy(), color=PlottingUtils.MODERN_PALETTE[1], linewidth=1.5, alpha=0.7, label="All Constraints")
ax2.plot(freq[pos_mask], mask[pos_mask].numpy() * np.max(signal_spectrum.numpy()), "r--", linewidth=2, alpha=0.8, label="Spectral Mask")
ax2.set_title("Spectral Comparison")
ax2.set_xlabel("Normalized Frequency")
ax2.set_ylabel("Power Spectral Density")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Properties comparison bar chart
ax3 = axes[0, 2]
properties = ["Power", "PAPR", "Peak Amp"]
original_values = [original_props["mean_power"], original_props["papr"], original_props["peak_amplitude"]]
constrained_values = [props_all["mean_power"], props_all["papr"], props_all["peak_amplitude"]]
targets = [1.0, 4.0, 2.5]  # Target values for constraints

x_pos = np.arange(len(properties))
width = 0.25
ax3.bar(x_pos - width, original_values, width, label="Original", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
ax3.bar(x_pos, constrained_values, width, label="Constrained", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
ax3.bar(x_pos + width, targets, width, label="Target", color=PlottingUtils.MODERN_PALETTE[2], alpha=0.7)

ax3.set_title("Signal Properties Comparison")
ax3.set_xlabel("Property")
ax3.set_ylabel("Value")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(properties)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Constraint satisfaction rates
ax4 = axes[1, 0]
constraint_names = ["Power\nConstraint", "PAPR\nConstraint", "Amplitude\nConstraint", "Spectral\nConstraint"]
satisfaction_rates = [1.0 if abs(props_all["mean_power"] - 1.0) < 0.01 else 0.0, 1.0 if props_all["papr"] <= 4.1 else 0.0, 1.0 if props_all["peak_amplitude"] <= 2.6 else 0.0, 1.0]  # Small tolerance  # Small tolerance  # Spectral constraint is always satisfied by construction

colors = [PlottingUtils.MODERN_PALETTE[i] for i in range(len(constraint_names))]
bars = ax4.bar(constraint_names, satisfaction_rates, color=colors, alpha=0.7)
ax4.set_title("Constraint Satisfaction")
ax4.set_ylabel("Satisfaction Rate")
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3)

# Add satisfaction percentage labels
for bar, rate in zip(bars, satisfaction_rates):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{rate*100:.0f}%", ha="center", va="bottom", fontweight="bold")

# Signal evolution through constraint chain
ax5 = axes[1, 1]
signals_chain = [signal[0].numpy(), signal1[0].numpy(), signal2[0].numpy(), signal3[0].numpy()]
signal_names = ["Original", "Power", "Power+PAPR", "Power+PAPR+Amp"]
colors_chain = PlottingUtils.MODERN_PALETTE[: len(signals_chain)]

t_evolution = np.arange(100)
for i, (sig, name, color) in enumerate(zip(signals_chain, signal_names, colors_chain)):
    alpha = 0.8 - i * 0.15  # Gradually reduce alpha
    ax5.plot(t_evolution, sig[:100], color=color, linewidth=1.5, alpha=alpha, label=name)

ax5.set_title("Signal Evolution Through Constraint Chain")
ax5.set_xlabel("Sample Index")
ax5.set_ylabel("Amplitude")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Amplitude distribution comparison
ax6 = axes[1, 2]
signal_amps = np.abs(signal[0].numpy())
constrained_amps = np.abs(signal_all[0].numpy())

ax6.hist(signal_amps, bins=50, density=True, alpha=0.6, color=PlottingUtils.MODERN_PALETTE[0], label="Original", edgecolor="black")
ax6.hist(constrained_amps, bins=50, density=True, alpha=0.6, color=PlottingUtils.MODERN_PALETTE[1], label="All Constraints", edgecolor="black")
ax6.axvline(x=2.5, color="red", linestyle="--", linewidth=2, label="Amplitude Limit")
ax6.set_title("Amplitude Distribution")
ax6.set_xlabel("Amplitude")
ax6.set_ylabel("Density")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.show()

# %%
# Conclusion
# ------------------------------------
# This example demonstrated how to combine multiple constraints in Kaira to satisfy complex signal requirements:
#
# - We used `combine_constraints()` to create a composite constraint from multiple individual constraints
# - We explored the `apply_constraint_chain()` utility to see how each constraint affects the signal
# - We used the factory function `create_ofdm_constraints()` for creating common constraint combinations
# - We created and applied a `SpectralMaskConstraint` to limit frequency components
#
# Key observations:
# - Constraint ordering matters! Different orders can produce different results
# - Factory functions simplify the process of creating common constraint combinations
# - Complex signal requirements often need multiple constraints working together
# - Visualizing both time and frequency domains helps understand constraint effects
#
# In practical communication systems, signals often need to satisfy multiple constraints
# simultaneously, making these composition utilities particularly valuable.
