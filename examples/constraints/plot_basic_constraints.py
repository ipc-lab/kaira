"""
==========================================================================================================================================================================
Understanding Basic Power Constraints in Kaira
==========================================================================================================================================================================

This example demonstrates the usage of basic power constraints in Kaira.
We'll explore how to apply various constraints to signals and visualize their effects.
"""

# %%
# Imports and Setup
# ----------------------------------------------------------
# We start by importing the necessary modules and setting up the environment.

import matplotlib.pyplot as plt
import numpy as np
import torch

from kaira.constraints import AveragePowerConstraint, PAPRConstraint, TotalPowerConstraint
from kaira.constraints.utils import measure_signal_properties

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Create Sample Signals
# ------------------------------------------------------------------------
# Let's create some sample signals to apply constraints to.

# Create a simple sinusoidal signal
t = np.linspace(0, 1, 1000)
frequency = 5  # Hz
amplitude = 2.0
sine_signal = amplitude * np.sin(2 * np.pi * frequency * t)
sine_tensor = torch.from_numpy(sine_signal).float().reshape(1, -1)

# Create a multi-tone signal with high PAPR
frequencies = [5, 15, 25, 35]
multi_tone = np.zeros_like(t)
for freq in frequencies:
    multi_tone += np.sin(2 * np.pi * freq * t)
multi_tone_tensor = torch.from_numpy(multi_tone).float().reshape(1, -1)

# Create a random signal
random_signal = np.random.randn(1000)
random_tensor = torch.from_numpy(random_signal).float().reshape(1, -1)

# Dictionary of signals for processing
signals = {"Sine Wave": sine_tensor, "Multi-tone": multi_tone_tensor, "Random": random_tensor}

# Display properties of original signals
print("Original Signal Properties:")
for name, signal in signals.items():
    props = measure_signal_properties(signal)
    print(f"{name}:")
    print(f"  Power: {props['mean_power']:.4f}")
    print(f"  PAPR: {props['papr']:.2f} ({props['papr_db']:.2f} dB)")
    print(f"  Max Amplitude: {props['peak_amplitude']:.4f}")

# %%
# Apply Total Power Constraint
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# We'll apply a TotalPowerConstraint to normalize signals to a specific power level.

target_power = 1.0
power_constraint = TotalPowerConstraint(total_power=target_power)

power_results = {}
original_powers_dict = {}
print(f"\nApplying TotalPowerConstraint (target power = {target_power}):")
for name, signal in signals.items():
    # Store original power
    original_props = measure_signal_properties(signal)
    original_powers_dict[name] = original_props["mean_power"]

    # Apply constraint
    constrained_signal = power_constraint(signal)
    props = measure_signal_properties(constrained_signal)

    # Store for visualization
    power_results[name] = constrained_signal.squeeze().numpy()

    # Print results
    print(f"{name}:")
    print(f"  Constrained Power: {props['mean_power']:.4f}")
    print(f"  PAPR: {props['papr']:.2f} ({props['papr_db']:.2f} dB)")
    print(f"  Max Amplitude: {props['peak_amplitude']:.4f}")

# %%
# Visualize Total Power Constraint Results
# -------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Total Power Constraint Analysis", fontsize=16, fontweight="bold")

# Plot original signals
ax1 = axes[0, 0]
for i, (name, signal) in enumerate(signals.items()):
    ax1.plot(t[:200], signal.squeeze()[:200], color=PlottingUtils.MODERN_PALETTE[i], linewidth=2, label=f"{name} (Original)", alpha=0.7)
ax1.set_title("Original Signals")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot power comparison
ax2 = axes[0, 1]
signal_names = list(signals.keys())
original_powers = [original_powers_dict[name] for name in signal_names]
constrained_powers = [measure_signal_properties(torch.tensor(power_results[name]).unsqueeze(0))["mean_power"] for name in signal_names]

x_pos = np.arange(len(signal_names))
width = 0.35
ax2.bar(x_pos - width / 2, original_powers, width, label="Original", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
ax2.bar(x_pos + width / 2, constrained_powers, width, label="Constrained", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
ax2.axhline(y=target_power, color="red", linestyle="--", label=f"Target ({target_power})")
ax2.set_title("Power Comparison")
ax2.set_xlabel("Signal Type")
ax2.set_ylabel("Power")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(signal_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add constraint satisfaction info
ax3 = axes[1, 0]
satisfaction_rates = [1.0 if abs(constrained_powers[i] - target_power) < 0.1 else 0.0 for i in range(len(signal_names))]
ax3.bar(signal_names, satisfaction_rates, color=PlottingUtils.MODERN_PALETTE[2], alpha=0.7)
ax3.set_title("Constraint Satisfaction Rate")
ax3.set_xlabel("Signal Type")
ax3.set_ylabel("Satisfaction Rate")
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3)

# Power reduction plot
ax4 = axes[1, 1]
power_reductions = [(orig - const) / orig * 100 for orig, const in zip(original_powers, constrained_powers)]
ax4.bar(signal_names, power_reductions, color=PlottingUtils.MODERN_PALETTE[3], alpha=0.7)
ax4.set_title("Power Reduction (%)")
ax4.set_xlabel("Signal Type")
ax4.set_ylabel("Reduction (%)")
ax4.grid(True, alpha=0.3)

fig.show()

# %%
# Apply PAPR Constraint
# -------------------------------------------------------------------------
# Now let's apply a constraint on Peak-to-Average Power Ratio (PAPR).

max_papr = 2.0  # in linear units (approximately 3 dB)
papr_constraint = PAPRConstraint(max_papr=max_papr)

papr_results = {}
print(f"\nApplying PAPRConstraint (max PAPR = {max_papr}):")
for name, signal in signals.items():
    # Apply constraint - reshaping to handle the dimension issue
    # The constraint expects a specific tensor shape to work with torch.max()
    signal_reshaped = signal.reshape(signal.shape[0], -1)  # Ensure it's [batch, sequence]
    constrained_signal = papr_constraint(signal_reshaped)
    props = measure_signal_properties(constrained_signal)

    # Store for visualization
    papr_results[name] = constrained_signal.squeeze().numpy()

    # Print results
    print(f"{name}:")
    print(f"  Power: {props['mean_power']:.4f}")
    print(f"  Constrained PAPR: {props['papr']:.2f} ({props['papr_db']:.2f} dB)")
    print(f"  Max Amplitude: {props['peak_amplitude']:.4f}")

# %%
# Visualize PAPR Constraint Results
# ------------------------------------------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("PAPR Constraint Analysis", fontsize=16, fontweight="bold")

# Plot original vs PAPR constrained signals
ax1 = axes[0, 0]
for i, (name, signal) in enumerate(signals.items()):
    original_np = signal.squeeze().numpy()
    papr_constrained_np = papr_results[name]

    ax1.plot(t[:200], original_np[:200], color=PlottingUtils.MODERN_PALETTE[i], linewidth=2, label=f"{name} (Original)", alpha=0.7)
    ax1.plot(t[:200], papr_constrained_np[:200], color=PlottingUtils.MODERN_PALETTE[i], linewidth=2, label=f"{name} (PAPR Constrained)", linestyle="--", alpha=0.9)

ax1.set_title("Original vs PAPR Constrained Signals")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot PAPR comparison
ax2 = axes[0, 1]
signal_names = list(signals.keys())
original_paprs = [measure_signal_properties(signals[name])["papr"] for name in signal_names]
constrained_paprs = [measure_signal_properties(torch.tensor(papr_results[name]).unsqueeze(0))["papr"] for name in signal_names]

x_pos = np.arange(len(signal_names))
width = 0.35
ax2.bar(x_pos - width / 2, original_paprs, width, label="Original", color=PlottingUtils.MODERN_PALETTE[0], alpha=0.7)
ax2.bar(x_pos + width / 2, constrained_paprs, width, label="PAPR Constrained", color=PlottingUtils.MODERN_PALETTE[1], alpha=0.7)
ax2.axhline(y=max_papr, color="red", linestyle="--", label=f"Max PAPR ({max_papr})")
ax2.set_title("PAPR Comparison")
ax2.set_xlabel("Signal Type")
ax2.set_ylabel("PAPR")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(signal_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot amplitude distribution before/after constraint
ax3 = axes[1, 0]
for i, name in enumerate(signal_names):
    original_np = signals[name].squeeze().numpy()
    constrained_np = papr_results[name]

    ax3.hist(original_np, bins=30, alpha=0.5, density=True, label=f"{name} (Original)", color=PlottingUtils.MODERN_PALETTE[i])
    ax3.hist(constrained_np, bins=30, alpha=0.5, density=True, label=f"{name} (Constrained)", color=PlottingUtils.MODERN_PALETTE[i], linestyle="--", histtype="step", linewidth=2)

ax3.set_title("Amplitude Distribution")
ax3.set_xlabel("Amplitude")
ax3.set_ylabel("Density")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Power vs PAPR trade-off
ax4 = axes[1, 1]
original_powers = [measure_signal_properties(signals[name])["mean_power"] for name in signal_names]
constrained_powers = [measure_signal_properties(torch.tensor(papr_results[name]).unsqueeze(0))["mean_power"] for name in signal_names]

power_changes = [(const - orig) / orig * 100 for orig, const in zip(original_powers, constrained_powers)]
papr_reductions = [(orig - const) / orig * 100 for orig, const in zip(original_paprs, constrained_paprs)]

ax4.scatter(power_changes, papr_reductions, s=100, c=range(len(signal_names)), cmap="viridis", alpha=0.7, edgecolors="black")
for i, name in enumerate(signal_names):
    ax4.annotate(name, (power_changes[i], papr_reductions[i]), xytext=(5, 5), textcoords="offset points", fontsize=10)

ax4.set_title("Power vs PAPR Trade-off")
ax4.set_xlabel("Power Change (%)")
ax4.set_ylabel("PAPR Reduction (%)")
ax4.grid(True, alpha=0.3)

fig.show()

# %%
# Apply Average Power Constraint
# --------------------------------------------------------------------------------------------------------------------
# The AveragePowerConstraint is useful when you want to control the average power per sample.

avg_power = 0.5
avg_power_constraint = AveragePowerConstraint(average_power=avg_power)

avg_power_results = {}
print(f"\nApplying AveragePowerConstraint (average power = {avg_power}):")
for name, signal in signals.items():
    # Apply constraint
    constrained_signal = avg_power_constraint(signal)
    props = measure_signal_properties(constrained_signal)

    # Store for visualization
    avg_power_results[name] = constrained_signal.squeeze().numpy()

    # Print results
    print(f"{name}:")
    print(f"  Constrained Average Power: {props['mean_power']:.4f}")
    print(f"  PAPR: {props['papr']:.2f} ({props['papr_db']:.2f} dB)")
    print(f"  Max Amplitude: {props['peak_amplitude']:.4f}")

# %%
# Compare the Effects of Different Constraints
# -----------------------------------------------
# Let's compare how different constraints affect the same signal

fig, axes = plt.subplots(len(signals), 1, figsize=(15, 12))
if len(signals) == 1:
    axes = [axes]

for i, (name, original) in enumerate(signals.items()):
    original_np = original.squeeze().numpy()
    power_np = power_results[name]
    papr_np = papr_results[name]
    avg_power_np = avg_power_results[name]

    axes[i].plot(t, original_np, "b-", alpha=0.7, label="Original", linewidth=1.5)
    axes[i].plot(t, power_np, "g-", alpha=0.7, label=f"Total Power = {target_power}", linewidth=1.5)
    axes[i].plot(t, papr_np, "r-", alpha=0.7, label=f"Max PAPR = {max_papr}", linewidth=1.5)
    axes[i].plot(t, avg_power_np, "m-", alpha=0.7, label=f"Avg Power = {avg_power}", linewidth=1.5)

    # Measure properties for display
    orig_props = measure_signal_properties(original)
    power_props = measure_signal_properties(torch.tensor(power_np).reshape(1, -1))
    papr_props = measure_signal_properties(torch.tensor(papr_np).reshape(1, -1))
    avg_props = measure_signal_properties(torch.tensor(avg_power_np).reshape(1, -1))

    axes[i].set_title(
        f"{name} - Comparison of Constraints\n"
        f'Original: Power={orig_props["mean_power"]:.2f}, PAPR={orig_props["papr_db"]:.2f} dB | '
        f'TotalPower: Power={power_props["mean_power"]:.2f}, PAPR={power_props["papr_db"]:.2f} dB | '
        f'PAPR: Power={papr_props["mean_power"]:.2f}, PAPR={papr_props["papr_db"]:.2f} dB'
    )

    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylabel("Amplitude")
    axes[i].legend()

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.show()

# %%
# Signal Properties Comparison
# --------------------------------------------
# Visualize how different constraints affect signal properties

# Convert constrained signals back to tensors for comparison
power_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in power_results.items()}
papr_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in papr_results.items()}
avg_power_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in avg_power_results.items()}

# %%
# Signal Properties Comparison
# --------------------------------------------
# Visualize how different constraints affect signal properties

# Convert constrained signals back to tensors for comparison
power_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in power_results.items()}
papr_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in papr_results.items()}
avg_power_signals = {name: torch.tensor(data).reshape(1, -1) for name, data in avg_power_results.items()}

# Compare constraints side by side
print("\n=== Signal Properties Comparison ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Signal Properties Analysis - All Constraints", fontsize=16, fontweight="bold")

# Collect properties for all signals and constraints
signal_names = list(signals.keys())
properties = {
    "Original": [measure_signal_properties(signals[name]) for name in signal_names],
    "Total Power": [measure_signal_properties(power_signals[name]) for name in signal_names],
    "PAPR": [measure_signal_properties(papr_signals[name]) for name in signal_names],
    "Avg Power": [measure_signal_properties(avg_power_signals[name]) for name in signal_names],
}

# Power comparison across all constraints
ax1 = axes[0, 0]
x_pos = np.arange(len(signal_names))
width = 0.2
for i, (constraint_name, props_list) in enumerate(properties.items()):
    powers = [props["mean_power"] for props in props_list]
    ax1.bar(x_pos + i * width, powers, width, label=constraint_name, color=PlottingUtils.MODERN_PALETTE[i], alpha=0.7)

ax1.set_title("Power Comparison Across Constraints")
ax1.set_xlabel("Signal Type")
ax1.set_ylabel("Mean Power")
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(signal_names)
ax1.legend()
ax1.grid(True, alpha=0.3)

# PAPR comparison across all constraints
ax2 = axes[0, 1]
for i, (constraint_name, props_list) in enumerate(properties.items()):
    paprs = [props["papr"] for props in props_list]
    ax2.bar(x_pos + i * width, paprs, width, label=constraint_name, color=PlottingUtils.MODERN_PALETTE[i], alpha=0.7)

ax2.set_title("PAPR Comparison Across Constraints")
ax2.set_xlabel("Signal Type")
ax2.set_ylabel("PAPR")
ax2.set_xticks(x_pos + width * 1.5)
ax2.set_xticklabels(signal_names)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Peak amplitude comparison
ax3 = axes[1, 0]
for i, (constraint_name, props_list) in enumerate(properties.items()):
    peak_amps = [props["peak_amplitude"] for props in props_list]
    ax3.bar(x_pos + i * width, peak_amps, width, label=constraint_name, color=PlottingUtils.MODERN_PALETTE[i], alpha=0.7)

ax3.set_title("Peak Amplitude Comparison")
ax3.set_xlabel("Signal Type")
ax3.set_ylabel("Peak Amplitude")
ax3.set_xticks(x_pos + width * 1.5)
ax3.set_xticklabels(signal_names)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Constraint effectiveness summary
ax4 = axes[1, 1]
# Calculate how well each constraint achieves its target
effectiveness_data = {
    "Total Power": [abs(target_power - props["mean_power"]) / target_power for props in properties["Total Power"]],
    "PAPR": [max(0, props["papr"] - max_papr) / max_papr for props in properties["PAPR"]],
    "Avg Power": [abs(avg_power - props["mean_power"]) / avg_power for props in properties["Avg Power"]],
}

x_pos_eff = np.arange(len(signal_names))
width_eff = 0.25
for i, (constraint_name, effectiveness) in enumerate(effectiveness_data.items()):
    ax4.bar(x_pos_eff + i * width_eff, effectiveness, width_eff, label=f"{constraint_name} Error", color=PlottingUtils.MODERN_PALETTE[i + 1], alpha=0.7)

ax4.set_title("Constraint Compliance (Lower is Better)")
ax4.set_xlabel("Signal Type")
ax4.set_ylabel("Relative Error")
ax4.set_xticks(x_pos_eff + width_eff)
ax4.set_xticklabels(signal_names)
ax4.legend()
ax4.grid(True, alpha=0.3)

fig.show()

# %%
# Conclusion
# ------------------------------------
# This example demonstrated how to use the basic power constraints in Kaira:
#
# - **TotalPowerConstraint**: Normalizes the signal to have a specific total power
# - **PAPRConstraint**: Limits the peak-to-average power ratio, which is important in many
#   communication systems to prevent amplifier saturation
# - **AveragePowerConstraint**: Controls the average power per sample
#
# Key observations:
# - The TotalPowerConstraint preserves the signal shape while scaling its amplitude
# - The PAPRConstraint clips peaks to limit the PAPR, which may introduce distortion
# - The AveragePowerConstraint adjusts the overall signal level
# - Each constraint affects different aspects of the signal and has trade-offs
# - Visualization helps understand the impact of each constraint on signal characteristics
#
# These constraints are essential tools for:
# - Meeting hardware limitations (amplifier saturation)
# - Satisfying regulatory requirements (power spectral density)
# - Optimizing system performance (energy efficiency)
# - Ensuring signal quality (avoiding distortion)
