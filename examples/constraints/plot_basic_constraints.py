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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from kaira.constraints import (
    TotalPowerConstraint, 
    AveragePowerConstraint, 
    PAPRConstraint
)
from kaira.constraints.utils import measure_signal_properties

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
signals = {
    "Sine Wave": sine_tensor,
    "Multi-tone": multi_tone_tensor,
    "Random": random_tensor
}

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
print(f"\nApplying TotalPowerConstraint (target power = {target_power}):")
for name, signal in signals.items():
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

plt.figure(figsize=(15, 10))
for i, (name, signal) in enumerate(signals.items()):
    # Plot original signal
    plt.subplot(len(signals), 2, i*2 + 1)
    plt.plot(t, signal.squeeze().numpy(), 'b-')
    props = measure_signal_properties(signal)
    plt.title(f'Original {name}\nPower: {props["mean_power"]:.2f}, PAPR: {props["papr_db"]:.2f} dB')
    plt.grid(True)
    plt.ylabel('Amplitude')
    
    # Plot power-constrained signal
    plt.subplot(len(signals), 2, i*2 + 2)
    plt.plot(t, power_results[name], 'g-')
    plt.title(f'After TotalPowerConstraint\nPower: {target_power:.2f}')
    plt.grid(True)
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

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

plt.figure(figsize=(15, 10))
for i, (name, signal) in enumerate(signals.items()):
    # Plot original signal
    plt.subplot(len(signals), 2, i*2 + 1)
    plt.plot(t, signal.squeeze().numpy(), 'b-')
    props = measure_signal_properties(signal)
    plt.title(f'Original {name}\nPower: {props["mean_power"]:.2f}, PAPR: {props["papr_db"]:.2f} dB')
    plt.grid(True)
    plt.ylabel('Amplitude')
    
    # Plot PAPR-constrained signal
    plt.subplot(len(signals), 2, i*2 + 2)
    plt.plot(t, papr_results[name], 'r-')
    constrained_props = measure_signal_properties(torch.tensor(papr_results[name]).reshape(1, -1))
    plt.title(f'After PAPRConstraint\nPAPR: {constrained_props["papr_db"]:.2f} dB')
    plt.grid(True)
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

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

plt.figure(figsize=(15, 12))
for i, (name, original) in enumerate(signals.items()):
    original_np = original.squeeze().numpy()
    power_np = power_results[name]
    papr_np = papr_results[name]
    avg_power_np = avg_power_results[name]
    
    plt.subplot(len(signals), 1, i+1)
    plt.plot(t, original_np, 'b-', alpha=0.7, label='Original')
    plt.plot(t, power_np, 'g-', alpha=0.7, label=f'Total Power = {target_power}')
    plt.plot(t, papr_np, 'r-', alpha=0.7, label=f'Max PAPR = {max_papr}')
    plt.plot(t, avg_power_np, 'm-', alpha=0.7, label=f'Avg Power = {avg_power}')
    
    # Measure properties for display
    orig_props = measure_signal_properties(original)
    power_props = measure_signal_properties(torch.tensor(power_np).reshape(1, -1))
    papr_props = measure_signal_properties(torch.tensor(papr_np).reshape(1, -1))
    avg_props = measure_signal_properties(torch.tensor(avg_power_np).reshape(1, -1))
    
    plt.title(f'{name} - Comparison of Constraints\n'
              f'Original: Power={orig_props["mean_power"]:.2f}, PAPR={orig_props["papr_db"]:.2f} dB | '
              f'TotalPower: Power={power_props["mean_power"]:.2f}, PAPR={power_props["papr_db"]:.2f} dB | '
              f'PAPR: Power={papr_props["mean_power"]:.2f}, PAPR={papr_props["papr_db"]:.2f} dB')
    
    plt.grid(True)
    plt.ylabel('Amplitude')
    plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

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
# - The PAPRConstraint affects peaks while preserving lower amplitude portions
# - Different signals respond differently to the same constraints
#
# These constraints are fundamental building blocks in communication system design,
# particularly for signals that will be transmitted through physical channels 
# with power limitations.