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

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec

from kaira.constraints import (
    TotalPowerConstraint,
    PAPRConstraint, 
    PeakAmplitudeConstraint,
    SpectralMaskConstraint,
    IdentityConstraint
)
from kaira.constraints.utils import (
    combine_constraints, 
    create_ofdm_constraints,
    apply_constraint_chain,
    measure_signal_properties
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
X_freq = torch.complex(
    torch.randn(1, n_symbols, n_subcarriers),
    torch.randn(1, n_symbols, n_subcarriers)
)

# Convert to time domain with IFFT
X_time = torch.fft.ifft(X_freq, dim=2)

# Reshape to a continuous signal
signal = X_time.reshape(1, -1)
signal = torch.cat((signal.real, signal.imag), dim=0)  # Separate real and imaginary parts

# Display properties of the original signal
original_props = measure_signal_properties(signal)
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
combined_constraint = combine_constraints([
    power_constraint,
    papr_constraint,
    amplitude_constraint
])

# Apply the combined constraint
signal_combined = combined_constraint(signal.clone())
props_combined = measure_signal_properties(signal_combined)

print("\nCombined Constraint Application:")
print(f"  Power: {props_combined['mean_power']:.4f}")
print(f"  PAPR: {props_combined['papr']:.2f} ({props_combined['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_combined['peak_amplitude']:.4f}")

# %%
# Visualizing the Effect of Constraint Composition
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Time vector for plotting
t = np.arange(signal.shape[1]) / (sample_rate * n_subcarriers)

plt.figure(figsize=(15, 10))
signals_to_plot = [
    ("Original", signal[0].numpy()),
    ("Power Constraint", signal1[0].numpy()),
    ("+ PAPR Constraint", signal2[0].numpy()),
    ("+ Amplitude Constraint", signal3[0].numpy()),
    ("Combined Constraints", signal_combined[0].numpy())
]

# Plot a segment of the signal to see details
plot_segment = slice(0, 200)  # Adjust as needed
for i, (name, sig) in enumerate(signals_to_plot):
    plt.subplot(len(signals_to_plot), 1, i+1)
    plt.plot(t[plot_segment], sig[plot_segment], linewidth=1.5)
    
    # Calculate properties for display
    if i == 0:
        props = original_props
    elif i == 1:
        props = props1
    elif i == 2:
        props = props2
    elif i == 3:
        props = props3
    else:
        props = props_combined
        
    plt.title(f"{name}\nPower: {props['mean_power']:.4f}, PAPR: {props['papr_db']:.2f} dB, Max Amplitude: {props['peak_amplitude']:.4f}")
    plt.grid(True)
    plt.ylabel("Amplitude")
    
plt.xlabel("Time")
plt.tight_layout()
plt.show()

# %%
# Using apply_constraint_chain with Verbose Output
# --------------------------------------------------------------------------------------------------------------------
# The apply_constraint_chain utility lets us visualize how each constraint affects the signal.

constraints = [
    power_constraint,
    papr_constraint,
    amplitude_constraint
]

# Apply constraints with verbose output
signal_chain = apply_constraint_chain(constraints, signal.clone(), verbose=True)
props_chain = measure_signal_properties(signal_chain)

print("\nConstraint Chain Result:")
print(f"  Power: {props_chain['mean_power']:.4f}")
print(f"  PAPR: {props_chain['papr']:.2f} ({props_chain['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_chain['peak_amplitude']:.4f}")

# %%
# Using Factory Functions for Common Constraint Combinations
# -------------------------------------------------------------------------------------
# Kaira provides factory functions for creating common constraint combinations.

# Create OFDM constraints (typical for OFDM transmission)
ofdm_constraints = create_ofdm_constraints(
    total_power=1.0,
    max_papr=5.0,  # ~7 dB
    is_complex=True,
    peak_amplitude=1.8
)

# Apply the OFDM constraints
signal_ofdm = ofdm_constraints(signal.clone())
props_ofdm = measure_signal_properties(signal_ofdm)

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
signal_spectrum = torch.abs(signal_freq)**2

# Create and apply spectral mask constraint
spectral_constraint = SpectralMaskConstraint(mask)
signal_spectral = spectral_constraint(signal.clone())

# Apply FFT to see constrained spectrum
signal_spectral_freq = torch.fft.fft(signal_spectral[0])
signal_spectral_spectrum = torch.abs(signal_spectral_freq)**2

# Plot the spectra
plt.figure(figsize=(12, 8))
freq = np.fft.fftfreq(n_freq) * n_freq
mask_for_plot = mask.numpy() * torch.max(signal_spectrum).item()  # Scale for visualization

plt.subplot(2, 1, 1)
plt.semilogy(freq, signal_spectrum.numpy(), 'b', label='Original')
plt.semilogy(freq, mask_for_plot, 'r--', label='Spectral Mask')
plt.title("Original Signal Spectrum")
plt.grid(True)
plt.ylabel("Power")
plt.legend()

plt.subplot(2, 1, 2)
plt.semilogy(freq, signal_spectral_spectrum.numpy(), 'g', label='Constrained')
plt.semilogy(freq, mask_for_plot, 'r--', label='Spectral Mask')
plt.title("Spectrum After Spectral Mask Constraint")
plt.grid(True)
plt.xlabel("Normalized Frequency")
plt.ylabel("Power")
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Combining All Constraints Together
# -----------------------------------------------------------------------------
# Let's apply all constraints together: power, PAPR, amplitude, and spectral mask.

# Create a combined constraint with all our requirements
all_constraints = combine_constraints([
    power_constraint,
    papr_constraint,
    amplitude_constraint,
    spectral_constraint
])

# Apply the combined constraint
signal_all = all_constraints(signal.clone())

# Calculate spectrum of the fully constrained signal
signal_all_freq = torch.fft.fft(signal_all[0])
signal_all_spectrum = torch.abs(signal_all_freq)**2

# Measure properties
props_all = measure_signal_properties(signal_all)

print("\nAll Constraints Combined:")
print(f"  Power: {props_all['mean_power']:.4f}")
print(f"  PAPR: {props_all['papr']:.2f} ({props_all['papr_db']:.2f} dB)")
print(f"  Peak Amplitude: {props_all['peak_amplitude']:.4f}")

# Create visualization of all constraints
plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2)

# Time domain plots
plt.subplot(gs[0, :])
plt.plot(t[plot_segment], signal[0].numpy()[plot_segment], 'b-', label='Original')
plt.plot(t[plot_segment], signal_all[0].numpy()[plot_segment], 'r-', label='All Constraints')
plt.title(f"Time Domain - Original vs. All Constraints\n"
          f"Power: {props_all['mean_power']:.2f}, PAPR: {props_all['papr_db']:.2f} dB, Max Amplitude: {props_all['peak_amplitude']:.2f}")
plt.grid(True)
plt.legend()

# Frequency domain plots
plt.subplot(gs[1, 0])
plt.semilogy(freq, signal_spectrum.numpy(), 'b')
plt.title("Original Spectrum")
plt.grid(True)
plt.ylabel("Power")

plt.subplot(gs[1, 1])
plt.semilogy(freq, signal_all_spectrum.numpy(), 'r')
plt.semilogy(freq, mask_for_plot, 'k--', alpha=0.7)
plt.title("Constrained Spectrum")
plt.grid(True)

# Amplitude distribution plot (histogram)
plt.subplot(gs[2, :])
plt.hist(signal[0].numpy(), bins=50, alpha=0.5, label='Original')
plt.hist(signal_all[0].numpy(), bins=50, alpha=0.5, label='Constrained')
plt.axvline(x=props_all['peak_amplitude'], color='r', linestyle='--', 
           label=f'Max Amplitude: {props_all["peak_amplitude"]:.2f}')
plt.title("Amplitude Distribution")
plt.grid(True)
plt.legend()

plt.tight_layout()
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
