"""
==========================================================================================================================================================================
Poisson Channel for Signal-Dependent Noise
==========================================================================================================================================================================

This example demonstrates the PoissonChannel in Kaira, which models signal-dependent
noise commonly found in optical systems and photon-counting detectors. Unlike AWGN
where noise is independent of signal intensity, Poisson noise increases with 
signal strength, making it essential for accurate modeling of optical communications
and imaging systems.
"""

# %%
# Imports and Setup
# ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from kaira.channels import AWGNChannel, PoissonChannel, PerfectChannel
from kaira.utils import snr_to_noise_power
from kaira.metrics import BitErrorRate

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Create Test Images
# ---------------------------------------------------------
# Let's create some simple test images with varying intensities to observe
# the signal-dependent nature of Poisson noise.

# Create a ramp test pattern (increasing intensity)
ramp_size = 256
ramp = np.linspace(0.1, 1.0, ramp_size)
ramp_image = np.tile(ramp.reshape(1, -1), (ramp_size // 8, 1))

# Create a gradient test pattern
x = np.linspace(-3, 3, ramp_size)
y = np.linspace(-3, 3, ramp_size // 8)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
gradient_image = np.exp(-R**2 / 3)

# Combine test patterns
test_image = np.vstack([ramp_image, gradient_image])

# Convert to torch tensor
input_signal = torch.from_numpy(test_image).float().unsqueeze(0)  # Add batch dimension
print(f"Input signal shape: {input_signal.shape}")
print(f"Signal range: [{input_signal.min().item():.4f}, {input_signal.max().item():.4f}]")

# %%
# Define Channel Models
# ------------------------------------------------------------------------
# We'll compare the Poisson channel with AWGNChannel at equivalent SNR levels.

# Define different rate factors for the Poisson channel
# Higher rate means more photons per pixel, resulting in less relative noise
rate_factors = [5, 20, 100]

# Create PoissonChannel models with different rate factors
poisson_channels = []
for rate in rate_factors:
    # Create Poisson channel with a specific rate factor
    poisson_channel = PoissonChannel(rate_factor=rate, normalize=True)
    poisson_channels.append((rate, poisson_channel))
    print(f"Created PoissonChannel with rate factor: {rate}")

# Also create AWGN channels with equivalent SNR
# For Poisson noise, the SNR is approximately equal to the rate factor
# (because variance equals mean for Poisson distribution)
awgn_channels = []
signal_power = 1.0  # Normalized image power
for rate in rate_factors:
    # SNR = rate for Poisson distribution
    snr_linear = rate
    noise_power = signal_power / snr_linear
    
    # Create AWGN channel
    awgn_channel = AWGNChannel(avg_noise_power=noise_power)
    awgn_channels.append((rate, awgn_channel))
    print(f"Created AWGNChannel with equivalent SNR: {10*np.log10(rate):.1f} dB")

# %%
# Process Test Image Through Channels
# -------------------------------------------------------------------------------------------------------------------
# Pass our test image through each channel model and visualize the results.

# Function to process through a channel and collect results
def process_through_channel(channel, input_signal):
    with torch.no_grad():
        output = channel(input_signal)
    return output.squeeze(0).numpy()

# Process through all channels
poisson_outputs = []
for rate, channel in poisson_channels:
    output = process_through_channel(channel, input_signal)
    poisson_outputs.append((rate, output))

awgn_outputs = []
for rate, channel in awgn_channels:
    output = process_through_channel(channel, input_signal)
    awgn_outputs.append((rate, output))

# %%
# Visualize Noise Characteristics
# -------------------------------------------------------------------------------------------------------------
# Let's visualize how Poisson noise differs from Gaussian (AWGN) noise.

plt.figure(figsize=(15, 12))

# Plot the original image
plt.subplot(len(rate_factors) + 1, 2, 1)
plt.imshow(test_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Empty plot for alignment
plt.subplot(len(rate_factors) + 1, 2, 2)
plt.axis('off')

# Plot each noisy image
for i, ((rate1, poisson_output), (rate2, awgn_output)) in enumerate(zip(poisson_outputs, awgn_outputs)):
    # Plot Poisson channel output
    plt.subplot(len(rate_factors) + 1, 2, 2*i + 3)
    plt.imshow(poisson_output, cmap='gray')
    plt.title(f'Poisson Channel (Rate = {rate1})')
    plt.axis('off')
    
    # Plot AWGN channel output
    plt.subplot(len(rate_factors) + 1, 2, 2*i + 4)
    plt.imshow(awgn_output, cmap='gray')
    plt.title(f'AWGN Channel (SNR = {10*np.log10(rate2):.1f} dB)')
    plt.axis('off')

plt.tight_layout()
plt.show()

# %%
# Analyze Signal-Dependent Noise
# -----------------------------------------------------------------------------------------------
# Let's extract noise patterns from both channels and analyze 
# how noise varies with signal intensity.

# Function to extract noise component
def extract_noise(output, original):
    return output - original

# Extract horizontal profiles from middle rows of each region
def get_profiles(image, original):
    h, w = original.shape
    ramp_row = h // 8  # Middle of ramp region
    gradient_row = 3 * h // 4  # Middle of gradient region
    
    ramp_profile = image[ramp_row, :]
    gradient_profile = image[gradient_row, :]
    
    # Extract corresponding noise
    ramp_noise = extract_noise(ramp_profile, original[ramp_row, :])
    gradient_noise = extract_noise(gradient_profile, original[gradient_row, :])
    
    # Signal values
    ramp_signal = original[ramp_row, :]
    gradient_signal = original[gradient_row, :]
    
    return {
        'ramp_profile': ramp_profile,
        'ramp_noise': ramp_noise,
        'ramp_signal': ramp_signal,
        'gradient_profile': gradient_profile,
        'gradient_noise': gradient_noise,
        'gradient_signal': gradient_signal
    }

# Get profiles from each output
poisson_profiles = []
for rate, output in poisson_outputs:
    profiles = get_profiles(output, test_image)
    poisson_profiles.append((rate, profiles))

awgn_profiles = []
for rate, output in awgn_outputs:
    profiles = get_profiles(output, test_image)
    awgn_profiles.append((rate, profiles))

# %%
# Plot Signal Profiles and Noise
# ------------------------------------------------------------------------------------------------------------
# Let's visualize the horizontal profiles to see how noise varies with signal intensity.

plt.figure(figsize=(15, 10))

# Choose a single rate factor for detailed analysis
analysis_idx = 1  # Middle rate factor
rate = rate_factors[analysis_idx]
poisson_profile = poisson_profiles[analysis_idx][1]
awgn_profile = awgn_profiles[analysis_idx][1]

# Plot ramp profiles
plt.subplot(2, 2, 1)
plt.plot(poisson_profile['ramp_signal'], label='Original Signal')
plt.plot(poisson_profile['ramp_profile'], 'r-', alpha=0.5, label='Poisson Output')
plt.plot(awgn_profile['ramp_profile'], 'g-', alpha=0.5, label='AWGN Output')
plt.grid(True)
plt.title(f'Ramp Signal Profile (Rate = {rate})')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.legend()

# Plot ramp noise
plt.subplot(2, 2, 2)
plt.plot(poisson_profile['ramp_signal'], poisson_profile['ramp_noise'], 'r.', alpha=0.5, 
         label='Poisson Noise')
plt.plot(awgn_profile['ramp_signal'], awgn_profile['ramp_noise'], 'g.', alpha=0.5, 
         label='AWGN Noise')
plt.grid(True)
plt.title('Noise vs. Signal Intensity')
plt.xlabel('Signal Intensity')
plt.ylabel('Noise Value')
plt.legend()

# Plot noise standard deviation vs. signal level
num_bins = 10
signal_bins = np.linspace(0.1, 1.0, num_bins)
bin_width = signal_bins[1] - signal_bins[0]

poisson_std_devs = []
awgn_std_devs = []

# Calculate binned noise standard deviation
for i in range(len(signal_bins)-1):
    bin_min, bin_max = signal_bins[i], signal_bins[i+1]
    bin_center = (bin_min + bin_max) / 2
    
    # Get indices of signal values within this bin
    p_indices = np.where((poisson_profile['ramp_signal'] >= bin_min) & 
                         (poisson_profile['ramp_signal'] < bin_max))[0]
    a_indices = np.where((awgn_profile['ramp_signal'] >= bin_min) & 
                         (awgn_profile['ramp_signal'] < bin_max))[0]
    
    # Calculate noise std dev if there are enough samples
    if len(p_indices) > 5:
        p_std = np.std(poisson_profile['ramp_noise'][p_indices])
        poisson_std_devs.append((bin_center, p_std))
    
    if len(a_indices) > 5:
        a_std = np.std(awgn_profile['ramp_noise'][a_indices])
        awgn_std_devs.append((bin_center, a_std))

# Plot noise standard deviation vs. signal level
plt.subplot(2, 2, 3)
p_centers, p_stds = zip(*poisson_std_devs) if poisson_std_devs else ([], [])
a_centers, a_stds = zip(*awgn_std_devs) if awgn_std_devs else ([], [])

plt.plot(p_centers, p_stds, 'ro-', label='Poisson Noise Std')
plt.plot(a_centers, a_stds, 'go-', label='AWGN Noise Std')

# Plot theoretical curves
x = np.linspace(0.1, 1.0, 100)
# For Poisson: std = sqrt(rate * x) / rate = sqrt(x / rate)
poisson_theory = np.sqrt(x / rate)
# For AWGN: std is constant
awgn_theory = np.sqrt(signal_power / rate) * np.ones_like(x)

plt.plot(x, poisson_theory, 'r--', alpha=0.7, label='Poisson Theory')
plt.plot(x, awgn_theory, 'g--', alpha=0.7, label='AWGN Theory')

plt.grid(True)
plt.title('Noise Standard Deviation vs. Signal Level')
plt.xlabel('Signal Intensity')
plt.ylabel('Noise Standard Deviation')
plt.legend()

# Plot noise histograms
plt.subplot(2, 2, 4)
bright_region = np.where(poisson_profile['ramp_signal'] > 0.8)[0]
dark_region = np.where(poisson_profile['ramp_signal'] < 0.2)[0]

if len(bright_region) > 0 and len(dark_region) > 0:
    plt.hist(poisson_profile['ramp_noise'][bright_region], bins=20, alpha=0.5, density=True,
             label='Poisson (Bright Region)', color='red')
    plt.hist(poisson_profile['ramp_noise'][dark_region], bins=20, alpha=0.5, density=True,
             label='Poisson (Dark Region)', color='darkred')
    plt.hist(awgn_profile['ramp_noise'][bright_region], bins=20, alpha=0.3, density=True,
             label='AWGN (Bright Region)', color='green')
    plt.hist(awgn_profile['ramp_noise'][dark_region], bins=20, alpha=0.3, density=True,
             label='AWGN (Dark Region)', color='darkgreen')

plt.grid(True)
plt.title('Noise Distribution in Bright vs. Dark Regions')
plt.xlabel('Noise Value')
plt.ylabel('Probability Density')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# %%
# Signal-to-Noise Ratio Analysis
# -------------------------------------------------------------------------------------------------------------
# Let's compare the effective SNR across different signal intensities.

plt.figure(figsize=(10, 6))

# Calculate local SNR for both channels across the ramp
signal_levels = np.linspace(0.1, 1.0, 100)
bin_width = signal_levels[1] - signal_levels[0]

poisson_snr = []
awgn_snr = []

# Function to compute local SNR within bins
def compute_local_snr(signal, noise, signal_levels):
    snr_values = []
    for i in range(len(signal_levels)-1):
        min_level = signal_levels[i]
        max_level = signal_levels[i+1]
        center = (min_level + max_level) / 2
        
        # Find points within this signal range
        indices = np.where((signal >= min_level) & (signal < max_level))[0]
        
        if len(indices) > 5:  # Need enough points for reliable statistics
            local_signal = np.mean(signal[indices])
            local_noise_var = np.var(noise[indices])
            
            if local_noise_var > 0:
                snr = local_signal**2 / local_noise_var
                snr_db = 10 * np.log10(snr)
                snr_values.append((center, snr_db))
    
    return snr_values

# Compute local SNR for both channels
poisson_snr = compute_local_snr(poisson_profile['ramp_signal'], 
                               poisson_profile['ramp_noise'],
                               signal_levels)
awgn_snr = compute_local_snr(awgn_profile['ramp_signal'], 
                            awgn_profile['ramp_noise'],
                            signal_levels)

# Plot SNR vs signal level
if poisson_snr and awgn_snr:
    p_centers, p_snr_db = zip(*poisson_snr)
    a_centers, a_snr_db = zip(*awgn_snr)
    
    plt.plot(p_centers, p_snr_db, 'ro-', label='Poisson Channel')
    plt.plot(a_centers, a_snr_db, 'go-', label='AWGN Channel')
    
    # Plot theoretical curves
    x = np.linspace(0.1, 1.0, 100)
    # For Poisson: SNR = rate * x (signal ^ 2 / signal = signal)
    poisson_theory_snr = 10 * np.log10(rate * x)
    # For AWGN: SNR = x^2 / (1/rate) = x^2 * rate
    awgn_theory_snr = 10 * np.log10(x**2 * rate)
    
    plt.plot(x, poisson_theory_snr, 'r--', alpha=0.7, label='Poisson Theory')
    plt.plot(x, awgn_theory_snr, 'g--', alpha=0.7, label='AWGN Theory')

plt.grid(True)
plt.title(f'Local SNR vs. Signal Level (Rate = {rate})')
plt.xlabel('Signal Intensity')
plt.ylabel('SNR (dB)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Application to Digital Transmission
# --------------------------------------------------------------------------------------------------------------------
# Let's demonstrate how Poisson noise impacts digital signal transmission.

# Create a binary signal (0s and 1s)
def create_binary_signal(length=500):
    # Generate random bits
    bits = np.random.randint(0, 2, length)
    
    # Convert to signal levels (use 0.2 for '0' and 0.8 for '1' to stay in valid range)
    signal = 0.2 + 0.6 * bits
    
    return torch.from_numpy(signal).float().unsqueeze(0), bits

# Generate binary signal
binary_signal, original_bits = create_binary_signal(1000)

# Process through channels with different rate factors
poisson_binary_outputs = []
for rate, channel in poisson_channels:
    with torch.no_grad():
        output = channel(binary_signal)
    poisson_binary_outputs.append((rate, output.squeeze(0).numpy()))

# Process through equivalent AWGN channels
awgn_binary_outputs = []
for rate, channel in awgn_channels:
    with torch.no_grad():
        output = channel(binary_signal)
    awgn_binary_outputs.append((rate, output.squeeze(0).numpy()))

# %%
# Plot binary signal transmission results
plt.figure(figsize=(15, 8))

# Plot a segment of the signals
segment_start = 0
segment_length = 100
segment = slice(segment_start, segment_start + segment_length)

# Plot original signal
plt.subplot(len(rate_factors) + 1, 2, 1)
plt.step(np.arange(segment_length), binary_signal.numpy()[0][segment], where='mid', 
         color='black', linewidth=2, label='Original')
plt.grid(True)
plt.title('Original Binary Signal')
plt.ylabel('Level')
plt.ylim(0, 1)

# Empty plot for alignment
plt.subplot(len(rate_factors) + 1, 2, 2)
plt.axis('off')

# Plot each channel output
for i, ((rate1, poisson_out), (rate2, awgn_out)) in enumerate(zip(
        poisson_binary_outputs, awgn_binary_outputs)):
    
    # Poisson channel output
    plt.subplot(len(rate_factors) + 1, 2, 3 + i*2)
    plt.step(np.arange(segment_length), poisson_out[segment], where='mid', 
             linewidth=1.5, color='red', alpha=0.8)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8)  # Decision threshold
    plt.grid(True)
    plt.title(f'Poisson Channel (Rate = {rate1})')
    plt.ylabel('Level')
    plt.ylim(0, 1)
    
    # AWGN channel output
    plt.subplot(len(rate_factors) + 1, 2, 4 + i*2)
    plt.step(np.arange(segment_length), awgn_out[segment], where='mid', 
             linewidth=1.5, color='blue', alpha=0.8)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8)  # Decision threshold
    plt.grid(True)
    plt.title(f'AWGN Channel (SNR = {10*np.log10(rate2):.1f} dB)')
    plt.ylabel('Level')
    plt.ylim(0, 1)
    
    # Add x-label to bottom plots
    if i == len(rate_factors) - 1:
        plt.subplot(len(rate_factors) + 1, 2, 3 + i*2)
        plt.xlabel('Bit Index')
        plt.subplot(len(rate_factors) + 1, 2, 4 + i*2)
        plt.xlabel('Bit Index')
        
plt.tight_layout()
plt.show()

# %%
# Bit Error Rate Analysis
# ------------------------------------------------------------------------------
# Calculate and compare BER for Poisson and AWGN channels using Kaira's BitErrorRate metric.

# Create a Kaira BitErrorRate metric
ber_metric = BitErrorRate()

# Convert the original bits to a torch tensor
original_bits_tensor = torch.tensor(original_bits, dtype=torch.float32)

# Calculate BER for each channel
poisson_ber_results = []
for rate, output in poisson_binary_outputs:
    # Threshold the output to get binary decisions
    output_tensor = torch.tensor(output > 0.5, dtype=torch.float32)
    # Calculate BER using Kaira's metric
    ber = ber_metric(output_tensor, original_bits_tensor).item()
    poisson_ber_results.append((rate, ber))
    print(f"Poisson Channel (Rate={rate}): BER = {ber:.6f}")

awgn_ber_results = []
for rate, output in awgn_binary_outputs:
    # Threshold the output to get binary decisions
    output_tensor = torch.tensor(output > 0.5, dtype=torch.float32)
    # Calculate BER using Kaira's metric
    ber = ber_metric(output_tensor, original_bits_tensor).item()
    awgn_ber_results.append((rate, ber))
    print(f"AWGN Channel (SNR={10*np.log10(rate):.1f} dB): BER = {ber:.6f}")

# Plot BER vs. rate factor/SNR
plt.figure(figsize=(10, 6))

poisson_rates = [r for r, _ in poisson_ber_results]
poisson_bers = [ber for _, ber in poisson_ber_results]
awgn_rates = [r for r, _ in awgn_ber_results]
awgn_bers = [ber for _, ber in awgn_ber_results]

plt.loglog(poisson_rates, poisson_bers, 'ro-', linewidth=2, label='Poisson Channel')
plt.loglog(awgn_rates, awgn_bers, 'bo-', linewidth=2, label='AWGN Channel')

# Add theoretical BER curves for comparison
snr_range = np.logspace(0, 3, 100)  # Rate factors from 1 to 1000
from scipy.special import erfc

# For AWGN with OOK: BER = 0.5*erfc(sqrt(SNR/2))
awgn_theory_ber = 0.5 * erfc(np.sqrt(snr_range / 2))

# For Poisson with OOK: approximation based on Poisson statistics
# This is simplified - actual formula depends on threshold and signal levels
poisson_theory_ber = 0.5 * (np.exp(-0.2 * snr_range) + 
                           (1 - np.exp(-(0.2 + 0.6) * snr_range)))

plt.loglog(snr_range, awgn_theory_ber, 'b--', alpha=0.7, label='AWGN Theory')
plt.loglog(snr_range, poisson_theory_ber, 'r--', alpha=0.7, label='Poisson Approx.')

plt.grid(True)
plt.xlabel('Rate Factor / SNR')
plt.ylabel('Bit Error Rate')
plt.title('BER vs. Rate Factor/SNR Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------------------------
# This example demonstrates several key characteristics of the Poisson channel:
#
# - Poisson noise is signal-dependent, with stronger signals experiencing larger 
#   absolute noise but better signal-to-noise ratio
# - In low-light conditions (small rate factor), Poisson noise dominates and can
#   significantly degrade image quality and signal detection
# - Unlike AWGN, where noise has constant variance regardless of signal level,
#   Poisson noise variance scales with signal intensity
# - For binary transmission, this means that '1' bits (high level) experience
#   more noise than '0' bits (low level)
# - This asymmetric noise behavior is fundamental to optical communication systems,
#   where photon counting is the underlying physical process
#
# The PoissonChannel in Kaira provides an accurate model for these effects,
# which is essential for simulating optical communication systems, biological imaging,
# astronomy, and other applications where photon counting is involved.