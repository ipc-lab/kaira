"""
=========================================
Differential Phase-Shift Keying (DPSK)
=========================================

This example demonstrates differential modulation schemes in Kaira,
specifically DBPSK and DQPSK. Differential modulation encodes information
in the phase differences between consecutive symbols, making it robust
against phase ambiguity.
"""

# %%
# Imports and Setup
# --------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from kaira.modulations import (
    DBPSKModulator, DBPSKDemodulator,
    DQPSKModulator, DQPSKDemodulator,
)
from kaira.modulations.utils import plot_constellation
from kaira.channels import AWGNChannel, FlatFadingChannel
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Generate Test Data and Create Modulators
# ---------------------------------------------------------------------
n_symbols = 1000

# Generate random bits
bits_dbpsk = torch.randint(0, 2, (1, n_symbols))
bits_dqpsk = torch.randint(0, 2, (1, 2 * n_symbols))  # DQPSK uses 2 bits per symbol

# Create modulators and demodulators
dbpsk_mod = DBPSKModulator()
dbpsk_demod = DBPSKDemodulator()
dqpsk_mod = DQPSKModulator()
dqpsk_demod = DQPSKDemodulator()

# Modulate the data
dbpsk_symbols = dbpsk_mod(bits_dbpsk)
dqpsk_symbols = dqpsk_mod(bits_dqpsk)

# %%
# Visualize Phase Transitions
# ------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# DBPSK phase transitions
axs[0].plot(dbpsk_symbols.real.numpy().flatten()[:-1], 
            dbpsk_symbols.real.numpy().flatten()[1:], 
            'b.', alpha=0.5, label='Transitions')
axs[0].grid(True)
axs[0].set_xlabel('Current Symbol (Real)')
axs[0].set_ylabel('Next Symbol (Real)')
axs[0].set_title('DBPSK Phase Transitions')
axs[0].set_aspect('equal')

# DQPSK constellation
plot_constellation(dqpsk_symbols.flatten(),
                  title='DQPSK Constellation',
                  marker='.',
                  ax=axs[1])

# DQPSK phase transitions
angles = torch.angle(dqpsk_symbols)
axs[2].plot(angles[0, :-1].numpy(), angles[0, 1:].numpy(),
            'r.', alpha=0.5)
axs[2].set_xlabel('Current Phase (rad)')
axs[2].set_ylabel('Next Phase (rad)')
axs[2].set_title('DQPSK Phase Transitions')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# %%
# Compare Performance in AWGN and Fading Channels
# -----------------------------------------------------------------------------------
snr_db_range = np.arange(0, 21, 2)
n_trials = 100  # Number of trials for each SNR point

# Initialize arrays for storing BER results
ber_dbpsk_awgn = []
ber_dqpsk_awgn = []
ber_dbpsk_fading = []
ber_dqpsk_fading = []

for snr_db in snr_db_range:
    # Setup channels
    noise_power = snr_to_noise_power(1.0, snr_db)
    awgn_channel = AWGNChannel(avg_noise_power=noise_power)
    fading_channel = FlatFadingChannel(
        fading_type='rayleigh',     # Use Rayleigh fading
        coherence_time=10,          # Fading remains constant for 10 symbols
        avg_noise_power=noise_power
    )
    
    # Arrays for storing BER for current SNR
    ber_dbpsk_awgn_trials = []
    ber_dqpsk_awgn_trials = []
    ber_dbpsk_fading_trials = []
    ber_dqpsk_fading_trials = []
    
    for _ in range(n_trials):
        # AWGN channel transmission
        received_dbpsk_awgn = awgn_channel(dbpsk_symbols)
        received_dqpsk_awgn = awgn_channel(dqpsk_symbols)
        
        demod_dbpsk_awgn = dbpsk_demod(received_dbpsk_awgn)
        demod_dqpsk_awgn = dqpsk_demod(received_dqpsk_awgn)
        
        # Note: Differential demodulation produces one less symbol, so we skip the first bit
        ber_dbpsk_awgn_trials.append(
            torch.mean((demod_dbpsk_awgn != bits_dbpsk[:, 1:]).float()).item())
        ber_dqpsk_awgn_trials.append(
            torch.mean((demod_dqpsk_awgn != bits_dqpsk[:, 2:]).float()).item())
        
        # Fading channel transmission
        received_dbpsk_fading = fading_channel(dbpsk_symbols)
        received_dqpsk_fading = fading_channel(dqpsk_symbols)
        
        demod_dbpsk_fading = dbpsk_demod(received_dbpsk_fading)
        demod_dqpsk_fading = dqpsk_demod(received_dqpsk_fading)
        
        ber_dbpsk_fading_trials.append(
            torch.mean((demod_dbpsk_fading != bits_dbpsk[:, 1:]).float()).item())
        ber_dqpsk_fading_trials.append(
            torch.mean((demod_dqpsk_fading != bits_dqpsk[:, 2:]).float()).item())
    
    # Store average BER for current SNR
    ber_dbpsk_awgn.append(np.mean(ber_dbpsk_awgn_trials))
    ber_dqpsk_awgn.append(np.mean(ber_dqpsk_awgn_trials))
    ber_dbpsk_fading.append(np.mean(ber_dbpsk_fading_trials))
    ber_dqpsk_fading.append(np.mean(ber_dqpsk_fading_trials))

# %%
# Plot BER Performance Comparison
# ---------------------------------------------------
plt.figure(figsize=(12, 5))

# AWGN channel performance
plt.subplot(121)
plt.semilogy(snr_db_range, ber_dbpsk_awgn, 'bo-', label='DBPSK')
plt.semilogy(snr_db_range, ber_dqpsk_awgn, 'ro-', label='DQPSK')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Performance in AWGN Channel')
plt.legend()

# Fading channel performance
plt.subplot(122)
plt.semilogy(snr_db_range, ber_dbpsk_fading, 'bo-', label='DBPSK')
plt.semilogy(snr_db_range, ber_dqpsk_fading, 'ro-', label='DQPSK')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('Performance in Flat Fading Channel')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Visualize Phase Recovery
# -------------------------------------
n_test = 100
test_bits = torch.randint(0, 2, (1, n_test))
test_symbols = dbpsk_mod(test_bits)

# Add phase rotation
phase_rotation = np.pi / 3  # 60 degrees rotation
rotated_symbols = test_symbols * torch.exp(1j * torch.tensor(phase_rotation))

# Demodulate rotated symbols
demod_bits = dbpsk_demod(rotated_symbols)

# Plot original and rotated symbols
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original constellation
plot_constellation(test_symbols.flatten(),
                  title='Original DBPSK Symbols',
                  marker='.',
                  ax=axs[0])

# Rotated constellation
plot_constellation(rotated_symbols.flatten(),
                  title=f'Rotated Symbols ({phase_rotation:.1f} rad)',
                  marker='.',
                  ax=axs[1])

# Phase differences
phase_diff = torch.angle(rotated_symbols[0, 1:] * torch.conj(rotated_symbols[0, :-1]))
axs[2].hist(phase_diff.numpy(), bins=50)
axs[2].set_title('Phase Differences Distribution')
axs[2].set_xlabel('Phase Difference (rad)')
axs[2].set_ylabel('Count')
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Print demodulation accuracy despite rotation - compare with bits[1:] since differential demodulation loses first symbol
accuracy = torch.mean((demod_bits == test_bits[:, 1:]).float()).item()
print(f"Demodulation accuracy with {phase_rotation:.1f} rad rotation: {accuracy:.2%}")

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. Implementation of DBPSK and DQPSK modulation using Kaira
# 2. Visualization of phase transitions and constellation diagrams
# 3. Performance comparison in AWGN and fading channels
# 4. Phase ambiguity tolerance of differential modulation
#
# Key observations:
#
# - Differential modulation maintains performance under phase rotation
# - DBPSK offers more robust performance than DQPSK
# - Performance degradation is more severe in fading channels
# - Phase differences remain consistent despite absolute phase rotation
# - DQPSK provides higher spectral efficiency at the cost of BER performance