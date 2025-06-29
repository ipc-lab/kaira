"""
==========================================
Quadrature Amplitude Modulation (QAM)
==========================================

This example demonstrates the usage of Quadrature Amplitude Modulation (QAM)
in the Kaira library. We'll explore different QAM orders (4-QAM, 16-QAM, 64-QAM)
and analyze their performance characteristics.
"""

import matplotlib.pyplot as plt

# %%
# Imports and Setup
# --------------------------------
import numpy as np
import torch

from kaira.channels import AWGNChannel
from kaira.metrics.signal import BER
from kaira.modulations import QAMDemodulator, QAMModulator
from kaira.modulations.utils import plot_constellation
from kaira.utils import snr_to_noise_power

# Plotting imports
from kaira.utils.plotting import PlottingUtils

PlottingUtils.setup_plotting_style()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Create QAM Modulators with Different Orders
# --------------------------------------------------------------------------------
qam_orders: list[int] = [4, 16, 64, 256]
n_symbols = 1000

modulators: dict[int, QAMModulator] = {order: QAMModulator(order=order) for order in qam_orders}  # type: ignore
demodulators: dict[int, QAMDemodulator] = {order: QAMDemodulator(order=order) for order in qam_orders}  # type: ignore

# Generate random bits for each QAM order
bits_per_symbol = {4: 2, 16: 4, 64: 6, 256: 8}  # 4-QAM (same as QPSK)  # 16-QAM  # 64-QAM  # 256-QAM

input_bits = {}
modulated_symbols = {}
for order in qam_orders:
    n_bits = bits_per_symbol[order] * n_symbols
    input_bits[order] = torch.randint(0, 2, (1, n_bits))
    modulated_symbols[order] = modulators[order](input_bits[order])

# %%
# Plot Constellation Diagrams
# -------------------------------------------------
# Comment: Visualize constellation diagrams for different QAM orders
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("QAM Constellation Diagrams", fontsize=16, fontweight="bold")

for i, order in enumerate(qam_orders):
    ax = axes.flat[i]
    symbols = modulated_symbols[order].numpy().flatten()
    ax.scatter(symbols.real, symbols.imag, color=PlottingUtils.MODERN_PALETTE[i], s=50, alpha=0.7, label=f"{order}-QAM")
    ax.set_title(f"{order}-QAM Constellation")
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")

fig.show()

# %%
# Simulate Transmission over AWGN Channel
# ---------------------------------------------------------------------
snr_db_range = np.arange(0, 31, 2)
ber_results: dict[int, list[float]] = {order: [] for order in qam_orders}

# Initialize BER metric
ber_metric = BER()

for snr_db in snr_db_range:
    noise_power = snr_to_noise_power(1.0, snr_db)
    channel = AWGNChannel(avg_noise_power=noise_power.item())

    for order in qam_orders:
        # Transmit through channel
        received = channel(modulated_symbols[order])

        # Demodulate
        demod_bits = demodulators[order](received)

        # Calculate BER
        ber = ber_metric(demod_bits, input_bits[order]).item()
        ber_results[order].append(ber)

# %%
# Plot BER vs SNR Performance
# -------------------------------------------------
# Comment: Compare BER performance across different QAM orders
ber_values = [np.array(ber_results[order]) for order in qam_orders]
labels = [f"{order}-QAM" for order in qam_orders]
fig = PlottingUtils.plot_performance_vs_snr(snr_range=snr_db_range, performance_values=ber_values, labels=labels, title="BER Performance of Different QAM Orders", ylabel="Bit Error Rate", use_log_scale=True, xlabel="SNR (dB)")
fig.show()

# %%
# Visualize Effect of Noise on 16-QAM
# -----------------------------------------------------------------
test_snr_db = [25, 15, 10]  # Explicit Python integers
n_test_symbols = 1000
qam16_mod = modulators[16]

# Generate random 16-QAM symbols
test_bits = torch.randint(0, 2, (1, 4 * n_test_symbols))  # 4 bits per symbol for 16-QAM
qam16_symbols = qam16_mod(test_bits)

# Create noisy versions for visualization
noisy_symbols = {}
for snr_db_val in test_snr_db:
    snr_db_local: int = int(snr_db_val)  # Ensure it's a Python int
    noise_power = snr_to_noise_power(1.0, float(snr_db_local))
    channel = AWGNChannel(avg_noise_power=noise_power.item())
    noisy_symbols[snr_db_local] = channel(qam16_symbols)

# Comment: Demonstrate effect of noise on 16-QAM constellation

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for idx, snr_db_val in enumerate(test_snr_db):
    ax_idx: int = int(idx)  # Ensure it's a Python int
    snr_val: int = int(snr_db_val)  # Ensure it's a Python int
    plot_constellation(noisy_symbols[snr_val].flatten(), title=f"16-QAM at {snr_val} dB SNR", marker=".", ax=axs[ax_idx])
    axs[ax_idx].grid(True)

plt.tight_layout()
fig.show()

# %%
# Spectral Efficiency Comparison
# ---------------------------------------------------
# Comment: Compare spectral efficiency across QAM orders

fig, ax = plt.subplots(figsize=(8, 5))

# Calculate spectral efficiency (bits/symbol)
spectral_efficiency = [np.log2(order) for order in qam_orders]

bars = ax.bar(range(len(qam_orders)), spectral_efficiency, color="skyblue", edgecolor="navy")
ax.set_xticks(range(len(qam_orders)))
ax.set_xticklabels([f"{order}-QAM" for order in qam_orders])
ax.set_ylabel("Spectral Efficiency (bits/symbol)")
ax.set_title("Spectral Efficiency Comparison")

for i, v in enumerate(spectral_efficiency):
    ax.text(i, v + 0.1, f"{v:.1f}", ha="center")

plt.tight_layout()
fig.show()

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. Implementation of different QAM orders using Kaira
# 2. Constellation visualization for 4-QAM, 16-QAM, and 64-QAM
# 3. BER performance analysis across different SNR levels
# 4. Effect of noise on constellation diagrams
# 5. Spectral efficiency comparison
#
# Key observations:
#
# - Higher order QAM (64-QAM) provides better spectral efficiency but worse BER performance
# - 4-QAM (QPSK) is most robust against noise but has lowest spectral efficiency
# - The trade-off between spectral efficiency and error robustness is clearly visible
# - Noise significantly affects the constellation shape, especially at lower SNR values
#
# This demonstrates the fundamental trade-off in digital modulation between
# spectral efficiency and error performance.
