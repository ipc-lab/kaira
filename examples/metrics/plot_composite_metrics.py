"""
==========================================
Composite Metrics
==========================================

This example demonstrates how to use and create composite metrics
in the Kaira library. Composite metrics allow you to combine multiple
metrics into a single entity, which is useful for multi-objective
evaluation of communication systems.
"""
# %%
# Imports and Setup
# --------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from kaira.channels import AWGNChannel
from kaira.modulations import QAMModulator, QAMDemodulator
from kaira.metrics import (
    BER, PSNR, SSIM, SNR,
    CompositeMetric, BaseMetric
)
from kaira.utils import snr_to_noise_power

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# 1. Creating a Composite Metric
# -----------------------------
# We'll first create a composite metric that combines BER and SNR

# Initialize individual metrics
ber_metric = BER()
snr_metric = SNR()

# Create a composite metric
ber_snr_composite = CompositeMetric(
    metrics={"BER": ber_metric, "SNR": snr_metric}
)

# Generate some test data
n_bits = 1000
bits = torch.randint(0, 2, (1, n_bits))
# Introduce some errors (5% error rate)
error_probability = 0.05
errors = torch.rand(1, n_bits) < error_probability
received_bits = torch.logical_xor(bits, errors).int()

# For SNR calculation, we need a signal
signal = (2 * bits - 1.0).float()  # Convert 0/1 bits to -1.0/+1.0 signal
noise = 0.2 * torch.randn_like(signal)
received_signal = signal + noise

# Calculate composite metric
result = ber_snr_composite(received_bits, bits, received_signal, signal)
print("Composite Metric Result:")
print(f"BER: {result['BER'].item():.5f}")
print(f"SNR: {result['SNR'].item():.2f} dB")
print(f"Overall: {result['__all__'].item():.5f}")

# %%
# 2. Weighted Composite Metrics
# -----------------------------
# Creating a weighted composite metric with custom weights

# Create a weighted composite metric
# We're weighting BER more heavily than SNR
weighted_composite = CompositeMetric(
    metrics={"BER": ber_metric, "SNR": snr_metric},
    weights={"BER": 0.7, "SNR": 0.3},
    reduction="weighted_sum"
)

# Calculate weighted composite
result_weighted = weighted_composite(received_bits, bits, received_signal, signal)
print("\nWeighted Composite Metric Result:")
print(f"BER: {result_weighted['BER'].item():.5f}")
print(f"SNR: {result_weighted['SNR'].item():.2f} dB")
print(f"Weighted Composite: {result_weighted['__all__'].item():.5f}")

# %%
# 3. Different Reduction Methods
# -----------------------------
# Exploring different reduction methods for composite metrics

reduction_methods = ["sum", "mean", "weighted_sum", "weighted_mean", "min", "max"]
composite_results = {}

for method in reduction_methods:
    # Create composite with different reduction
    if "weighted" in method:
        composite = CompositeMetric(
            metrics={"BER": ber_metric, "SNR": snr_metric},
            weights={"BER": 0.7, "SNR": 0.3},
            reduction=method
        )
    else:
        composite = CompositeMetric(
            metrics={"BER": ber_metric, "SNR": snr_metric},
            reduction=method
        )
    
    # Calculate result
    result = composite(received_bits, bits, received_signal, signal)
    composite_results[method] = result["__all__"].item()

# Plot results for different reduction methods
plt.figure(figsize=(10, 6))
plt.bar(reduction_methods, [composite_results[m] for m in reduction_methods])
plt.title('Composite Metric Values with Different Reduction Methods')
plt.xlabel('Reduction Method')
plt.ylabel('Composite Value')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

for i, method in enumerate(reduction_methods):
    plt.text(i, composite_results[method] + 0.01, f"{composite_results[method]:.4f}",
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# 4. Practical Example: Image Transmission System Evaluation
# ---------------------------------------------------------
# We'll create a composite metric for evaluating an image transmission system

# Function to create a simple test image
def generate_test_image(size=64):
    """Generate a synthetic test image"""
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2 + 1)
    # Normalize to [0, 1]
    z = (z - z.min()) / (z.max() - z.min())
    return torch.from_numpy(z).float().unsqueeze(0).unsqueeze(0)

# Create test image
original_image = generate_test_image(64)

# Create a composite metric for image quality
psnr_metric = PSNR()
ssim_metric = SSIM()

image_quality_composite = CompositeMetric(
    metrics={"PSNR": psnr_metric, "SSIM": ssim_metric},
    weights={"PSNR": 0.4, "SSIM": 0.6},
    reduction="weighted_mean"
)

# Simulate transmission at different SNR levels
snr_values = np.arange(0, 31, 5)
received_images = []
psnr_results = []
ssim_results = []
composite_results = []

for snr_db in snr_values:
    # Convert SNR to linear scale for noise calculation
    snr_linear = 10**(snr_db/10)
    signal_power = torch.mean(original_image**2)
    noise_power = signal_power / snr_linear
    
    # Add noise
    noise = torch.sqrt(torch.tensor(noise_power)) * torch.randn_like(original_image)
    received_image = original_image + noise
    received_image = torch.clamp(received_image, 0, 1)
    received_images.append(received_image)
    
    # Calculate metrics
    result = image_quality_composite(received_image, original_image)
    
    psnr_results.append(result["PSNR"].item())
    ssim_results.append(result["SSIM"].item())
    composite_results.append(result["__all__"].item())

# %%
# Visualize transmitted images at different SNR levels
plt.figure(figsize=(15, 4))

# Original image
plt.subplot(1, len(snr_values) + 1, 1)
plt.imshow(original_image.squeeze().numpy(), cmap='gray')
plt.title('Original')
plt.axis('off')

# Received images at different SNRs
for i, snr_db in enumerate(snr_values):
    plt.subplot(1, len(snr_values) + 1, i + 2)
    plt.imshow(received_images[i].squeeze().numpy(), cmap='gray')
    plt.title(f'SNR = {snr_db} dB')
    plt.axis('off')

plt.tight_layout()
plt.show()

# %%
# Plot metrics vs SNR
plt.figure(figsize=(10, 6))

# Plot each metric and the composite
plt.plot(snr_values, psnr_results, 'bo-', label='PSNR (normalized)')
plt.plot(snr_values, ssim_results, 'ro-', label='SSIM')
plt.plot(snr_values, composite_results, 'go-', label='Composite (40% PSNR, 60% SSIM)')

plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Metric Value')
plt.title('Image Quality Metrics vs. SNR')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 5. Creating a Custom Composite Metric
# ------------------------------------
# Define a custom composite metric class with special reduction logic

class QualityPriceRatioMetric(CompositeMetric):
    """Custom composite metric that implements a quality-price ratio."""
    
    def __init__(self, quality_metric, cost_metric, **kwargs):
        metrics = {"quality": quality_metric, "cost": cost_metric}
        super().__init__(metrics=metrics, **kwargs)
    
    def reduce(self, results_dict):
        """Custom reduction: quality / cost ratio."""
        quality = results_dict["quality"]
        cost = results_dict["cost"]
        # Ensure cost is not zero
        cost = torch.clamp(cost, min=1e-6)
        # Calculate ratio, higher is better
        ratio = quality / cost
        
        # Store individual results
        self.last_results = {
            "quality": quality,
            "cost": cost,
            "__all__": ratio
        }
        return ratio

# %%
# Use our custom composite metric for system evaluation

# Define quality and cost metrics (adapting existing metrics for demo)
class SimplifiedQualityMetric(BaseMetric):
    def forward(self, snr):
        # Higher SNR means better quality
        return torch.log10(snr + 1)  # Log scale for quality perception

class SimplifiedCostMetric(BaseMetric):
    def forward(self, power):
        # Higher power means higher cost
        return power

# Initialize metrics
quality_metric = SimplifiedQualityMetric()
cost_metric = SimplifiedCostMetric()

# Create QoS metric
qpr_metric = QualityPriceRatioMetric(quality_metric, cost_metric)

# Evaluate system at different power levels
power_levels = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
resulting_snrs = power_levels * 2  # Simplified model: SNR = power * channel_gain

# Calculate QPR for each power level
qpr_results = []
quality_results = []
cost_results = []

for power, snr in zip(power_levels, resulting_snrs):
    result = qpr_metric(snr, power)
    qpr_results.append(result["__all__"].item())
    quality_results.append(result["quality"].item())
    cost_results.append(result["cost"].item())

# %%
# Visualize the quality-price tradeoff
plt.figure(figsize=(15, 5))

# Plot quality vs power
plt.subplot(1, 3, 1)
plt.plot(power_levels.numpy(), quality_results, 'bo-')
plt.grid(True)
plt.xlabel('Power')
plt.ylabel('Quality')
plt.title('Quality vs Power')

# Plot cost vs power
plt.subplot(1, 3, 2)
plt.plot(power_levels.numpy(), cost_results, 'ro-')
plt.grid(True)
plt.xlabel('Power')
plt.ylabel('Cost')
plt.title('Cost vs Power')

# Plot QPR vs power
plt.subplot(1, 3, 3)
plt.plot(power_levels.numpy(), qpr_results, 'go-')
plt.grid(True)
plt.xlabel('Power')
plt.ylabel('Quality-Price Ratio')
plt.title('QPR vs Power')

plt.tight_layout()
plt.show()

# %%
# Find the optimal operating point
optimal_idx = np.argmax(qpr_results)
optimal_power = power_levels[optimal_idx].item()
optimal_qpr = qpr_results[optimal_idx]

print(f"Optimal operating point:")
print(f"Power: {optimal_power}")
print(f"Quality: {quality_results[optimal_idx]}")
print(f"Cost: {cost_results[optimal_idx]}")
print(f"QPR: {optimal_qpr}")

plt.figure(figsize=(8, 6))
plt.plot(power_levels.numpy(), qpr_results, 'bo-')
plt.axvline(x=optimal_power, color='r', linestyle='--', 
           label=f'Optimal Power = {optimal_power}')
plt.grid(True)
plt.xlabel('Power')
plt.ylabel('Quality-Price Ratio')
plt.title('Optimal Operating Point')
plt.legend()
plt.show()

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. How to create and use composite metrics in kaira
# 2. Various reduction methods available for combining metrics
# 3. Using composite metrics for evaluating image transmission systems
# 4. Creating custom composite metrics with specialized reduction logic
# 5. Finding optimal operating points in systems with multiple objectives
#
# Key takeaways:
#
# - Composite metrics enable multi-objective evaluation of communication systems
# - Different reduction methods produce different combined results
# - Weighted combinations allow prioritizing certain metrics over others
# - Custom composite metrics can implement domain-specific evaluation criteria
# - Multi-objective evaluation can help identify optimal system configurations
