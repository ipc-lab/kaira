"""
==========================================
Metrics Registry
==========================================

This example demonstrates how to use the metrics registry in the Kaira library.
The MetricRegistry provides a convenient way to register, access, and manage
metrics throughout your project, facilitating both built-in and custom metrics.
"""
# %%
# Imports and Setup
# --------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from kaira.metrics import (
    MetricRegistry, 
    BER, PSNR, SSIM,
    BaseMetric
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# 1. Basic Registry Usage
# ----------------------
# Initialize a new metrics registry
registry = MetricRegistry()

# Register built-in metrics
registry.register("ber", BER())
registry.register("psnr", PSNR())
registry.register("ssim", SSIM())

# Print all registered metrics
print("Registered Metrics:")
for name in registry.get_metric_names():
    print(f"- {name}")

# %%
# 2. Accessing and Using Registered Metrics
# ----------------------------------------
# Generate some test data
n_bits = 1000
bits = torch.randint(0, 2, (1, n_bits))
# Introduce some errors (5% error rate)
error_probability = 0.05
errors = torch.rand(1, n_bits) < error_probability
received_bits = torch.logical_xor(bits, errors).int()

# Use a registered metric
ber_value = registry["ber"](received_bits, bits)
print(f"Measured BER: {ber_value.item():.5f}")

# %%
# 3. Creating and Registering Custom Metrics
# -----------------------------------------
# Define a custom metric
class BitsPerSecond(BaseMetric):
    """Metric to calculate bits per second throughput."""
    
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def forward(self, num_bits, time_seconds):
        """Calculate bits per second.
        
        Args:
            num_bits (torch.Tensor): Number of bits transmitted
            time_seconds (torch.Tensor): Time in seconds
        
        Returns:
            torch.Tensor: Bits per second
        """
        # Ensure time is not zero
        time_seconds = torch.clamp(time_seconds, min=1e-6)
        return num_bits / time_seconds

# Register the custom metric
registry.register("throughput", BitsPerSecond())

# Test the custom metric
bits_transmitted = torch.tensor([1000.0])
transmission_time = torch.tensor([0.1])  # 0.1 seconds

throughput = registry["throughput"](bits_transmitted, transmission_time)
print(f"Throughput: {throughput.item():.1f} bits per second")

# %%
# 4. Registering Metrics with Custom Names
# ---------------------------------------
# You can register the same metric type multiple times with different names

# Register PSNR with a custom name
registry.register("psnr_custom", PSNR(name="Custom PSNR"))

# Check if the metric exists
print(f"Is 'psnr_custom' registered? {registry.has_metric('psnr_custom')}")

# Get the metric by name
psnr_custom = registry["psnr_custom"]
print(f"Retrieved metric name: {psnr_custom.name}")

# %%
# 5. Registering Metrics with Different Parameters
# ----------------------------------------------
# Create a custom parameterized metric
class ParameterizedBER(BER):
    """BER metric with a threshold parameter."""
    
    def __init__(self, threshold=0.5, name=None):
        super().__init__(name=name)
        self.threshold = threshold
    
    def forward(self, y_pred, y_true):
        """Apply threshold before calculating BER."""
        thresholded_pred = (y_pred > self.threshold).float()
        return super().forward(thresholded_pred, y_true)

# Register BER metrics with different thresholds
registry.register("ber_low_threshold", ParameterizedBER(threshold=0.3))
registry.register("ber_high_threshold", ParameterizedBER(threshold=0.7))

# Test with soft bits (probabilistic values)
soft_bits = torch.rand(1, n_bits)  # Random values between 0 and 1
true_bits = torch.randint(0, 2, (1, n_bits))

ber_low = registry["ber_low_threshold"](soft_bits, true_bits)
ber_high = registry["ber_high_threshold"](soft_bits, true_bits)
ber_default = registry["ber"](soft_bits > 0.5, true_bits)

print(f"BER with threshold=0.3: {ber_low.item():.5f}")
print(f"BER with threshold=0.5 (default): {ber_default.item():.5f}")
print(f"BER with threshold=0.7: {ber_high.item():.5f}")

# %%
# Visualize the effect of different thresholds
thresholds = np.linspace(0.1, 0.9, 9)
ber_values = []

for threshold in thresholds:
    metric = ParameterizedBER(threshold=threshold)
    ber_values.append(metric(soft_bits, true_bits).item())

plt.figure(figsize=(10, 6))
plt.plot(thresholds, ber_values, 'bo-')
plt.grid(True)
plt.xlabel('Decision Threshold')
plt.ylabel('BER')
plt.title('Effect of Decision Threshold on BER')

# Add data points with labels
for i, (x, y) in enumerate(zip(thresholds, ber_values)):
    plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                 xytext=(0, 10), ha='center')

plt.tight_layout()
plt.show()

# %%
# 6. Practical Example: Communication System Evaluation Framework
# -------------------------------------------------------------
# Create a system evaluation class that uses the metrics registry

class SystemEvaluator:
    """Framework for evaluating communication system performance."""
    
    def __init__(self, metrics_registry=None):
        if metrics_registry is None:
            metrics_registry = MetricRegistry()
        self.registry = metrics_registry
        self.results = {}
    
    def register_metric(self, name, metric):
        """Register a new metric."""
        self.registry.register(name, metric)
        return self
    
    def evaluate_all(self, **kwargs):
        """Evaluate all registered metrics with the given inputs."""
        results = {}
        for name in self.registry.get_metric_names():
            try:
                # Try to evaluate the metric with the provided inputs
                metric = self.registry[name]
                # Extract only the arguments needed by this metric
                metric_result = metric(**{k: kwargs[k] for k in kwargs if k in metric.get_expected_args()})
                results[name] = metric_result.item()
            except Exception as e:
                print(f"Could not evaluate '{name}': {str(e)}")
        
        self.results = results
        return results
    
    def plot_results(self, x_values=None, metric_names=None):
        """Plot results of multiple evaluations."""
        if not hasattr(self, 'results_history'):
            print("No evaluation history to plot.")
            return
        
        if metric_names is None:
            metric_names = list(self.results_history[0].keys())
        
        plt.figure(figsize=(12, 6))
        for name in metric_names:
            values = [result.get(name, float('nan')) for result in self.results_history]
            plt.plot(x_values, values, 'o-', label=name)
        
        plt.grid(True)
        plt.xlabel('Evaluation Parameter')
        plt.ylabel('Metric Value')
        plt.title('System Performance Metrics')
        plt.legend()
        plt.tight_layout()
        plt.show()

# %%
# Add methods to BaseMetric to help with our evaluation framework
def dummy_get_expected_args(self):
    """Return expected argument names for each metric."""
    # For demo purposes - in a real implementation, this would be properly handled
    if isinstance(self, BER):
        return ['received_bits', 'true_bits']
    elif isinstance(self, PSNR) or isinstance(self, SSIM):
        return ['received_image', 'original_image']
    elif isinstance(self, BitsPerSecond):
        return ['num_bits', 'time_seconds']
    else:
        return []

# Add method to BaseMetric class instance
BaseMetric.get_expected_args = dummy_get_expected_args

# %%
# Create a complete evaluation framework
evaluator = SystemEvaluator()

# Register metrics
evaluator.register_metric("ber", BER())
evaluator.register_metric("throughput", BitsPerSecond())

# Prepare test data for a simple transmission scenario
true_bits = torch.randint(0, 2, (1, 1000))
received_bits = true_bits.clone()
# Add some random bit flips
error_mask = torch.rand(1, 1000) < 0.05  # 5% error rate
received_bits = torch.logical_xor(received_bits, error_mask).int()

# Transmission parameters
transmission_time = torch.tensor([0.1])  # seconds
num_bits = torch.tensor([1000.0])  # number of bits

# Evaluate
results = evaluator.evaluate_all(
    true_bits=true_bits,
    received_bits=received_bits,
    time_seconds=transmission_time,
    num_bits=num_bits
)

# Print results
print("\nSystem Evaluation Results:")
for name, value in results.items():
    print(f"{name}: {value:.5f}")

# %%
# 7. Dynamic Metric Creation and Registration
# ------------------------------------------
# Create and register metrics on-the-fly

# Define a function to create a scaled metric
def create_scaled_metric(base_metric_class, scale_factor, name=None):
    """Create a metric that scales the result of another metric."""
    
    class ScaledMetric(base_metric_class):
        def __init__(self, scale=scale_factor, metric_name=name):
            super().__init__(name=metric_name)
            self.scale = scale
        
        def forward(self, *args, **kwargs):
            # Get the original result and scale it
            result = super().forward(*args, **kwargs)
            return result * self.scale
    
    return ScaledMetric()

# Create a new registry
dynamic_registry = MetricRegistry()

# Create and register metrics dynamically
for scale in [0.5, 1.0, 2.0]:
    metric_name = f"scaled_ber_{scale}"
    scaled_metric = create_scaled_metric(BER, scale, name=metric_name)
    dynamic_registry.register(metric_name, scaled_metric)

# Test the dynamically created metrics
for name in dynamic_registry.get_metric_names():
    metric = dynamic_registry[name]
    result = metric(received_bits, true_bits)
    print(f"{name}: {result.item():.5f}")

# %%
# Visualize the effect of scaling
plt.figure(figsize=(8, 5))
scales = [0.5, 1.0, 2.0]
results = [dynamic_registry[f"scaled_ber_{s}"](received_bits, true_bits).item() for s in scales]

plt.bar(scales, results)
plt.grid(axis='y', alpha=0.3)
plt.xlabel('Scale Factor')
plt.ylabel('Scaled BER')
plt.title('Effect of Scaling on BER Metric')

# Add values on top of bars
for i, (x, y) in enumerate(zip(scales, results)):
    plt.text(x, y + 0.001, f"{y:.5f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# Conclusion
# ------------------
# This example demonstrated:
#
# 1. How to use the MetricRegistry to manage metrics in Kaira
# 2. Registering and accessing built-in metrics
# 3. Creating and registering custom metrics
# 4. Using parametrized metrics with the registry
# 5. Building a complete system evaluation framework
# 6. Dynamic metric creation and registration
#
# Key takeaways:
#
# - The metrics registry provides a flexible way to manage metrics
# - Custom metrics can be easily created and integrated
# - Registration enables using metrics by name throughout your code
# - Parameterized metrics allow for flexible evaluation strategies
# - System evaluation frameworks can leverage the registry for comprehensive analysis
