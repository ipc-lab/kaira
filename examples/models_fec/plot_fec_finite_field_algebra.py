"""
====================================================================
Finite Field Algebra for FEC Codes
====================================================================

This example demonstrates the essential finite field algebra operations in Kaira's
FEC module. We'll focus on the core functionality of BinaryPolynomial and FiniteBifield
classes that are fundamental to error correction codes.
"""

import matplotlib.pyplot as plt
import torch

from kaira.models.fec.algebra import BinaryPolynomial, FiniteBifield

# %%
# Setting up
# ----------------------
torch.manual_seed(42)

# %%
# Binary Polynomials - Core Operations
# ------------------------------------
# Binary polynomials are the foundation of many error correction codes.
# Let's demonstrate the essential operations.

# Create binary polynomials
p1 = BinaryPolynomial(0b1011)  # x³ + x + 1
p2 = BinaryPolynomial(0b110)  # x² + x

print(f"p1 = {p1} (binary: {bin(p1.value)})")
print(f"p2 = {p2} (binary: {bin(p2.value)})")

# Basic arithmetic operations
sum_poly = BinaryPolynomial(p1.value ^ p2.value)  # Addition in GF(2)
product = p1 * p2
quotient = p1.div(p2)
remainder = p1 % p2

print(f"p1 + p2 = {sum_poly}")
print(f"p1 * p2 = {product}")
print(f"p1 ÷ p2 = {quotient} remainder {remainder}")

# %%
# Visualizing Binary Polynomial Operations
# ----------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# Plot polynomial coefficients
def plot_polynomial(poly, ax, title, color):
    """Plot the coefficients of a binary polynomial.

    Args:
        poly (BinaryPolynomial): The polynomial to plot.
        ax (matplotlib.axes.Axes): The axes to plot on.
        title (str): The title of the plot.
        color (str): Color for the bars.
    """
    coeffs = poly.to_coefficient_list()
    degrees = list(range(len(coeffs)))

    ax.bar(degrees, coeffs, color=color, alpha=0.7, edgecolor="black")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Coefficient")
    ax.set_ylim(0, 1.2)

    # Add coefficient values on bars
    for i, v in enumerate(coeffs):
        if v > 0:
            ax.text(i, v + 0.05, str(int(v)), ha="center", fontweight="bold")


plot_polynomial(p1, ax1, f"p1 = {p1}", "#3498db")
plot_polynomial(p2, ax2, f"p2 = {p2}", "#e74c3c")

plt.tight_layout()
plt.show()

# %%
# Finite Binary Fields (GF(2^m))
# ------------------------------
# Finite fields are crucial for Reed-Solomon and BCH codes.

# Create a finite field GF(2^3)
field = FiniteBifield(3)  # GF(2^3)

print(f"Field: GF(2^{field.m})")
print(f"Field size: {field.size}")
print(f"Modulus polynomial: {field.modulus}")

# Generate field elements
elements = field.get_all_elements()
print(f"Field elements: {[elem.value for elem in elements]}")

# Demonstrate field operations
a = field(3)  # Field element with value 3
b = field(5)  # Field element with value 5
sum_field = a + b
prod_field = a * b
inv_a = a.inverse() if a.value != 0 else field(0)

print("\nField arithmetic in GF(2^3):")
print(f"{a.value} + {b.value} = {sum_field.value}")
print(f"{a.value} * {b.value} = {prod_field.value}")
print(f"inverse of {a.value} = {inv_a.value}")

# %%
# Application to Error Correction
# -------------------------------
# These algebraic structures are fundamental to advanced error correction codes.

# Example: Generate a primitive element sequence (useful for Reed-Solomon codes)
alpha = field.primitive_element()  # Get primitive element
current = field.one  # Start with unity

print(f"Powers of primitive element α = {alpha.value} in GF(2³):")
for i in range(7):
    print("α^{} = {} (binary: {})".format(i, current.value, format(current.value, "03b")))
    current = current * alpha

# %%
# Summary
# -------
# This example demonstrated the core Kaira finite field algebra functionality:
# - BinaryPolynomial operations (addition, multiplication, division)
# - FiniteBifield arithmetic in GF(2^m)
# - Practical applications to error correction codes
#
# These building blocks are essential for implementing advanced codes like
# Reed-Solomon and BCH codes in communication systems.
