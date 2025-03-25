"""
===================================
Example Title Goes Here
===================================

This example demonstrates [brief description of what this example shows].
Write 2-3 sentences that explain the purpose and value of this example.
Include any key concepts that will be covered.
"""

# %%
# Imports and Setup
# ----------------
# Explain what packages are being imported and why.
import numpy as np
import matplotlib.pyplot as plt
import torch
from kaira.[module] import [Class]

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
# Section Title
# ------------
# Each section should have an explanatory paragraph before the code.
# This helps users understand what's happening in the code that follows.

# Code goes here
x = np.linspace(0, 10, 100)
y = np.sin(x)

# %%
# Visualization
# ------------
# Explain what's being visualized and why it's relevant.

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.grid(True)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Example Plot')
plt.show()

# %%
# Conclusion
# ---------
# Summarize what was demonstrated in this example and what the reader should have learned.
# Highlight key takeaways and potential applications.
#
# Key points:
# - Bullet point 1
# - Bullet point 2
# - Bullet point 3
#
# References:
# - Reference 1
# - Reference 2
"""
This meta-comment will not appear in the rendered output but can be used to
provide additional context for maintainers of the example.
"""