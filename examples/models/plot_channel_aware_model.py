"""
=====================================================================
Channel-Aware Model Implementation
=====================================================================

This example demonstrates how to implement and use channel-aware models in Kaira.

Channel-aware models adapt their processing based on the current state of the 
communication channel. The ChannelAwareBaseModel provides standardized handling
of Channel State Information (CSI), ensuring consistent usage across different 
model implementations.

This example shows:
1. How to implement simple channel-aware models
2. How to properly handle CSI in different formats
3. How to use the utility methods provided by the base class
4. How channel-aware models can be composed
"""

# sphinx_gallery_thumbnail_number = 1
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel
from kaira.models.base import CSIFormat, ChannelAwareBaseModel
from kaira.constraints import PowerNormalization

# %%
# Basic Channel-Aware Model Implementation
# ---------------------------------------
# 
# First, we'll implement a simple channel-aware model that adapts its processing
# based on CSI. This model implements a channel-aware gain factor that compensates
# for channel attenuation.


class SimpleChannelAwareModel(ChannelAwareBaseModel):
    """A simple channel-aware model that applies adaptive gain based on CSI.
    
    This model demonstrates basic CSI usage by scaling the input based on the 
    provided channel state information. It implements adaptive gain to compensate
    for channel attenuation.
    """
    
    def __init__(self):
        """Initialize the model."""
        super().__init__(
            expected_csi_dims=(1,),  # Expected CSI shape: [batch_size, 1]
            expected_csi_format=CSIFormat.LINEAR,  # Expect linear-scale CSI
            csi_min_value=0.001,  # Minimum expected CSI value
            csi_max_value=10.0,   # Maximum expected CSI value
        )
        
        # Simple layer for processing
        self.dense = nn.Linear(10, 10)
        
    def forward(self, x, csi, *args, **kwargs):
        """Apply channel-aware processing to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 10]
            csi (torch.Tensor): Channel state information of shape [batch_size, 1]
            
        Returns:
            torch.Tensor: Processed output tensor
        """
        # Validate and normalize CSI
        if not self.validate_csi(csi):
            csi = self.normalize_csi(csi)
            
        # Apply adaptive gain based on CSI (compensate for channel attenuation)
        # For weak channels (low CSI values), apply higher gain
        gain = 1.0 / torch.clamp(csi, min=0.01)
        
        # Process input
        out = self.dense(x)
        
        # Apply adaptive gain
        out = out * gain
        
        return out


# %%
# Channel-Aware Composite Model
# ----------------------------
# 
# Now, we'll implement a more complex model that consists of multiple channel-aware
# components. This demonstrates how to compose channel-aware models and propagate
# CSI to submodules.


class ChannelAwareEncoder(ChannelAwareBaseModel):
    """A channel-aware encoder model."""
    
    def __init__(self, in_dim=10, latent_dim=5):
        """Initialize the encoder.
        
        Args:
            in_dim (int): Input dimension
            latent_dim (int): Latent dimension
        """
        super().__init__(expected_csi_format=CSIFormat.DB)
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 20),
            nn.ReLU(),
            nn.Linear(20, latent_dim),
        )
        
    def forward(self, x, csi, *args, **kwargs):
        """Encode the input into a latent representation.
        
        Args:
            x (torch.Tensor): Input tensor
            csi (torch.Tensor): Channel state information
            
        Returns:
            torch.Tensor: Encoded representation
        """
        # Normalize CSI to dB scale
        csi = self.normalize_csi(csi)
        
        # Adjust encoding based on CSI
        # - For poor channels (low SNR/CSI), make encoding more robust
        # - For good channels (high SNR/CSI), focus on fidelity
        robustness_factor = torch.sigmoid(-csi)  # Higher for lower SNR
        
        # Get basic encoding
        z = self.net(x)
        
        # Apply robustness scaling based on channel conditions
        # Limiting the dynamic range for poor channels
        z_scaled = torch.tanh(z * (1.0 - robustness_factor))
        
        return z_scaled


class ChannelAwareDecoder(ChannelAwareBaseModel):
    """A channel-aware decoder model."""
    
    def __init__(self, latent_dim=5, out_dim=10):
        """Initialize the decoder.
        
        Args:
            latent_dim (int): Latent dimension
            out_dim (int): Output dimension
        """
        super().__init__(expected_csi_format=CSIFormat.LINEAR)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 20),
            nn.ReLU(),
            nn.Linear(20, out_dim),
        )
        
    def forward(self, z, csi, *args, **kwargs):
        """Decode the latent representation.
        
        Args:
            z (torch.Tensor): Latent representation
            csi (torch.Tensor): Channel state information
            
        Returns:
            torch.Tensor: Decoded output
        """
        # Normalize CSI to linear scale
        csi = self.normalize_csi(csi)
        
        # Apply adaptive processing based on CSI
        # - For poor channels, apply more aggressive denoising
        # - For good channels, perform lighter processing
        denoise_strength = 1.0 / torch.clamp(csi, min=0.01)
        
        # Basic decoding
        out = self.net(z)
        
        # Apply denoising based on channel quality (simple illustrative example)
        # Clip values more aggressively for poor channels
        out_denoised = torch.tanh(out * torch.sigmoid(denoise_strength))
        
        return out_denoised


class CompositeChannelAwareModel(ChannelAwareBaseModel):
    """A composite model that combines encoder, channel, and decoder."""
    
    def __init__(self):
        """Initialize the composite model."""
        super().__init__()
        
        self.encoder = ChannelAwareEncoder()
        self.decoder = ChannelAwareDecoder()
        self.constraint = PowerNormalization()
        self.channel = AWGNChannel(snr_db=10)
        
    def forward(self, x, csi, *args, **kwargs):
        """Process input through the full pipeline.
        
        Args:
            x (torch.Tensor): Input tensor
            csi (torch.Tensor): Channel state information
            
        Returns:
            torch.Tensor: Reconstructed output
        """
        # Format CSI appropriately for encoder (which expects dB format)
        encoder_csi = self.format_csi_for_submodules(csi, self.encoder)
        
        # Encode
        z = self.encoder(x, encoder_csi)
        
        # Apply power constraint
        z_constrained = self.constraint(z)
        
        # Pass through channel
        z_received = self.channel(z_constrained)
        
        # Format CSI appropriately for decoder (which expects linear format)
        decoder_csi = self.format_csi_for_submodules(csi, self.decoder)
        
        # Decode
        out = self.decoder(z_received, decoder_csi)
        
        return out


# %%
# Testing and Visualization
# ------------------------
# 
# Now, let's test our models and visualize how they adapt to different channel conditions.

# Create test data
batch_size = 5
data = torch.randn(batch_size, 10)

# Create CSI at different qualities
csi_db_range = torch.tensor([-20, -10, 0, 10, 20]).reshape(batch_size, 1)
csi_linear = 10 ** (csi_db_range / 10)  # Convert dB to linear scale

# Create models
simple_model = SimpleChannelAwareModel()
composite_model = CompositeChannelAwareModel()

# Process data through models
with torch.no_grad():
    # Process with different CSI formats to demonstrate conversion
    simple_output_db = simple_model(data, csi_db_range)
    simple_output_linear = simple_model(data, csi_linear)
    
    composite_output_db = composite_model(data, csi_db_range)
    composite_output_linear = composite_model(data, csi_linear)

# %%
# Visualize how outputs change with CSI quality
plt.figure(figsize=(12, 8))

# Plot simple model outputs
plt.subplot(2, 1, 1)
for i in range(batch_size):
    plt.plot(simple_output_db[i].numpy(), label=f'SNR: {csi_db_range[i].item()} dB')
plt.title('Simple Channel-Aware Model Output')
plt.xlabel('Feature Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)

# Plot composite model outputs
plt.subplot(2, 1, 2)
for i in range(batch_size):
    plt.plot(composite_output_db[i].numpy(), label=f'SNR: {csi_db_range[i].item()} dB')
plt.title('Composite Channel-Aware Model Output')
plt.xlabel('Feature Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Channel-Aware Sequential Processing
# ---------------------------------
# 
# The ChannelAwareBaseModel provides utility methods to pass CSI through sequential
# modules. Let's demonstrate this functionality.


class SequentialChannelAwareModel(ChannelAwareBaseModel):
    """A model that demonstrates sequential processing with CSI."""
    
    def __init__(self):
        """Initialize the model."""
        super().__init__()
        
        # Create a mix of channel-aware and regular modules
        self.ca_module1 = SimpleChannelAwareModel()
        self.regular_module = nn.Linear(10, 10)
        self.ca_module2 = ChannelAwareEncoder(in_dim=10, latent_dim=10)
        
        # Collect modules in a list for sequential processing
        self.modules_list = [
            self.ca_module1,
            self.regular_module,
            self.ca_module2,
        ]
        
    def forward(self, x, csi, *args, **kwargs):
        """Process input sequentially with CSI.
        
        Args:
            x (torch.Tensor): Input tensor
            csi (torch.Tensor): Channel state information
            
        Returns:
            torch.Tensor: Processed output
        """
        # Use the utility method to forward through sequential modules
        return self.forward_csi_to_sequential(x, self.modules_list, csi, *args, **kwargs)


# Create and test sequential model
sequential_model = SequentialChannelAwareModel()
with torch.no_grad():
    sequential_output = sequential_model(data, csi_db_range)

# %%
# Visualize sequential model output
plt.figure(figsize=(10, 6))
for i in range(batch_size):
    plt.plot(sequential_output[i].numpy(), label=f'SNR: {csi_db_range[i].item()} dB')
plt.title('Sequential Channel-Aware Model Output')
plt.xlabel('Feature Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Extracting CSI from Channel Output
# --------------------------------
# 
# Some channels return CSI along with the processed signal. Let's demonstrate how to
# extract and use CSI from channel outputs.

# Create data and channel
data = torch.randn(batch_size, 10)
channel = AWGNChannel(snr_db=10)

# Simulate channel with CSI output
channel_output = {
    'signal': channel(data),
    'snr': torch.tensor([5.0, 7.5, 10.0, 12.5, 15.0]).reshape(batch_size, 1)
}

# Create and use model with extracted CSI
model = SimpleChannelAwareModel()
with torch.no_grad():
    # Extract CSI from channel output
    extracted_csi = model.extract_csi_from_channel_output(channel_output)
    
    # Use extracted CSI for processing
    output = model(channel_output['signal'], extracted_csi)

# %%
# Visualize output with extracted CSI
plt.figure(figsize=(10, 6))
for i in range(batch_size):
    plt.plot(output[i].numpy(), label=f'SNR: {extracted_csi[i].item()} dB')
plt.title('Model Output with Extracted CSI')
plt.xlabel('Feature Index')
plt.ylabel('Output Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Conclusion
# ---------
# 
# This example demonstrated how to implement and use channel-aware models in Kaira.
# The ChannelAwareBaseModel provides standardized handling of CSI, ensuring consistent
# usage across different model implementations. The key features demonstrated include:
# 
# 1. Creating simple and composite channel-aware models
# 2. Handling CSI in different formats (dB, linear)
# 3. Sequential processing with CSI
# 4. Extracting CSI from channel outputs
# 
# Channel-aware models are particularly useful in wireless communication systems
# where adaptive processing based on channel conditions can significantly improve
# performance.