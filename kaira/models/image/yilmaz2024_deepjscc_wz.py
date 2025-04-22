"""Implementation of DeepJSCC-WZ (Deep Joint Source-Channel Coding with Wyner-Ziv) models.

This module contains the neural network implementations for DeepJSCC-WZ as proposed in
Yilmaz et al. 2024. DeepJSCC-WZ is a deep learning-based joint source-channel coding
approach that incorporates side information at the decoder using Wyner-Ziv coding principles
for improved communication efficiency.

The Wyner-Ziv coding principle refers to lossy compression with side information available
at the decoder but not at the encoder. In this implementation, neural networks learn to
exploit correlations between transmitted data and side information available at the receiver.

The implementation includes different variants:
- Standard DeepJSCC-WZ encoder/decoder: Full-featured implementation with separate networks
  for encoding and decoding
- Small (lightweight) DeepJSCC-WZ encoder/decoder: Parameter-efficient version with shared
  parameters between encoder components
- Conditional DeepJSCC-WZ encoder/decoder: Enhanced variant where side information is
  available during both encoding and decoding (serves as performance upper bound)

All variants use adaptive feature modules (AFModule) to incorporate channel conditions
into the encoding/decoding process for channel-adaptive communication.

Reference:
    Yilmaz et al. "DeepJSCC-WZ: Deep Joint Source-Channel Coding with Wyner-Ziv", 2024
"""

from typing import Any

import torch
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
from torch import nn

from kaira.channels.base import BaseChannel
from kaira.constraints.base import BaseConstraint
from kaira.models.base import BaseModel
from kaira.models.components.afmodule import AFModule
from kaira.models.registry import ModelRegistry
from kaira.models.wyner_ziv import WynerZivModel


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZSmallEncoder(BaseModel):
    """DeepJSCC-WZ-sm Encoder Module :cite:`yilmaz2024deepjsccwz`.

    This is a lightweight version of the DeepJSCC-WZ encoder that transforms input images
    into a compressed latent representation suitable for transmission over noisy channels.
    The encoder consists of a series of residual blocks with downsampling, attention modules,
    and adaptive feature modules that incorporate channel state information (CSI).

    DeepJSCC-WZ-sm shares encoder parameters for encoding image at the transmitter and
    encoding side information at the receiver, resulting in a parameter-efficient design
    while maintaining competitive performance.

    Architecture highlights:
    - 4 stages of downsampling (factor of 16 total spatial reduction)
    - Attention mechanisms to capture important features
    - AFModule layers that adapt features based on channel conditions
    - Progressive compression: 3×H×W → M×(H/16)×(W/16)
    - Channel-aware design through CSI conditioning
    """

    def __init__(self, N: int, M: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCC-WZ-sm encoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of output channels in the final latent representation.
                     Determines the compression rate and bandwidth usage.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 2),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 2),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 2),
                AttentionBlock(M),
            ]
        )

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Process input image through the encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
                                Contains SNR or other channel quality indicators.
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Encoded representation ready for transmission.
                          Shape: [B, M, H/16, W/16], where M is the number of channels
                          specified during initialization.
        """
        csi_transmitter = torch.cat([csi, torch.zeros_like(csi)], dim=1)
        for layer in self.g_a:
            if isinstance(layer, AFModule):
                # Pass x and csi_transmitter as separate arguments
                x = layer(x, csi_transmitter, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)
        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZSmallDecoder(BaseModel):
    """DeepJSCC-WZ-sm Decoder Module :cite:`yilmaz2024deepjsccwz`.

    This lightweight decoder reconstructs the original image from the received noisy representation
    and available side information. It employs a symmetric structure to the encoder
    with upsampling operations and feature fusion with side information.

    The decoder follows a multi-scale fusion approach where the side information is
    encoded using the same encoder as the main signal, and features are fused at
    multiple scales during decoding. This approach effectively exploits correlations
    between the received signal and the side information.

    DeepJSCC-WZ-sm shares encoder parameters for encoding image at the transmitter and
    encoding side information at the receiver, providing parameter efficiency.

    Key features:
    - Progressive upsampling to restore spatial dimensions (H/16×W/16 → H×W)
    - Multi-scale side information fusion at 5 different resolution levels
    - Attention mechanisms to focus on important features
    - Channel-adaptive processing through AFModule layers
    - Residual connections for improved gradient flow
    """

    encoder: Yilmaz2024DeepJSCCWZSmallEncoder  # More specific type hint

    def __init__(self, N: int, M: int, encoder: Yilmaz2024DeepJSCCWZSmallEncoder, *args: Any, **kwargs: Any) -> None:
        """Initialize the DeepJSCC-WZ-sm decoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of input channels from the encoded representation.
                     Matches the encoder's output channel count.
            encoder (Yilmaz2024DeepJSCCWZSmallEncoder): Reference to the small encoder model for feature sharing.
                                This enables the decoder to process side information
                                using the same parameters as the main encoder.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_s = nn.ModuleList(
            [
                AttentionBlock(2 * M),
                ResidualBlock(in_ch=2 * M, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                AttentionBlock(2 * N),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=3, upsample=2),
                AFModule(3 * 2, 1),
                ResidualBlock(in_ch=3 * 2, out_ch=3),
            ]
        )

        self.encoder = encoder

    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the received representation into a reconstructed image.

        This method first processes the side information through the shared encoder,
        then progressively decodes the received signal while fusing with side information
        features at multiple scales.

        Args:
            x (torch.Tensor): Received noisy encoded representation of shape [B, M, H/16, W/16].
            x_side (torch.Tensor): Side information tensor of shape [B, 3, H, W] to assist in decoding.
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, 3, H, W].
        """
        csi_sideinfo = torch.cat([csi, torch.ones_like(csi)], dim=1)

        xs_list = []
        for idx, layer in enumerate(self.encoder.g_a):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(x_side)

            if isinstance(layer, AFModule):
                # Pass x_side and csi_sideinfo as separate arguments
                x_side = layer(x_side, csi_sideinfo, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x_side = layer(x_side, *args, **kwargs)

        xs_list.append(x_side)

        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)

            if isinstance(layer, AFModule):
                # Pass x and csi as separate arguments
                x = layer(x, csi, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)

        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZEncoder(BaseModel):
    """DeepJSCC-WZ Encoder Module :cite:`yilmaz2024deepjsccwz`.

    The full-size encoder for the DeepJSCC-WZ model that compresses input images
    into a compact latent representation. It includes two parallel encoding paths:
    g_a for processing the main input and g_a2 for potential preprocessing of side information.

    Unlike the small variant, this encoder uses separate parameters for the main signal
    and side information processing paths, potentially allowing for more specialized
    feature extraction at the cost of increased parameter count.

    Architecture highlights:
    - 4 stages of downsampling through residual blocks (16× spatial reduction)
    - Channel state information adaptation via AFModule
    - Attention mechanisms for feature refinement
    - Sophisticated feature extraction with residual connections
    - Progressive compression: 3×H×W → M×(H/16)×(W/16)
    """

    def __init__(self, N: int, M: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the full-size DeepJSCC-WZ encoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of output channels in the final latent representation.
                     Determines the compression rate and bandwidth usage.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

        self.g_a2 = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode the input image into a compact representation.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
                                Contains SNR or other channel quality indicators.
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Encoded representation ready for transmission.
                          Shape: [B, M, H/16, W/16], where M is the number of channels
                          specified during initialization.
        """
        csi_transmitter = csi

        for layer in self.g_a:
            if isinstance(layer, AFModule):
                # Pass x and csi_transmitter as separate arguments
                x = layer(x, csi_transmitter, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)

        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZDecoder(BaseModel):
    """DeepJSCC-WZ Decoder Module :cite:`yilmaz2024deepjsccwz`.

    The full-size decoder for the DeepJSCC-WZ model that reconstructs the original image
    from the received noisy representation and side information. It follows a symmetric
    structure to the encoder with progressive upsampling and feature fusion mechanisms.

    Unlike the small variant, this decoder uses a dedicated set of parameters for processing
    side information, potentially allowing for more specialized feature extraction at the
    cost of increased parameter count.

    Key features:
    - Multi-scale feature fusion with side information at 5 different resolution levels
    - Progressive spatial resolution recovery (4 upsampling stages, H/16×W/16 → H×W)
    - Attention-based feature refinement
    - Channel-adaptive processing through AFModule layers
    - Sophisticated feature reconstruction with residual connections
    """

    def __init__(self, N: int, M: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the full-size DeepJSCC-WZ decoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of input channels from the encoded representation.
                     Matches the encoder's output channel count.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_s = nn.ModuleList(
            [
                AttentionBlock(2 * M),
                ResidualBlock(in_ch=2 * M, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                AttentionBlock(2 * N),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=3, upsample=2),
                AFModule(3 * 2, 1),
                ResidualBlock(in_ch=3 * 2, out_ch=3),
            ]
        )

        # Add the g_a2 module list for processing side information
        self.g_a2 = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the received representation into a reconstructed image.

        This method first processes the side information through the g_a2 encoder path,
        then progressively decodes the received signal while fusing with side information
        features at multiple scales.

        Args:
            x (torch.Tensor): Received noisy encoded representation of shape [B, M, H/16, W/16].
            x_side (torch.Tensor): Side information tensor of shape [B, 3, H, W] to assist in decoding.
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, 3, H, W].
        """
        csi_sideinfo = csi

        # Process side information through g_a2 to extract features
        xs_list = []
        xs = x_side  # Use side information as initial input

        # First collect features from side information at different resolutions
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(xs)  # Save feature before downsampling

            if isinstance(layer, AFModule):
                # Pass xs and csi_sideinfo as separate arguments
                xs = layer(xs, csi_sideinfo, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                xs = layer(xs, *args, **kwargs)  # Apply the layer

        # Add the final feature map
        xs_list.append(xs)

        # Decode process - fuse features from side_info at multiple scales
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                # Fusion points: concatenate with side info features
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)

            if isinstance(layer, AFModule):
                # Pass x and csi as separate arguments
                x = layer(x, csi, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)

        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZConditionalEncoder(BaseModel):
    """DeepJSCC-WZ Conditional Encoder Module :cite:`yilmaz2024deepjsccwz`.

    This variant of the DeepJSCC-WZ encoder actively incorporates side information during
    the encoding process. This model is designed for scenarios where side information is available
    at both encoder and decoder, serving as an upper bound for performance comparison.

    The conditional encoder features three processing paths:
    - g_a: Main encoding path that fuses the input with side information features
    - g_a2: Processing path for side information for the decoder
    - g_a3: Auxiliary path for feature extraction from side information for the encoder

    By leveraging correlations between the main signal and side information at encoding time,
    this model achieves more efficient compression and better reconstruction quality compared
    to the standard DeepJSCC-WZ model, at the cost of requiring side information during encoding.

    Architecture highlights:
    - Early fusion of input and side information (6-channel input)
    - Multi-scale feature fusion with side information
    - 4 stages of downsampling (16× spatial reduction)
    - Channel-adaptive processing with AFModule
    """

    def __init__(self, N: int, M: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the conditional DeepJSCC-WZ encoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of output channels in the final latent representation.
                     Determines the compression rate and bandwidth usage.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_a = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=6, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=2 * N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=2 * N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=2 * N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

        self.g_a2 = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

        self.g_a3 = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                None,
                None,
                None,
            ]
        )

    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode the input image with conditional side information.

        This method processes both the main input image and side information in parallel,
        fusing features from the side information stream into the main encoding path
        at multiple scales. The side information is available at the encoder, allowing
        for more efficient compression compared to the standard DeepJSCC-WZ model.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].
            x_side (torch.Tensor): Side information tensor of shape [B, 3, H, W] used during encoding.
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Encoded representation ready for transmission.
                          Shape: [B, M, H/16, W/16], where M is the number of channels
                          specified during initialization.
        """
        xs_encoder = x_side
        csi_transmitter = csi

        for layer, layer_s in zip(self.g_a, self.g_a3):
            if isinstance(layer, ResidualBlockWithStride):
                x = torch.cat([x, xs_encoder], dim=1)

            if isinstance(layer, AFModule):
                # Pass x and csi_transmitter as separate arguments
                x = layer(x, csi_transmitter, *args, **kwargs)
                if layer_s is not None:
                    # Pass xs_encoder and csi_transmitter as separate arguments
                    xs_encoder = layer_s(xs_encoder, csi_transmitter, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)
                if layer_s is not None:
                    xs_encoder = layer_s(xs_encoder, *args, **kwargs)

        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZConditionalDecoder(BaseModel):
    """DeepJSCC-WZ Conditional Decoder Module :cite:`yilmaz2024deepjsccwz`.

    The decoder counterpart to the conditional encoder, designed to reconstruct images
    from representations created by the conditional encoder. This decoder leverages
    side information and received encoded representation to generate high-quality
    reconstructions.

    DeepJSCC-WZ Conditional is designed for scenarios where side information is available
    at both the encoder and decoder, serving as a performance upper bound. The decoder's
    architecture is optimized to work with the conditional encoder's output, where side
    information correlations have already been exploited during encoding.

    Key features:
    - Multi-scale feature fusion with side information at 5 different resolution levels
    - Progressive upsampling to restore spatial dimensions (H/16×W/16 → H×W)
    - Attention-based feature refinement
    - Channel-adaptive processing through AFModule layers
    - Optimized for conditionally encoded representations
    """

    def __init__(self, N: int, M: int, *args: Any, **kwargs: Any) -> None:
        """Initialize the conditional DeepJSCC-WZ decoder.

        Args:
            N (int): Number of intermediate channels in the residual blocks.
                     Controls the network capacity and feature dimension.
            M (int): Number of input channels from the encoded representation.
                     Matches the encoder's output channel count.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.g_s = nn.ModuleList(
            [
                AttentionBlock(2 * M),
                ResidualBlock(in_ch=2 * M, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                AttentionBlock(2 * N),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=N, upsample=2),
                AFModule(2 * N, 1),
                ResidualBlock(in_ch=2 * N, out_ch=N),
                ResidualBlockUpsample(in_ch=N, out_ch=3, upsample=2),
                AFModule(3 * 2, 1),
                ResidualBlock(in_ch=3 * 2, out_ch=3),
            ]
        )

        # Add the g_a2 module list for processing side information
        self.g_a2 = nn.ModuleList(
            [
                ResidualBlockWithStride(in_ch=3, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                AttentionBlock(N),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=N, stride=2),
                AFModule(N, 1),
                ResidualBlock(in_ch=N, out_ch=N),
                ResidualBlockWithStride(in_ch=N, out_ch=M, stride=2),
                AFModule(M, 1),
                AttentionBlock(M),
            ]
        )

    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Decode the received representation into a reconstructed image.

        This method first processes the side information through the g_a2 encoder path,
        then progressively decodes the received signal while fusing with side information
        features at multiple scales.

        Args:
            x (torch.Tensor): Received noisy encoded representation of shape [B, M, H/16, W/16].
            x_side (torch.Tensor): Side information tensor of shape [B, 3, H, W] to assist in decoding.
            csi (torch.Tensor): Channel state information tensor of shape [B, 1, 1, 1].
            *args: Additional positional arguments (passed to internal layers).
            **kwargs: Additional keyword arguments (passed to internal layers).

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, 3, H, W].
        """
        csi_sideinfo = csi

        # Process side information through g_a2 to extract features
        xs_list = []
        xs = x_side  # Use side information as initial input

        # First collect features from side information at different resolutions
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(xs)  # Save feature before downsampling

            if isinstance(layer, AFModule):
                # Pass xs and csi_sideinfo as separate arguments
                xs = layer(xs, csi_sideinfo, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                xs = layer(xs, *args, **kwargs)  # Apply the layer

        # Add the final feature map
        xs_list.append(xs)

        # Decode process - fuse features from side_info at multiple scales
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                # Fusion points: concatenate with side info features
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)

            if isinstance(layer, AFModule):
                # Pass x and csi as separate arguments
                x = layer(x, csi, *args, **kwargs)
            else:
                # Pass *args, **kwargs to other layers
                x = layer(x, *args, **kwargs)

        return x


@ModelRegistry.register_model()
class Yilmaz2024DeepJSCCWZModel(WynerZivModel):
    """A specialized Wyner-Ziv model for neural joint source-channel coding with side information.
    :cite:`yilmaz2024deepjsccwz,wyner1976rate`.

    This model implements the DeepJSCC-WZ architecture from Yilmaz et al. 2024, which applies
    deep learning techniques to the Wyner-Ziv coding paradigm (lossy compression with decoder-side
    information). The system is designed specifically for wireless image transmission scenarios
    where correlated side information is available at the receiver.

    Unlike traditional separate source and channel coding approaches, DeepJSCC-WZ:
    1. Jointly optimizes source compression and channel coding in an end-to-end manner
    2. Adapts to varying channel conditions through explicit CSI conditioning
    3. Exploits correlations between the transmitted signal and side information at the decoder
    4. Provides graceful degradation under challenging channel conditions

    Three model variants are supported:
    - Standard: Separate encoder/decoder with independent parameters (highest parameter count)
    - Small: Parameter-efficient design with shared encoder components
    - Conditional: Side information available at both encoder and decoder (performance upper bound)

    The model automatically detects which variant is being used based on the encoder class.

    Technical details:
    - Compression ratio: determined by channel dimension M and spatial downsampling (16× by default)
    - Channel adaptation: AFModule layers condition the network on current channel SNR
    - Side information fusion: Multi-scale fusion at multiple network layers at the decoder
    - Power normalization: Required constraint to ensure proper signal power scaling

    Attributes:
        encoder (BaseModel): Encoder network (standard, small, or conditional variant)
        channel (BaseChannel): Channel simulation model (e.g., AWGN, Rayleigh fading)
        decoder (BaseModel): Decoder network that utilizes side information
        constraint (BaseConstraint): Signal power normalization constraint
        is_conditional (bool): Auto-detected flag indicating if using the conditional variant
    """

    def __init__(
        self,
        encoder: BaseModel,
        channel: BaseChannel,
        decoder: BaseModel,
        constraint: BaseConstraint,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Yilmaz2024DeepJSCCWZ model.

        Args:
            encoder: Neural encoder model that compresses the source image.
                     Must be one of the DeepJSCC-WZ encoder variants (standard, small, or conditional).
            channel: Channel simulation model that applies noise and/or fading effects to the
                     encoded representation during transmission.
            decoder: Neural decoder model that reconstructs the image using received data and side
                     information. Must match the encoder variant.
            constraint: Power normalization constraint that ensures transmitted signals maintain
                       appropriate power levels. This is crucial for fair comparisons across
                       different models and transmission scenarios.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Ensure constraint is not None as it's required
        if constraint is None:
            raise ValueError("A constraint must be provided for Yilmaz2024DeepJSCCWZ model")

        # Initialize the parent class without quantizer and syndrome_generator
        kwargs = kwargs.copy()
        kwargs["encoder"] = encoder
        kwargs["decoder"] = decoder
        kwargs["channel"] = channel
        kwargs["constraint"] = constraint
        kwargs["correlation_model"] = None
        kwargs["quantizer"] = None
        kwargs["syndrome_generator"] = None

        super().__init__(*args, **kwargs)
        # Auto-detect if using conditional model based on encoder class
        self.is_conditional = isinstance(encoder, Yilmaz2024DeepJSCCWZConditionalEncoder)

    def forward(self, source: torch.Tensor, side_info: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Execute the complete Wyner-Ziv coding process on the source image.

        This method implements the full DeepJSCC-WZ model:
        1. Encodes the source image into a compact representation
        - For conditional models: utilizes side information during encoding
        - For non-conditional models: encodes without access to side information
        2. Applies power normalization to the encoded representation
        3. Simulates transmission through a noisy channel
        4. Reconstructs the image using the received data and side information

        All steps are differentiable, allowing for end-to-end training that jointly
        optimizes the entire transmission system for a given distortion metric and
        channel model.

        Args:
            source: Source image tensor to encode and transmit, shape [B, C, H, W].
                   Typically RGB images with values normalized to [0,1].
            side_info: Correlated side information available at the decoder, shape [B, C, H, W].
                      This could be a previous frame in a video, a low-resolution version,
                      or other correlated information that helps in reconstruction.
            *args: Additional positional arguments passed to internal components.
            **kwargs: Additional keyword arguments passed to internal components.
                      Must include 'csi' (torch.Tensor): Channel state information tensor
                      of shape [B, 1, 1, 1]. Contains the signal-to-noise ratio (SNR)
                      or other channel quality indicators that allow the model to adapt
                      to current channel conditions.

        Returns:
            torch.Tensor: The final reconstructed image tensor of shape [B, C, H, W].

        Raises:
            ValueError: If side_info or csi is None, as these are required parameters.

        Note:
            CSI values are typically provided in dB and should be normalized to an appropriate
            range as expected by the model's training configuration.
        """
        # Validate parameters
        if side_info is None:
            raise ValueError("Side information must be provided for Yilmaz2024DeepJSCCWZ model")

        # Extract csi and remove it from kwargs to prevent duplicate passing
        csi = kwargs.pop("csi", None)
        if csi is None:
            raise ValueError("Channel state information (CSI) must be provided in kwargs for Yilmaz2024DeepJSCCWZ model")

        # Source encoding - pass *args and the modified **kwargs (without csi)
        if self.is_conditional:
            encoded = self.encoder(source, side_info, csi, *args, **kwargs)
        else:
            # For non-conditional models, don't pass the side_info parameter
            encoded = self.encoder(source, csi, *args, **kwargs)

        # Apply mandatory power/rate constraint - pass *args and modified **kwargs
        if self.constraint is None:
            raise RuntimeError("Constraint is unexpectedly None. This should not happen if __init__ validation is working.")
        # Assuming constraint does not need csi explicitly, pass remaining kwargs
        constrained = self.constraint(encoded, *args, **kwargs)

        # Transmit through channel - pass *args and modified **kwargs
        # Assuming channel does not need csi explicitly, pass remaining kwargs
        received = self.channel(constrained, *args, **kwargs)

        # Decode using received representation and side information - pass *args and modified **kwargs (without csi)
        decoded = self.decoder(received, side_info, csi, *args, **kwargs)

        return decoded
