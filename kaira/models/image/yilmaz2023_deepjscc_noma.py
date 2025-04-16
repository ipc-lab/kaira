"""DeepJSCC-NOMA module for Kaira.

This module contains the Yilmaz2023DeepJSCCNOMA model, which implements Distributed Deep Joint
Source-Channel Coding over a Multiple Access Channel as described in the paper by Yilmaz et al.
(2023).
"""

from typing import Any, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from kaira.channels.base import BaseChannel
from kaira.constraints.base import BaseConstraint
from kaira.models.base import BaseModel
from kaira.models.image.tung2022_deepjscc_q import (
    Tung2022DeepJSCCQ2Decoder,
    Tung2022DeepJSCCQ2Encoder,
)
from kaira.models.multiple_access_channel import MultipleAccessChannelModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model()
class Yilmaz2023DeepJSCCNOMAEncoder(Tung2022DeepJSCCQ2Encoder):
    """DeepJSCC-NOMA Encoder Module :cite:`yilmaz2023distributed`.

    This encoder transforms input images into latent representations. This class extends the
    Tung2022DeepJSCCQ2Encoder class with parameter adaptation as used in the paper :cite:t:`yilmaz2023distributed`.
    """

    def __init__(self, N=64, M=16, in_ch=4, csi_length=1, *args: Any, **kwargs: Any):
        """Initialize the DeepJSCCNOMAEncoder.

        Args:
            N (int, optional): Number of channels in the network.
            M (int, optional): Latent dimension of the bottleneck representation.
            in_ch (int, optional): Number of input channels. Defaults to 4.
            csi_length (int, optional): The number of dimensions in the CSI data. Defaults to 1.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(N=N, M=M, in_ch=in_ch, csi_length=csi_length)

    # Forward method is inherited from Tung2022DeepJSCCQ2Encoder, which already handles *args, **kwargs


@ModelRegistry.register_model()
class Yilmaz2023DeepJSCCNOMADecoder(Tung2022DeepJSCCQ2Decoder):
    """DeepJSCC-NOMA Decoder Module :cite:`yilmaz2023distributed`.

    This decoder reconstructs images from received channel signals, supporting both
    individual device decoding and shared decoding for multiple devices. This class extends
    the Tung2022DeepJSCCQ2Decoder class with parameter adaptation as used in the paper :cite:t:`yilmaz2023distributed`.
    """

    def __init__(self, N=64, M=16, out_ch_per_device=3, csi_length=1, num_devices=1, shared_decoder=False, *args: Any, **kwargs: Any):
        """Initialize the DeepJSCCNOMADecoder.

        Args:
            N (int, optional): Number of channels in the network. Defaults to 64 if not provided.
            M (int, optional): Latent dimension of the bottleneck representation. Defaults to 16 if not provided.
            out_ch_per_device (int, optional): Number of output channels per device. Defaults to 3.
            csi_length (int, optional): The number of dimensions in the CSI data. Defaults to 1.
            num_devices (int, optional): Number of devices. Used for shared decoder. Defaults to 1.
            shared_decoder (bool, optional): Whether this is a shared decoder. Defaults to False.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Store additional parameters
        self.num_devices = num_devices
        self.shared_decoder = shared_decoder

        super().__init__(N=N, M=M, out_ch=self.num_devices * out_ch_per_device, csi_length=csi_length)

    # Forward method is inherited from Tung2022DeepJSCCQ2Decoder, which already handles *args, **kwargs


# Use Tung2022DeepJSCCQ2 models as default
DEFAULT_ENCODER = Yilmaz2023DeepJSCCNOMAEncoder
DEFAULT_DECODER = Yilmaz2023DeepJSCCNOMADecoder


@ModelRegistry.register_model("deepjscc_noma")
class Yilmaz2023DeepJSCCNOMAModel(MultipleAccessChannelModel):
    """Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel
    :cite:`yilmaz2023distributed`.

    This model implements the DeepJSCC-NOMA system from the paper by Yilmaz et al. (2023),
    which enables multiple devices to transmit jointly encoded data over a shared
    wireless channel using Non-Orthogonal Multiple Access (NOMA).

    Attributes:
        M: Channel bandwidth expansion/compression factor
        latent_dim: Dimension of latent representation
        use_perfect_sic: Whether to use perfect successive interference cancellation
        use_device_embedding: Whether to use device embeddings
        image_shape: Shape of the input images used for embedding
        device_images: Embedding table for device-specific embeddings
    """

    def __init__(
        self,
        channel: BaseChannel,
        power_constraint: BaseConstraint,
        encoder: Optional[Union[Type[BaseModel], BaseModel]] = None,  # Allow class or instance
        decoder: Optional[Union[Type[BaseModel], BaseModel]] = None,  # Allow class or instance
        num_devices: int = 2,
        M: float = 1.0,
        latent_dim: int = 16,
        shared_encoder: bool = False,
        shared_decoder: bool = False,
        use_perfect_sic: bool = False,
        use_device_embedding: Optional[bool] = None,
        image_shape: Tuple[int, int] = (32, 32),
        csi_length: int = 1,
        ckpt_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the DeepJSCC-NOMA model.

        Args:
            channel: Channel model for transmission
            power_constraint: Power constraint to apply to transmitted signals
            encoder: Encoder network class or constructor (default: Tung2022DeepJSCCQ2Encoder)
            decoder: Decoder network class or constructor (default: Tung2022DeepJSCCQ2Decoder)
            num_devices: Number of transmitting devices
            M: Channel bandwidth expansion/compression factor
            latent_dim: Dimension of latent representation
            shared_encoder: Whether to use a shared encoder across devices
            shared_decoder: Whether to use a shared decoder across devices
            use_perfect_sic: Whether to use perfect successive interference cancellation
            use_device_embedding: Whether to use device embeddings
            image_shape: Shape of input images (height, width) for determining embedding dimensions
            csi_length: The length of CSI (Channel State Information) vector
            ckpt_path: Path to checkpoint file for loading pre-trained weights
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        # Initialize DeepJSCC-NOMA specific attributes
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.M = M
        self.latent_dim = latent_dim
        self.use_perfect_sic = use_perfect_sic
        self.use_device_embedding = use_device_embedding if use_device_embedding is not None else shared_encoder
        self.image_shape = image_shape
        self.csi_length = csi_length
        self.embedding_dim = image_shape[0] * image_shape[1]
        # Ensure num_devices is set before using it below
        self.num_devices = num_devices

        # Determine encoder/decoder classes or instances
        encoder_config = encoder if encoder is not None else DEFAULT_ENCODER
        decoder_config = decoder if decoder is not None else DEFAULT_DECODER

        # Prepare decoder config based on shared_decoder flag BEFORE calling super().__init__
        final_decoder_config: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList]
        if not self.shared_decoder and isinstance(decoder_config, type):
            # Need separate instances, create them now using appropriate args
            # Assuming default N=64, out_ch_per_device=3 based on context/defaults
            decoder_args = {
                "N": kwargs.get("N", 64),  # Allow override via kwargs if provided
                "M": self.latent_dim,
                "out_ch_per_device": kwargs.get("out_ch_per_device", 3),  # Allow override, assume 3 for RGB
                "csi_length": self.csi_length,
                # Pass specific args for Yilmaz decoder
                "num_devices": 1,  # Each instance conceptually handles one device stream
                "shared_decoder": False,  # Explicitly false for each instance
            }
            # Instantiate num_devices separate decoders
            final_decoder_config = [decoder_config(**decoder_args) for _ in range(self.num_devices)]
        else:
            # Shared decoder, or already an instance/list: pass as is to superclass
            final_decoder_config = decoder_config

        # Initialize the base class, passing the potentially modified decoder config
        super().__init__(encoders=encoder_config, decoders=final_decoder_config, channel=channel, power_constraint=power_constraint, num_devices=num_devices)  # Use the processed config

        # Device embedding setup (needs num_devices from base class)
        if self.use_device_embedding:
            self.device_images = nn.Embedding(self.num_devices, embedding_dim=self.embedding_dim)
        # Loading checkpoint needs to happen *after* models are created by super().__init__
        if ckpt_path is not None:
            self._load_checkpoint(ckpt_path)

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the DeepJSCC-NOMA model.

        Args:
            x: Input data.
               - If use_perfect_sic=True: A single tensor [batch_size, num_devices, channels, height, width].
               - If use_perfect_sic=False: A list of tensors [batch_size, channels, height, width] for each device.
            *args: Additional positional arguments passed to internal components.
            **kwargs: Additional keyword arguments passed to internal components (e.g., csi).

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        csi = kwargs.get("csi")

        # --- Perfect SIC Mode ---
        if self.use_perfect_sic:
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Input 'x' must be a single tensor when use_perfect_sic=True, but got {type(x)}.")
            # Check tensor shape for perfect SIC mode
            if x.ndim != 5 or x.shape[1] != self.num_devices:
                raise ValueError(f"Input tensor shape mismatch for perfect SIC. Expected [B, {self.num_devices}, C, H, W], got {x.shape}")
            if csi is None:
                raise ValueError("Keyword argument 'csi' must be provided when use_perfect_sic=True.")
            # Remove csi from kwargs before passing to avoid duplicate argument error
            kwargs_without_csi = kwargs.copy()
            kwargs_without_csi.pop("csi", None)  # Remove csi if it exists
            # Call the perfect SIC forward method
            return self._forward_perfect_sic(x, csi, *args, **kwargs_without_csi)

        # --- Standard NOMA Mode (Non-Perfect SIC) ---
        if not isinstance(x, list) or len(x) != self.num_devices:
            raise ValueError(f"Input 'x' must be a list of {self.num_devices} tensors when use_perfect_sic=False, but got {type(x)} with length {len(x) if isinstance(x, list) else 'N/A'}.")

        processed_x = []
        # Add device embeddings if enabled
        if self.use_device_embedding:
            h, w = self.image_shape
            if not x:
                raise ValueError("Input list 'x' cannot be empty when use_device_embedding is True.")
            batch_size = x[0].size(0)
            device = x[0].device
            for i in range(self.num_devices):
                if i >= len(x):
                    raise ValueError(f"Input list 'x' has length {len(x)}, but expected at least {i+1} tensors for num_devices={self.num_devices}.")
                emb_i = self.device_images(torch.ones((batch_size), dtype=torch.long, device=device) * i).view(batch_size, 1, h, w)
                if x[i].ndim != 4:
                    raise ValueError(f"Input tensor for device {i} has unexpected dimensions: {x[i].shape}")
                processed_x.append(torch.cat([x[i], emb_i], dim=1))
        else:
            processed_x = x

        # Encode inputs
        transmissions: List[torch.Tensor] = []
        for i in range(self.num_devices):
            encoder = self.encoders[0] if self.shared_encoder else self.encoders[i]
            device_input = processed_x[i]
            tx = encoder(device_input, *args, **kwargs)
            tx = self.power_constraint(tx, *args, **kwargs)
            transmissions.append(tx)

        # Combine signals for NOMA
        x_stacked = torch.stack(transmissions, dim=1)
        x_summed = torch.sum(x_stacked, dim=1)

        # Apply channel
        x_channel_out = self.channel(x_summed, *args, **kwargs)

        decoded_outputs: List[torch.Tensor] = []
        # In standard NOMA, the decoder receives the summed signal from the channel
        decoder_input = x_channel_out

        if self.shared_decoder:
            decoder = self.decoders[0]
            # Shared decoder processes the summed signal
            x_decoded_all = decoder(decoder_input, *args, **kwargs)  # Expected: [B, D*C_out, H, W]

            try:
                # Assumes decoder has 'out_ch' attribute storing total output channels (D*C_out)
                num_out_channels_total = decoder.out_ch
            except AttributeError:
                # Fallback for decoders without 'out_ch'
                if x_decoded_all.shape[1] % self.num_devices != 0:
                    raise ValueError(f"Cannot determine output channels per device for shared decoder. Output shape: {x_decoded_all.shape}")
                num_out_channels_total = x_decoded_all.shape[1]

            if num_out_channels_total % self.num_devices != 0:
                raise ValueError(f"Shared decoder total output channels ({num_out_channels_total}) not divisible by num_devices ({self.num_devices}).")
            num_out_channels_per_device = num_out_channels_total // self.num_devices

            for i in range(self.num_devices):
                start_channel = i * num_out_channels_per_device
                end_channel = (i + 1) * num_out_channels_per_device
                decoded_outputs.append(x_decoded_all[:, start_channel:end_channel, :, :])
        else:
            # Non-shared decoders: each processes the same summed input
            for i in range(self.num_devices):
                decoder = self.decoders[i]
                x_decoded = decoder(decoder_input, *args, **kwargs)  # Expected: [B, C_out, H, W]
                decoded_outputs.append(x_decoded)

        # Stack results
        x_out = torch.stack(decoded_outputs, dim=1)  # Shape: [B, N_dev, C_out, H, W]

        return x_out

    def _forward_perfect_sic(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with perfect successive interference cancellation.

        Args:
            x: Input data with shape [batch_size, num_devices, channels, height, width]
            csi: Channel state information with shape [batch_size, csi_length]
            *args: Additional positional arguments passed to internal components.
            **kwargs: Additional keyword arguments passed to internal components.

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        # Add device embeddings if enabled
        if self.use_device_embedding:
            h, w = self.image_shape
            emb = torch.stack([self.device_images(torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i).view(x.size(0), 1, h, w) for i in range(self.num_devices)], dim=1)
            x = torch.cat([x, emb], dim=2)

        transmissions: List[torch.Tensor] = []

        # Apply encoders and channel with SIC - support different encoder interfaces
        for i in range(self.num_devices):
            # Use shared_encoder flag
            encoder = self.encoders[0] if self.shared_encoder else self.encoders[i]
            device_input = x[:, i, ...]

            # Pass csi explicitly as a keyword argument to the encoder
            t = encoder(device_input, *args, csi=csi, **kwargs)

            # Apply power constraint - Assuming output t is 4D [B, C, H, W]
            # The original power constraint logic seemed specific and might need review
            # For simplicity, let's assume a standard power constraint applied per device signal
            t = self.power_constraint(t, *args, **kwargs)  # Ensure output is 4D

            # Use the provided channel model for each transmission - Pass only the 4D signal tensor
            t_channel = self.channel(t, *args, **kwargs)  # Input is 4D, output is 4D

            transmissions.append(t_channel)  # List of 4D tensors

        # Decode each transmission - support different decoder interfaces
        results: List[torch.Tensor] = []
        for i in range(self.num_devices):
            # Use shared_decoder flag
            decoder = self.decoders[0] if self.shared_decoder else self.decoders[i]

            # Pass only the relevant 4D tensor transmission to the decoder, including csi=csi
            xi = decoder(transmissions[i], csi=csi, *args, **kwargs)  # Input is 4D

            if self.shared_decoder and xi.ndim == 4 and xi.shape[1] > decoder.out_ch_per_device:  # Example check
                # If shared decoder outputs all devices, select the relevant one
                num_out_channels_total = xi.shape[1]
                num_out_channels_per_device = num_out_channels_total // self.num_devices
                start_channel = i * num_out_channels_per_device
                end_channel = (i + 1) * num_out_channels_per_device
                xi = xi[:, start_channel:end_channel, :, :]

            results.append(xi)  # List of [B, C_out, H, W]

        return torch.stack(results, dim=1)  # Stack to [B, D, C_out, H, W]
