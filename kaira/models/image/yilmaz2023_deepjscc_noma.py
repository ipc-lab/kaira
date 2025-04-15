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
        self.M = M
        self.latent_dim = latent_dim
        self.use_perfect_sic = use_perfect_sic
        self.use_device_embedding = use_device_embedding if use_device_embedding is not None else shared_encoder
        self.image_shape = image_shape
        self.csi_length = csi_length
        self.embedding_dim = image_shape[0] * image_shape[1]

        # Determine encoder/decoder classes or instances
        encoder_config = encoder if encoder is not None else DEFAULT_ENCODER
        decoder_config = decoder if decoder is not None else DEFAULT_DECODER

        # Initialize the base class, passing *args and **kwargs for potential instantiation
        super().__init__(
            encoders=encoder_config,  # Pass class or instance directly
            decoders=decoder_config,  # Pass class or instance directly
            channel=channel,
            power_constraint=power_constraint,
            num_devices=num_devices,
            shared_encoder=shared_encoder,  # Pass flag
            shared_decoder=shared_decoder,  # Pass flag
        )

        # Device embedding setup (needs num_devices from base class)
        if self.use_device_embedding:
            self.device_images = nn.Embedding(self.num_devices, embedding_dim=self.embedding_dim)
            # Loading checkpoint needs to happen *after* models are created by super().__init__
            if ckpt_path is not None:
                self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        """Load pre-trained weights from checkpoint.

        Args:
            ckpt_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(ckpt_path)
        # Load into the ModuleLists managed by this class
        enc_dict = checkpoint.get("encoder_state_dict", {})
        dec_dict = checkpoint.get("decoder_state_dict", {})
        img_dict = checkpoint.get("device_image_state_dict", {})

        # Base class __init__ created self.encoders and self.decoders (ModuleLists)
        if self.shared_encoder:
            if enc_dict:  # Only load if dict is not empty
                self.encoders[0].load_state_dict(enc_dict)
        else:
            if enc_dict:  # Check if dict is not empty
                # Ensure keys match ModuleList structure (e.g., "0", "1", ...)
                # If checkpoint saved a single shared encoder, this might need adjustment
                try:
                    self.encoders.load_state_dict(enc_dict)
                except RuntimeError as e:
                    print(f"Warning: Could not load encoder state dict directly: {e}. Attempting to load first encoder only.")
                    if "0" in enc_dict:
                        self.encoders[0].load_state_dict(enc_dict["0"])
                    elif len(self.encoders) > 0:
                        # Fallback: Assume the dict is for the first encoder if keys don't match
                        try:
                            self.encoders[0].load_state_dict(enc_dict)
                            print("Successfully loaded state into the first encoder.")
                        except Exception as inner_e:
                            print(f"Failed to load state into the first encoder: {inner_e}")

        if self.shared_decoder:
            if dec_dict:
                self.decoders[0].load_state_dict(dec_dict)
        else:
            if dec_dict:
                try:
                    self.decoders.load_state_dict(dec_dict)
                except RuntimeError as e:
                    print(f"Warning: Could not load decoder state dict directly: {e}. Attempting to load first decoder only.")
                    if "0" in dec_dict:
                        self.decoders[0].load_state_dict(dec_dict["0"])
                    elif len(self.decoders) > 0:
                        try:
                            self.decoders[0].load_state_dict(dec_dict)
                            print("Successfully loaded state into the first decoder.")
                        except Exception as inner_e:
                            print(f"Failed to load state into the first decoder: {inner_e}")

        if self.use_device_embedding and img_dict:
            self.device_images.load_state_dict(img_dict)
        print("checkpoint loaded")

    def forward(self, x: List[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the DeepJSCC-NOMA model.

        Args:
            x: Input data, a list of tensors [batch_size, channels, height, width] for each device.
            *args: Additional positional arguments passed to internal components.
            **kwargs: Additional keyword arguments passed to internal components (e.g., csi).

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        # Retrieve csi from kwargs if it exists, otherwise None
        kwargs.get("csi")

        if not isinstance(x, list) or len(x) != self.num_devices:
            raise ValueError(f"Input 'x' must be a list of {self.num_devices} tensors, but got {type(x)} with length {len(x) if isinstance(x, list) else 'N/A'}.")

        processed_x = []
        # Add device embeddings if enabled
        if self.use_device_embedding:
            h, w = self.image_shape
            if not x:  # Handle empty input list
                raise ValueError("Input list 'x' cannot be empty when use_device_embedding is True.")
            batch_size = x[0].size(0)  # Get batch size from the first tensor
            device = x[0].device  # Get device from the first tensor
            for i in range(self.num_devices):
                if i >= len(x):
                    raise ValueError(f"Input list 'x' has length {len(x)}, but expected at least {i+1} tensors for num_devices={self.num_devices}.")
                emb_i = self.device_images(torch.ones((batch_size), dtype=torch.long, device=device) * i).view(batch_size, 1, h, w)
                # Ensure x[i] has 4 dimensions [B, C, H, W] before concatenating
                if x[i].ndim != 4:
                    raise ValueError(f"Input tensor for device {i} has unexpected dimensions: {x[i].shape}")
                processed_x.append(torch.cat([x[i], emb_i], dim=1))  # Concatenate along channel dim
        else:
            processed_x = x  # Use original list if no embedding

        if self.use_perfect_sic:
            # Pass the processed list and original args/kwargs
            return self._forward_perfect_sic(processed_x, *args, **kwargs)

        # Encode inputs - support different encoder interfaces
        transmissions: List[torch.Tensor] = []
        for i in range(self.num_devices):
            # Use shared_encoder flag to get the correct encoder
            encoder = self.encoders[0] if self.shared_encoder else self.encoders[i]
            device_input = processed_x[i]  # Use the i-th tensor from the processed list

            # Pass original args and kwargs (which includes csi if needed by encoder)
            tx = encoder(device_input, *args, **kwargs)

            # Pass *args, **kwargs to power_constraint (includes csi if needed)
            tx = self.power_constraint(tx, *args, **kwargs)
            transmissions.append(tx)

        # Stack and SUM transmissions across devices to simulate NOMA superposition
        x_stacked = torch.stack(transmissions, dim=1)  # Shape: [B, N_dev, C_latent, H, W]
        x_summed = torch.sum(x_stacked, dim=1)  # Shape: [B, C_latent, H, W]

        # Apply channel - Pass *args, **kwargs
        x_channel_out = self.channel(x_summed, *args, **kwargs)  # Shape: [B, C_latent, H, W]

        decoded_outputs: List[torch.Tensor] = []
        for i in range(self.num_devices):
            decoder = self.decoders[i]
            # Pass *args, **kwargs to decoder
            x_decoded = decoder(x_channel_out, *args, **kwargs)  # Input is 4D
            # Assuming each non-shared decoder outputs [B, C_out_per_device, H, W]
            decoded_outputs.append(x_decoded)

        x = torch.stack(decoded_outputs, dim=1)  # Stack results -> [B, N_dev, C_out, H, W]

        return x

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

            t = encoder(device_input, csi=csi, *args, **kwargs)

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

            if self.shared_decoder and xi.ndim == 5:  # [B, num_devices, C, H, W]
                # If shared decoder outputs all devices, select the relevant one
                xi = xi[:, i, ...]
            # Assuming xi is now [B, C_out, H, W] or needs reshaping if shared decoder outputs differently

            results.append(xi)

        return torch.stack(results, dim=1)  # Stack results -> [B, N_dev, C_out, H, W]
