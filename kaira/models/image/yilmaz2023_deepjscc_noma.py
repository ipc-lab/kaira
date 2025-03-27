"""DeepJSCC-NOMA module for Kaira.

This module contains the Yilmaz2023DeepJSCCNOMA model, which implements Distributed Deep Joint
Source-Channel Coding over a Multiple Access Channel as described in the paper by Yilmaz et al.
(2023).
"""

from typing import Any, List, Optional, Tuple, Type

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

# Use Tung2022DeepJSCCQ2 models as default
DEFAULT_ENCODER = Tung2022DeepJSCCQ2Encoder
DEFAULT_DECODER = Tung2022DeepJSCCQ2Decoder


@ModelRegistry.register_model()
class Yilmaz2023DeepJSCCNOMAEncoder(Tung2022DeepJSCCQ2Encoder):
    """DeepJSCC-NOMA Encoder Module :cite:`yilmaz2023distributed`.

    This encoder transforms input images into latent representations. This class directly extends the Tung2022DeepJSCCQ2Encoder class without any modifications as in the paper :cite:t:`yilmaz2023distributed`.
    """

    pass


@ModelRegistry.register_model()
class Yilmaz2023DeepJSCCNOMADecoder(Tung2022DeepJSCCQ2Decoder):
    """DeepJSCC-NOMA Decoder Module :cite:`yilmaz2023distributed`.

    This decoder reconstructs images from received channel signals, supporting both
    individual device decoding and shared decoding for multiple devices. This class directly extends the Tung2022DeepJSCCQ2Decoder class without any modifications as in the paper :cite:t:`yilmaz2023distributed`.
    """

    pass


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
        encoder: Optional[Type[BaseModel]] = None,
        decoder: Optional[Type[BaseModel]] = None,
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
        """
        # Initialize the base class
        super().__init__(channel=channel, power_constraint=power_constraint, num_devices=num_devices, shared_encoder=shared_encoder, shared_decoder=shared_decoder)

        # Initialize DeepJSCC-NOMA specific attributes
        self.M = M
        self.latent_dim = latent_dim
        self.use_perfect_sic = use_perfect_sic
        self.use_device_embedding = use_device_embedding if use_device_embedding is not None else shared_encoder
        self.image_shape = image_shape
        self.csi_length = csi_length

        # Calculate embedding dimension based on image shape
        self.embedding_dim = image_shape[0] * image_shape[1]

        # Use provided encoder/decoder or fall back to defaults
        encoder_cls = encoder if encoder is not None else DEFAULT_ENCODER
        decoder_cls = decoder if decoder is not None else DEFAULT_DECODER

        encoder_count = 1 if shared_encoder else num_devices
        decoder_count = 1 if shared_decoder else num_devices
        encoder_channels = 4 if self.use_device_embedding else 3

        # Initialize encoders
        for _ in range(encoder_count):
            # Create new encoder instance based on whether encoder_cls is a class or object instance
            if isinstance(encoder_cls, type):  # It's a class
                try:
                    # Try instantiating with Tung2022DeepJSCCQ2Encoder expected parameters
                    enc = encoder_cls(N=latent_dim, M=latent_dim, csi_length=csi_length)
                except (TypeError, ValueError):
                    try:
                        # Try with C parameter (common in image encoders)
                        enc = encoder_cls(C=encoder_channels, latent_dim=latent_dim)
                    except (TypeError, ValueError):
                        try:
                            # Try with in_channels parameter (common alternative)
                            enc = encoder_cls(in_channels=encoder_channels, latent_dim=latent_dim)
                        except (TypeError, ValueError):
                            try:
                                # Try just with latent_dim
                                enc = encoder_cls(latent_dim=latent_dim)
                            except (TypeError, ValueError):
                                # Last resort: try with no parameters
                                enc = encoder_cls()
            else:  # It's already an instance
                enc = encoder_cls

            self.encoders.append(enc)

        # Initialize decoders
        for _ in range(decoder_count):
            # Create new decoder instance based on whether decoder_cls is a class or object instance
            if isinstance(decoder_cls, type):  # It's a class
                try:
                    # Try instantiating with Tung2022DeepJSCCQ2Decoder expected parameters
                    dec = decoder_cls(N=latent_dim, M=latent_dim, csi_length=csi_length)
                except (TypeError, ValueError):
                    try:
                        # Try with standard parameter set
                        dec = decoder_cls(latent_dim=latent_dim, num_devices=num_devices, shared_decoder=shared_decoder)
                    except (TypeError, ValueError):
                        try:
                            # Try with just latent_dim
                            dec = decoder_cls(latent_dim=latent_dim)
                        except (TypeError, ValueError):
                            # Last resort: try with no parameters
                            dec = decoder_cls()
            else:  # It's already an instance
                dec = decoder_cls

            self.decoders.append(dec)

        if self.use_device_embedding:
            self.device_images = nn.Embedding(num_devices, embedding_dim=self.embedding_dim)
            if ckpt_path is not None:
                self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        """Load pre-trained weights from checkpoint.

        Args:
            ckpt_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(ckpt_path)
        enc_dict = checkpoint.get("encoder_state_dict", {})
        dec_dict = checkpoint.get("decoder_state_dict", {})
        img_dict = checkpoint.get("device_image_state_dict", {})

        self.encoders.load_state_dict(enc_dict)
        self.decoders.load_state_dict(dec_dict)
        self.device_images.load_state_dict(img_dict)
        print("checkpoint loaded")

    def forward(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the DeepJSCC-NOMA model.

        Args:
            x: Input data with shape [batch_size, num_devices, channels, height, width]
            csi: Channel state information with shape [batch_size, csi_length]
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        # Add device embeddings if enabled
        if self.use_device_embedding:
            h, w = self.image_shape
            emb = torch.stack([self.device_images(torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i).view(x.size(0), 1, h, w) for i in range(self.num_devices)], dim=1)
            x = torch.cat([x, emb], dim=2)

        if self.use_perfect_sic:
            return self._forward_perfect_sic(x, csi, *args, **kwargs)

        # Encode inputs - support different encoder interfaces
        transmissions: List[torch.Tensor] = []
        for i in range(self.num_devices):
            encoder = self.encoders[0 if self.shared_encoder else i]
            device_input = x[:, i, ...]

            # Handle encoders with different input formats
            try:
                # Try tuple input with CSI
                tx = encoder((device_input, csi), *args, **kwargs)
            except (TypeError, ValueError):
                # Fall back to just the input data
                tx = encoder(device_input, *args, **kwargs)

            tx = self.power_constraint(tx)
            transmissions.append(tx)

        x = torch.stack(transmissions, dim=1)

        # Apply channel
        x = self.channel((x, csi))

        # Decode outputs - support different decoder interfaces
        if self.shared_decoder:
            decoder = self.decoders[0]
            try:
                # Try tuple input with CSI
                x_decoded = decoder((x, csi), *args, **kwargs)
            except (TypeError, ValueError):
                # Fall back to just the input data
                x_decoded = decoder(x, *args, **kwargs)

            # Make sure output has proper device dimension
            if x_decoded.ndim == 4:  # [B, C, H, W]
                x = x_decoded.unsqueeze(1).expand(-1, self.num_devices, -1, -1, -1)
            else:  # [B, num_devices, C, H, W]
                x = x_decoded
        else:
            # Process each device separately
            decoded_outputs: List[torch.Tensor] = []
            for i in range(self.num_devices):
                decoder = self.decoders[i]
                try:
                    # Try tuple input with CSI
                    x_decoded = decoder((x, csi), *args, **kwargs)
                except (TypeError, ValueError):
                    # Fall back to just the input data
                    x_decoded = decoder(x, *args, **kwargs)

                decoded_outputs.append(x_decoded)

            x = torch.stack(decoded_outputs, dim=1)

        return x

    def _forward_perfect_sic(self, x: torch.Tensor, csi: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass with perfect successive interference cancellation.

        Args:
            x: Input data with shape [batch_size, num_devices, channels, height, width]
            csi: Channel state information with shape [batch_size, csi_length]
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        transmissions: List[torch.Tensor] = []

        # Apply encoders and channel with SIC - support different encoder interfaces
        for i in range(self.num_devices):
            encoder = self.encoders[0 if self.shared_encoder else i]
            device_input = x[:, i, ...]

            # Handle encoders with different input formats
            try:
                # Try tuple input with CSI
                t = encoder((device_input, csi), *args, **kwargs)
            except (TypeError, ValueError):
                # Fall back to just the input data
                t = encoder(device_input, *args, **kwargs)

            t = self.power_constraint(t[:, None, ...], mult=torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))).sum(dim=1)

            # Use the provided channel model for each transmission
            t_channel = self.channel((t, csi))

            transmissions.append(t_channel)

        # Decode each transmission - support different decoder interfaces
        results: List[torch.Tensor] = []
        for i in range(self.num_devices):
            decoder = self.decoders[0 if self.shared_decoder else i]

            try:
                # Try tuple input with CSI
                xi = decoder((transmissions[i], csi), *args, **kwargs)
            except (TypeError, ValueError):
                # Fall back to just the input data
                xi = decoder(transmissions[i], *args, **kwargs)

            if self.shared_decoder and xi.ndim == 5:  # [B, num_devices, C, H, W]
                xi = xi[:, i, ...]

            results.append(xi)

        return torch.stack(results, dim=1)
