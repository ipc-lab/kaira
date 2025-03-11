"""DeepJSCC-NOMA module for Kaira.

This module contains the Yilmaz2023DeepJSCCNOMA model, which implements 
Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel 
as described in the paper by Yilmaz et al. (2023).
"""

import torch
from torch import nn

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint
from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model("deepjscc_noma")
class Yilmaz2023DeepJSCCNOMA(BaseModel):
    """Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel :cite:`yilmaz2023distributed`.

    This model implements the DeepJSCC-NOMA system from the paper by :cite:`yilmaz2023distributed`,
    which enables multiple devices to transmit jointly encoded data over a shared
    wireless channel using Non-Orthogonal Multiple Access (NOMA).

    Attributes:
        channel: The channel model for transmission
        power_constraint: Power constraint applied to transmitted signals
        encoders: List of encoder networks for each device
        decoders: List of decoder networks for each device
        M: Channel bandwidth expansion/compression factor
        num_devices: Number of transmitting devices in the system
        shared_encoder: Whether to use the same encoder for all devices
        shared_decoder: Whether to use the same decoder for all devices
        use_perfect_sic: Whether to use perfect successive interference cancellation
        use_device_embedding: Whether to use device embeddings
    """

    def __init__(
        self, 
        channel, 
        power_constraint, 
        encoder, 
        decoder, 
        num_devices, 
        M,
        shared_encoder=False,
        shared_decoder=False,
        use_perfect_sic=False,
        use_device_embedding=None,
        ckpt_path=None
    ):
        """Initialize the DeepJSCC-NOMA model.

        Args:
            channel: Channel model for transmission
            power_constraint: Power constraint to apply to transmitted signals
            encoder: Encoder network class or constructor
            decoder: Decoder network class or constructor
            num_devices: Number of transmitting devices
            M: Channel bandwidth expansion/compression factor
            shared_encoder: Whether to use a shared encoder across devices
            shared_decoder: Whether to use a shared decoder across devices
            use_perfect_sic: Whether to use perfect successive interference cancellation
            use_device_embedding: Whether to use device embeddings
            ckpt_path: Path to checkpoint file for loading pre-trained weights
        """
        super().__init__()

        self.channel = channel
        self.power_constraint = power_constraint
        self.M = M
        self.num_devices = num_devices
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.use_perfect_sic = use_perfect_sic
        self.use_device_embedding = use_device_embedding if use_device_embedding is not None else shared_encoder

        encoder_count = 1 if shared_encoder else num_devices
        decoder_count = 1 if shared_decoder else num_devices
        encoder_channels = 4 if self.use_device_embedding else 3
        decoder_channels = 3 * num_devices if shared_decoder else None
        
        self.encoders = nn.ModuleList([encoder(C=encoder_channels) for _ in range(encoder_count)])
        self.decoders = nn.ModuleList([decoder(C=decoder_channels) for _ in range(decoder_count)])
        
        if self.use_device_embedding:
            self.device_images = nn.Embedding(num_devices, embedding_dim=32 * 32)
            if ckpt_path is not None:
                self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, ckpt_path):
        """Load model weights from a checkpoint file.
        
        Args:
            ckpt_path: Path to the checkpoint file
        """
        state_dict = torch.load(ckpt_path)["state_dict"]
        
        enc_dict, dec_dict, img_dict = {}, {}, {}
        for k, v in state_dict.items():
            if k.startswith("net.encoders.0"):
                enc_dict[k.replace("net.encoders.0", "0")] = v
            elif k.startswith("net.decoders.0"):
                dec_dict[k.replace("net.decoders.0", "0")] = v
            elif k.startswith("net.device_images."):
                img_dict[k.replace("net.device_images.", "")] = v

        self.encoders.load_state_dict(enc_dict)
        self.decoders.load_state_dict(dec_dict)
        self.device_images.load_state_dict(img_dict)
        print("checkpoint loaded")

    def forward(self, x, csi):
        """Forward pass of the DeepJSCC-NOMA model.

        Args:
            x: Input images with shape [batch_size, num_devices, channels, height, width]
            csi: Channel state information values for the channel

        Returns:
            Reconstructed signals with shape [batch_size, num_devices, channels, height, width]
        """
        if self.use_device_embedding:
            emb = torch.stack(
                [self.device_images(torch.ones((x.size(0)), dtype=torch.long, device=x.device) * i).view(x.size(0), 1, 32, 32)
                for i in range(self.num_devices)], dim=1
            )
            x = torch.cat([x, emb], dim=2)
        
        if self.use_perfect_sic:
            return self._forward_perfect_sic(x, csi)
        
        # Encode inputs
        transmissions = [self.encoders[0 if self.shared_encoder else i]((x[:, i, ...], csi)) 
                         for i in range(self.num_devices)]
        x = torch.stack(transmissions, dim=1)
        
        # Apply channel
        x = self.power_constraint(x)
        x = self.channel((x, csi))

        # Decode outputs
        if self.shared_decoder:
            x = self.decoders[0]((x, csi))
            x = x.view(x.size(0), self.num_devices, 3, x.size(2), x.size(3))
        else:
            x = torch.stack([self.decoders[i]((x, csi)) for i in range(self.num_devices)], dim=1)

        return x
    
    def _forward_perfect_sic(self, x, csi):
        """Forward pass with perfect successive interference cancellation.
        
        Args:
            x: Input data
            csi: Channel state information
            
        Returns:
            Reconstructed signals
        """
        transmissions = []
        
        # Apply encoders and channel with SIC
        for i in range(self.num_devices):
            t = self.encoders[0 if self.shared_encoder else i]((x[:, i, ...], csi))
            t = self.power_constraint(
                t[:, None, ...], 
                mult=torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))
            ).sum(dim=1)
            
            # Use the provided channel model for each transmission
            # This ensures consistency with the channel model throughout the code
            # t_channel = t.unsqueeze(1)  # Add dimension for compatibility with channel
            t_channel = self.channel((t, csi)) #.squeeze(1)  # Pass through channel and remove dimension
            
            transmissions.append(t_channel)

        # Decode each transmission
        results = []
        for i in range(self.num_devices):
            xi = self.decoders[0 if self.shared_decoder else i]((transmissions[i], csi))
            if self.shared_decoder:
                xi = xi.view(xi.size(0), self.num_devices, 3, xi.size(2), xi.size(3))[:, i, ...]
            results.append(xi)
        
        return torch.stack(results, dim=1)