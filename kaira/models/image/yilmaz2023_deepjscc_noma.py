from kaira.models.base import BaseModel
import torch
from torch import nn

from src.models.components.channels import ComplexAWGNMAC


class Yilmaz2023DeepJSCCNOMA(BaseModel):
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
        """
        Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel implementation from :cite:`yilmaz2023distributed`.
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

    def forward(self, batch):
        x, csi = batch
        
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
        awgn = None
        transmissions = []
        
        # Apply encoders and channel with SIC
        for i in range(self.num_devices):
            t = self.encoders[0]((x[:, i, ...], csi))
            t = self.power_constraint(
                t[:, None, ...], 
                mult=torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))
            ).sum(dim=1)

            if awgn is None:
                awgn = torch.randn_like(t) * torch.sqrt(10.0 ** (-csi[..., None, None] / 10.0))
                if isinstance(self.channel, ComplexAWGNMAC):
                    awgn = awgn * torch.sqrt(torch.tensor(0.5, dtype=t.dtype, device=t.device))
            
            transmissions.append(t + awgn)

        # Decode each transmission
        results = []
        for i in range(self.num_devices):
            xi = self.decoders[0]((transmissions[i], csi))
            xi = xi.view(xi.size(0), self.num_devices, 3, xi.size(2), xi.size(3))[:, i, ...]
            results.append(xi)
        
        return torch.stack(results, dim=1)