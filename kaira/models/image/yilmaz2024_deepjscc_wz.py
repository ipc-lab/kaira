
from kaira.models.components.afmodule import AFModule
from torch import nn
from kaira.models.base import BaseModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
import torch

# TODO: Improve docstrings
# Implement Wyner Ziv Pipeline

class Yilmaz2024DeepJSCCWZSmallEncoder(BaseModel):
    """DeepJSCC-WZ-sm Encoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCWZEncoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 2),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 2),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 2),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2), 
            AFModule(M, 2),
            AttentionBlock(M),
        ])
    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        
        csi_transmitter = torch.cat([csi, torch.zeros_like(csi)], dim=1)
        
        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, csi_transmitter))
            else:
                x = layer(x)

        return x

class Yilmaz2024DeepJSCCWZSmallDecoder(BaseModel):
    """DeepJSCC-WZ-sm Decoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int, encoder: BaseModel) -> None:
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(2 * M),
            ResidualBlock(
                in_ch=2 * M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            AttentionBlock(2 * N),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2 * N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=3,
                upsample=2),
            AFModule(3*2, 1),
            ResidualBlock(
                in_ch=3*2,
                out_ch=3),
        ])
        
        self.encoder = encoder

    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        csi_sideinfo = torch.cat([csi, torch.ones_like(csi)], dim=1)

        xs_list = []
        for idx, layer in enumerate(self.g_a):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(x_side)
            
            if isinstance(layer, AFModule):
                x_side = layer((x_side, csi_sideinfo))
            else:
                x_side = layer(x_side)
        
        xs_list.append(x_side)
        
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)
            
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)

        return x

class Yilmaz2024DeepJSCCWZEncoder(BaseModel):
    """DeepJSCC-WZ Encoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCWZEncoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])
        
        self.g_a2 = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])
    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:

        csi_transmitter = csi
        
        for layer in self.g_a:
            if isinstance(layer, AFModule):
                x = layer((x, csi_transmitter))
            else:
                x = layer(x)

        return x

class Yilmaz2024DeepJSCCWZDecoder(BaseModel):
    """DeepJSCC-WZ Decoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCWZDecoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(2 * M),
            ResidualBlock(
                in_ch=2 * M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            AttentionBlock(2 * N),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2 * N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=3,
                upsample=2),
            AFModule(3*2, 1),
            ResidualBlock(
                in_ch=3*2,
                out_ch=3),
        ])

    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        csi_sideinfo = csi
        
        xs_list = []
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(x_side)
            
            if isinstance(layer, AFModule):
                xs = layer((x_side, csi_sideinfo))
            else:
                xs = layer(xs)
        
        xs_list.append(xs)
        
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)
            
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)

        return x

class Yilmaz2024DeepJSCCWZConditionalEncoder(BaseModel):
    """DeepJSCC-WZ Encoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCWZEncoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_a = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=6,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=2*N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])
        
        self.g_a2 = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=M,
                stride=2), 
            AFModule(M, 1),
            AttentionBlock(M),
        ])
        
        self.g_a3 = nn.ModuleList([
            ResidualBlockWithStride(
                in_ch=3,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            AttentionBlock(N),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            ResidualBlockWithStride(
                in_ch=N,
                out_ch=N,
                stride=2),
            AFModule(N, 1),
            ResidualBlock(
                in_ch=N,
                out_ch=N),
            None,
            None,
            None
        ])
    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        xs_encoder = x_side
        
        csi_transmitter = csi

        for layer, layer_s in zip(self.g_a, self.g_a3):
            if isinstance(layer, ResidualBlockWithStride):
                x = torch.cat([x, xs_encoder], dim=1)
            
            if isinstance(layer, AFModule):
                x = layer((x, csi_transmitter))
                if layer_s is not None:
                    xs_encoder = layer_s((xs_encoder, csi_transmitter))
            else:
                x = layer(x)
                if layer_s is not None:
                    xs_encoder = layer_s(xs_encoder)

        return x

class Yilmaz2024DeepJSCCWZConditionalDecoder(BaseModel):
    """DeepJSCC-WZ Decoder Module :cite:`yilmaz2024deepjsccwz`.

    """
    def __init__(self, N: int, M: int) -> None:
        """Initialize the DeepJSCCWZDecoder.

        Args:
            N (int): The number of output channels for the ResidualBlocks.
            M (int): The number of output channels in the last convolutional layer of the network.
        """
        super().__init__()

        self.g_s = nn.ModuleList([
            AttentionBlock(2 * M),
            ResidualBlock(
                in_ch=2 * M,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2*N, 1),
            AttentionBlock(2 * N),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=N,
                upsample=2),
            AFModule(2 * N, 1),
            ResidualBlock(
                in_ch=2 * N,
                out_ch=N),
            ResidualBlockUpsample(
                in_ch=N,
                out_ch=3,
                upsample=2),
            AFModule(3*2, 1),
            ResidualBlock(
                in_ch=3*2,
                out_ch=3),
        ])

    
    def forward(self, x: torch.Tensor, x_side: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        csi_sideinfo = csi
        
        xs_list = []
        for idx, layer in enumerate(self.g_a2):
            if isinstance(layer, ResidualBlockWithStride):
                xs_list.append(x_side)
            
            if isinstance(layer, AFModule):
                xs = layer((x_side, csi_sideinfo))
            else:
                xs = layer(xs)
        
        xs_list.append(xs)
        
        for idx, layer in enumerate(self.g_s):
            if idx in [0, 3, 6, 10, 13]:
                last_xs = xs_list.pop()
                x = torch.cat([x, last_xs], dim=1)
            
            if isinstance(layer, AFModule):
                x = layer((x, csi))
            else:
                x = layer(x)

        return x