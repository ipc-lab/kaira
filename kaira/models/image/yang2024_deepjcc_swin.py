import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


class _Mlp(nn.Module):
    """Multilayer perceptron implementation with two fully connected layers.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (defaults to in_features)
        out_features: Number of output features (defaults to in_features)
        act_layer: Activation layer (defaults to GELU)
        drop: Dropout probability (defaults to 0.0)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass of the MLP module.

        Applies the sequence of operations: linear transformation, activation,
        dropout, second linear transformation, and final dropout.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after applying all MLP operations
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _window_partition(x, window_size):
    """Partition the input tensor into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Size of each window (both height and width)

    Returns:
        windows: Partitioned windows of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def _window_reverse(windows, window_size, H, W):
    """Reverse the window partition operation.

    Args:
        windows: Window tensor of shape (num_windows*B, window_size, window_size, C)
        window_size: Size of each window (both height and width)
        H: Height of the original input
        W: Width of the original input

    Returns:
        x: Reconstructed tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class _WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.

    It supports both of shifted and non-shifted window.

    Args:
        dim: Number of input channels
        window_size: The height and width of the window
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to query, key, value (default: True)
        qk_scale: Override default qk scale of head_dim ** -0.5 if set (default: None)
        attn_drop: Dropout ratio of attention weight (default: 0.0)
        proj_drop: Dropout ratio of output (default: 0.0)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.registerBuffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, add_token=True, token_num=0, mask=None):
        """Forward pass for window-based multi-head self-attention.

        Args:
            x: Input features with shape (num_windows*B, N, C)
            add_token: Whether to add token-wise position bias (default: True)
            token_num: Number of tokens to offset when applying position bias (default: 0)
            mask: Optional attention mask with shape (num_windows, Wh*Ww, Wh*Ww) (default: None)

        Returns:
            Tensor with same shape as input after self-attention
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (N+1)x(N+1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Return a string containing extra representation of the module.

        This is used when printing the module details, showing key parameters
        such as dimension, window size, and number of heads.

        Returns:
            String with formatted module parameters
        """
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        """Calculate floating point operations (FLOPs) for this module.

        Args:
            N: Number of tokens in the window

        Returns:
            Total FLOPs required for one attention operation with N tokens
        """
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class _PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    This layer merges 2x2 neighboring patches into one, halving the spatial resolution
    and increasing the channel dimension.

    Args:
        input_resolution: Tuple of (H, W) representing resolution of input feature
        dim: Number of input channels
        out_dim: Number of output channels (defaults to dim*2)
        norm_layer: Normalization layer (default: nn.LayerNorm)
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H * W // 4, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string for the module.

        Includes information about input resolution and dimension.

        Returns:
            String representation with module parameters
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Calculate FLOPs for patch merging operation.

        Computes the total floating point operations for the downsampling
        and dimension change operations.

        Returns:
            Total FLOPs required for patch merging
        """
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class _PatchMerging4x(nn.Module):
    """Patch Merging for 4x downsampling.

    Applies patch merging twice to achieve 4x downsampling of spatial dimensions.

    Args:
        input_resolution: Tuple of (H, W) representing resolution of input feature
        dim: Number of input channels
        norm_layer: Normalization layer (default: nn.LayerNorm)
        use_conv: Whether to use convolution for downsampling (default: False)
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        H, W = input_resolution
        self.patch_merging1 = _PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = _PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        """Forward pass for 4x patch merging.

        Args:
            x: Input tensor
            H: Optional override for height resolution (default: None)
            W: Optional override for width resolution (default: None)

        Returns:
            Downsampled feature map with 1/4 spatial dimensions
        """
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H // 2, W // 2)
        return x


class _PatchReverseMerging(nn.Module):
    r"""Patch Reverse Merging Layer.

    This layer performs the reverse operation of PatchMerging, expanding spatial dimensions
    by 2x while reducing channel dimensions.

    Args:
        input_resolution: Tuple of (H, W) representing resolution of input feature
        dim: Number of input channels
        out_dim: Number of output channels
        norm_layer: Normalization layer (default: nn.LayerNorm)
    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string for the module.

        Includes information about input resolution and dimension.

        Returns:
            String representation with module parameters
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Calculate FLOPs for patch reverse merging operation.

        Computes the total floating point operations for the upsampling
        and dimension change operations.

        Returns:
            Total FLOPs required for patch reverse merging
        """
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class _PatchReverseMerging4x(nn.Module):
    """Patch Reverse Merging for 4x upsampling.

    Applies patch reverse merging twice to achieve 4x upsampling of spatial dimensions.

    Args:
        input_resolution: Tuple of (H, W) representing resolution of input feature
        dim: Number of input channels
        norm_layer: Normalization layer (default: nn.LayerNorm)
        use_conv: Whether to use transposed convolution for upsampling (default: False)
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = _PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = _PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        """Forward pass for 4x patch reverse merging.

        Args:
            x: Input tensor
            H: Optional override for height resolution (default: None)
            W: Optional override for width resolution (default: None)

        Returns:
            Upsampled feature map with 4x spatial dimensions
        """
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H * 2, W * 2)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string for the module.

        Includes information about input resolution and dimension.

        Returns:
            String representation with module parameters
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Calculate FLOPs for 4x patch reverse merging operation.

        Computes the total floating point operations for the 4x upsampling
        and dimension change operations.

        Returns:
            Total FLOPs required for 4x patch reverse merging
        """
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops


class _PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Splits an image into non-overlapping patches and then projects each patch into an embedding vector.

    Args:
        img_size: Size of input image (default: 224)
        patch_size: Size of each patch (default: 4)
        in_chans: Number of input channels (default: 3)
        embed_dim: Dimension of the embedding vector (default: 96)
        norm_layer: Normalization layer (default: None)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward pass for patch embedding.

        Takes an image tensor and embeds it into patch-wise tokens with
        projected dimensions.

        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            Patch embedding tensor [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """Calculate FLOPs for patch embedding operation.

        Computes the total floating point operations for projecting
        image patches to embedding vectors.

        Returns:
            Total FLOPs required for patch embedding
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class _SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    This block consists of a window-based multi-head self-attention module and a
    feed-forward network, with layer normalization applied before each module.

    Args:
        dim: Number of input channels
        input_resolution: Tuple of (H, W) representing resolution of input feature
        num_heads: Number of attention heads
        window_size: Size of attention window (default: 7)
        shift_size: Size of shifting for SW-MSA (default: 0)
        mlp_ratio: Ratio of mlp hidden dim to embedding dim (default: 4.0)
        qkv_bias: If True, add a learnable bias to query, key, value (default: True)
        qk_scale: Override default qk scale of head_dim ** -0.5 if set (default: None)
        act_layer: Activation layer (default: nn.GELU)
        norm_layer: Normalization layer (default: nn.LayerNorm)
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = _WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.registerBuffer("attn_mask", attn_mask)

    def forward(self, x):
        """Forward pass of the Swin Transformer block.

        Applies the following operations:
        1. Layer normalization and shortcut connection
        2. Window partitioning (with optional cyclic shift)
        3. Self-attention within windows
        4. Window reverse and cyclic shift reversal if applied
        5. Feed-forward network with second normalization

        Args:
            x: Input tensor [B, H*W, C]

        Returns:
            Output tensor with same shape after self-attention and FFN
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = _window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows, add_token=False, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        """Return extra representation string for the block.

        Includes key parameters like dimensions, resolution, and attention settings.

        Returns:
            String representation with block parameters
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """Calculate FLOPs for the entire Swin Transformer block.

        Computes the total floating point operations for normalization,
        window attention, and feed-forward network.

        Returns:
            Total FLOPs required for one block
        """
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        """Update the attention mask based on current input resolution.

        This is necessary when the input resolution changes, as the attention mask needs to be
        recomputed for the new dimensions.
        """
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class _BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    This layer consists of a series of Swin Transformer blocks followed by an optional
    patch merging (downsampling) layer.

    Args:
        dim: Number of input channels
        out_dim: Number of output channels
        input_resolution: Tuple of (H, W) representing resolution of input feature
        depth: Number of blocks in this layer
        num_heads: Number of attention heads
        window_size: Size of attention window
        mlp_ratio: Ratio of mlp hidden dim to embedding dim (default: 4.0)
        qkv_bias: If True, add a learnable bias to query, key, value (default: True)
        qk_scale: Override default qk scale of head_dim ** -0.5 if set (default: None)
        norm_layer: Normalization layer (default: nn.LayerNorm)
        downsample: Downsampling layer at the end of the layer (default: None)
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                _SwinTransformerBlock(
                    dim=out_dim, input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2), num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """Forward pass of the basic Swin Transformer layer.

        Applies optional downsampling followed by a series of transformer blocks.

        Args:
            x: Input feature tensor

        Returns:
            Processed feature tensor after all blocks
        """
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        """Return extra representation string for the layer.

        Includes information about dimension, resolution, and depth.

        Returns:
            String representation with layer parameters
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """Calculate FLOPs for the entire layer.

        Computes the total floating point operations for all blocks
        and the optional downsampling operation.

        Returns:
            Total FLOPs required for the layer
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        """Update the resolution of each block in the layer.

        Args:
            H: New height resolution
            W: New width resolution
        """
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)


class _AdaptiveModulator(nn.Module):
    """Adaptive modulator for rate and SNR adaptation.

    This module learns to generate modulation coefficients based on input SNR or rate values.
    These coefficients are used to adaptively scale features for different channel conditions.

    Args:
        hidden_dim: Hidden dimension size for the modulator network
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the adaptive modulator.

        Args:
            x: Input tensor of shape [B, 1] representing SNR or rate values

        Returns:
            Tensor of shape [B, hidden_dim] with modulation values
        """
        return self.fc(x)


@ModelRegistry.register_model()
class Yang2024DeepJSCCSwinEncoder(BaseModel):
    """Swin Transformer-based Joint Source-Channel Coding Encoder :cite:`yang2024swinjscc`.

    This encoder uses a Swin Transformer architecture with adaptive modulation capabilities for
    signal-to-noise ratio (SNR) and rate adaptation. It transforms input images into a latent
    representation optimized for transmission over noisy channels.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        patch_size: int,
        in_chans: int,
        embed_dims: List[int],
        depths: List[int],
        num_heads: List[int],
        C: Optional[int] = None,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        bottleneck_dim: int = 16,
        adaptation_hidden_factor: float = 1.5,
        adaptation_layers: int = 7,
        use_mixed_precision: bool = False,
        memory_efficient: bool = False,
    ):
        """Initialize the SwinJSCC encoder.

        Args:
            img_size: Input image size (height, width)
            patch_size: Patch size for embedding
            in_chans: Number of input channels
            embed_dims: List of embedding dimensions for each layer
            depths: List of depths for each layer
            num_heads: List of attention heads for each layer
            C: Output dimension (channel count)
            window_size: Size of attention windows
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QKV attention
            norm_layer: Normalization layer to use
            patch_norm: Whether to use normalization after patch embedding
            bottleneck_dim: Bottleneck dimension
            adaptation_hidden_factor: Factor for hidden dimension in adaptation networks
            adaptation_layers: Number of layers in adaptation networks
            use_mixed_precision: Whether to use mixed precision for performance
            memory_efficient: Whether to use memory-efficient operations
        """
        super().__init__()

        # Validate inputs
        if len(depths) != len(num_heads) or len(depths) != len(embed_dims) - 1:
            raise ValueError("Lengths of depths, num_heads, and embed_dims must match")

        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size

        # Convert scalar dimensions to tuples if needed
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2**self.num_layers)
        self.W = img_size[1] // (2**self.num_layers)

        # Patch embedding layer
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if patch_norm else None)

        # Dimensions for adaptation networks
        self.hidden_dim = int(self.embed_dims[-1] * adaptation_hidden_factor)
        self.layer_num = adaptation_layers

        # Build transformer layers
        self.layers = self._build_layers(embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer)

        self.norm = norm_layer(embed_dims[-1])

        # Channel output projection
        self.head = nn.Linear(embed_dims[-1], C) if C is not None else None

        # Initialize adaptation modules
        self.snr_adaptation = self._build_adaptation_network()
        self.rate_adaptation = self._build_adaptation_network()

        # Initialize weights
        self.apply(self._init_weights)

        # Logger for warnings and info
        self.logger = logging.getLogger(__name__)

        # Performance optimization settings
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient

    def _build_layers(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        """Build the transformer layers of the encoder."""
        layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = _BasicLayer(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else self.in_chans,
                out_dim=int(embed_dims[i_layer]),
                input_resolution=(self.patches_resolution[0] // (2**i_layer), self.patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=_PatchMerging if i_layer != 0 else None,
            )

            self.logger.info(f"Encoder layer {i_layer}: {layer.extra_repr()}")
            layers.append(layer)

        return layers

    def _build_adaptation_network(self):
        """Build an adaptation network (SNR or rate)"""
        modulators = nn.ModuleList()
        linears = nn.ModuleList()

        # First linear projects from embedding dim to hidden dim
        linears.append(nn.Linear(self.embed_dims[-1], self.hidden_dim))

        # Build layers for adaptation
        for i in range(self.layer_num):
            out_dim = self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim
            modulators.append(_AdaptiveModulator(self.hidden_dim))
            linears.append(nn.Linear(self.hidden_dim, out_dim))

        return {"modulators": modulators, "linears": linears, "sigmoid": nn.Sigmoid()}

    def forward(self, x: torch.Tensor, snr: Optional[float] = None, rate: Optional[int] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass through the encoder.

        Args:
            x: Input images [B, C, H, W]
            snr: Signal-to-noise ratio for SNR adaptation
            rate: Target bit rate for rate adaptation
            model_mode: Model variant to use:
                - 'SwinJSCC_w/o_SAandRA': Base model without adaptations
                - 'SwinJSCC_w/_SA': With SNR adaptation
                - 'SwinJSCC_w/_RA': With rate adaptation
                - 'SwinJSCC_w/_SAandRA': With both SNR and rate adaptation
            return_intermediate_features: Whether to return intermediate feature maps

        Returns:
            Output features, and optionally mask for rate adaptation and/or intermediate features
        """
        B, C, H, W = x.size()

        # Use mixed precision if enabled
        with torch.cuda.amp.autocast() if self.use_mixed_precision and torch.cuda.is_available() else nullcontext():
            # Store intermediate features if requested
            intermediate_features = {} if return_intermediate_features else None

            # Patch embedding and feature extraction
            x = self.patch_embed(x)
            if return_intermediate_features:
                intermediate_features["patch_embed"] = x.detach().clone()

            for i, layer in enumerate(self.layers):
                x = layer(x)
                if return_intermediate_features:
                    intermediate_features[f"layer_{i}"] = x.detach().clone()

            x = self.norm(x)
            if return_intermediate_features:
                intermediate_features["norm"] = x.detach().clone()

            # Apply the specified adaptation strategy
            result = None

            if model_mode == "SwinJSCC_w/o_SAandRA":
                if self.head is not None:
                    x = self.head(x)
                result = x

            elif model_mode == "SwinJSCC_w/_SA":
                if snr is None:
                    raise ValueError("SNR must be provided for SNR adaptation mode")
                result = self._apply_snr_adaptation(x, snr, B, H, W)

            elif model_mode == "SwinJSCC_w/_RA":
                if rate is None:
                    raise ValueError("Rate must be provided for rate adaptation mode")
                result = self._apply_rate_adaptation(x, rate, B, H, W)

            elif model_mode == "SwinJSCC_w/_SAandRA":
                if snr is None or rate is None:
                    raise ValueError("Both SNR and rate must be provided for joint adaptation mode")
                result = self._apply_snr_and_rate_adaptation(x, snr, rate, B, H, W)

            else:
                raise ValueError(f"Unknown model mode: {model_mode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'")

        # Handle intermediate features
        if return_intermediate_features:
            if isinstance(result, tuple):
                return (*result, intermediate_features)
            else:
                return result, intermediate_features

        return result

    def _apply_adaptation(self, x: torch.Tensor, value: float, network: Dict, batch_size: int, spatial_dim: int) -> torch.Tensor:
        """Common adaptation logic for SNR or rate adaptation.

        This method applies the adaptive modulation network to scale features based on
        an input value (SNR or rate).

        Args:
            x: Feature tensor to modulate
            value: The adaptation value (SNR in dB or rate in bits)
            network: Dictionary containing the adaptation network modules
            batch_size: Batch size for processing
            spatial_dim: Spatial dimension of features

        Returns:
            Tuple containing modulated features and modulation values
        """
        device = x.device
        value_tensor = torch.tensor(value, dtype=torch.float).to(device)
        value_batch = value_tensor.unsqueeze(0).expand(batch_size, -1)

        # Use detached features for adaptation path
        temp = x.detach()

        # Apply adaptation network
        modulators = network["modulators"]
        linears = network["linears"]

        # First linear
        temp = linears[0](temp)

        # Process through layers
        for i in range(self.layer_num):
            if i > 0:
                temp = linears[i](temp)

            # Apply modulation
            bm = modulators[i](value_batch).unsqueeze(1).expand(-1, spatial_dim, -1)
            temp = temp * bm

        # Final modulation values
        mod_val = network["sigmoid"](linears[-1](temp))

        # Apply modulation to features
        return x * mod_val, mod_val

    def _apply_snr_adaptation(self, x: torch.Tensor, snr: float, batch_size: int, H: int, W: int) -> torch.Tensor:
        """Apply SNR adaptation to features."""
        spatial_dim = H * W // (4**self.num_layers)
        x, _ = self._apply_adaptation(x, snr, self.snr_adaptation, batch_size, spatial_dim)

        if self.head is not None:
            x = self.head(x)
        return x

    def _apply_rate_adaptation(self, x: torch.Tensor, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rate adaptation to features."""
        spatial_dim = H * W // (4**self.num_layers)
        x, mod_val = self._apply_adaptation(x, rate, self.rate_adaptation, batch_size, spatial_dim)

        # Generate channel selection mask based on rate
        mask = self._generate_rate_mask(x, mod_val, rate, batch_size, spatial_dim)

        # Apply mask
        x = x * mask
        return x, mask

    def _generate_rate_mask(self, x: torch.Tensor, mod_val: torch.Tensor, rate: int, batch_size: int, spatial_dim: int) -> torch.Tensor:
        """Generate mask for rate adaptation based on modulation values."""
        device = x.device

        # Calculate importance of each channel
        channel_importance = torch.sum(mod_val, dim=1)

        # Select top channels
        _, indices = channel_importance.sort(dim=1, descending=True)
        selected_indices = indices[:, :rate]

        # Create batch offsets for flattened indexing
        batch_offsets = torch.arange(0, batch_size * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, rate)
        flattened_indices = selected_indices + batch_offsets.int()

        # Create mask from selected indices
        flat_mask = torch.zeros(channel_importance.size(), device=device).reshape(-1)
        flat_mask[flattened_indices.reshape(-1)] = 1

        # Reshape mask to match feature dimensions
        mask = flat_mask.reshape(batch_size, x.size(2))
        mask = mask.unsqueeze(1).expand(-1, spatial_dim, -1)

        return mask

    def _apply_snr_and_rate_adaptation(self, x: torch.Tensor, snr: float, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply both SNR and rate adaptation to features."""
        spatial_dim = H * W // (4**self.num_layers)

        # First apply SNR adaptation
        x, _ = self._apply_adaptation(x, snr, self.snr_adaptation, batch_size, spatial_dim)

        # Then apply rate adaptation
        x, mod_val = self._apply_adaptation(x, rate, self.rate_adaptation, batch_size, spatial_dim)

        # Generate and apply mask
        mask = self._generate_rate_mask(x, mod_val, rate, batch_size, spatial_dim)
        x = x * mask

        return x, mask

    def _init_weights(self, m):
        """Initialize weights for network layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_resolution(self, H: int, W: int):
        """Update the resolution of internal components."""
        self.patches_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)), W // (2 ** (i_layer + 1)))

    def flops(self) -> int:
        """Calculate the floating point operations for the model."""
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
        return flops

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that shouldn't use weight decay."""
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return parameter keywords that shouldn't use weight decay."""
        return {"relative_position_bias_table"}

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency."""
        if not self.memory_efficient:
            return

        def fn(module):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable

        self.apply(fn)

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Return model size information."""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total_params": num_params, "trainable_params": num_trainable, "param_size_mb": num_params * 4 / 1024 / 1024, "flops_g": self.flops() / 1e9}  # Assuming float32


@ModelRegistry.register_model()
class Yang2024DeepJSCCSwinDecoder(BaseModel):
    """Swin Transformer-based Joint Source-Channel Coding Decoder :cite:`yang2024swinjscc`.

    This decoder takes the encoded representation from Yang2024DeepJSCCSwinEncoder and reconstructs
    the original image. It supports SNR adaptation for robust performance across varying channel
    conditions.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]],
        embed_dims: List[int],
        depths: List[int],
        num_heads: List[int],
        C: Optional[int] = None,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        bottleneck_dim: int = 16,
        adaptation_hidden_factor: float = 1.5,
        adaptation_layers: int = 7,
        use_mixed_precision: bool = False,
        memory_efficient: bool = False,
    ):
        """Initialize the SwinJSCC decoder.

        Args:
            img_size: Output image size (height, width)
            embed_dims: List of embedding dimensions for each layer
            depths: List of depths for each layer
            num_heads: List of attention heads for each layer
            C: Input dimension from encoder
            window_size: Size of attention windows
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QKV attention
            norm_layer: Normalization layer to use
            ape: Whether to use absolute position embedding
            patch_norm: Whether to use normalization
            bottleneck_dim: Bottleneck dimension
            adaptation_hidden_factor: Factor for hidden dimension in adaptation networks
            adaptation_layers: Number of layers in adaptation networks
            use_mixed_precision: Whether to use mixed precision for performance
            memory_efficient: Whether to use memory-efficient operations
        """
        super().__init__()

        # Validate inputs
        if len(depths) != len(num_heads) or len(depths) != len(embed_dims) - 1:
            raise ValueError("Lengths of depths, num_heads, and embed_dims must match")

        # Convert scalar dimensions to tuples if needed
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))

        # Positional embedding if enabled
        if self.ape:
            num_patches = self.H // 4 * self.W // 4
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Input projection from encoder channels
        self.head = nn.Linear(C, embed_dims[0]) if C is not None else None

        # Build layers
        self.layers = self._build_layers(embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer)

        # Initialize SNR adaptation modules
        self.hidden_dim = int(embed_dims[0] * adaptation_hidden_factor)
        self.layer_num = adaptation_layers
        self.snr_adaptation = self._build_adaptation_network()

        self.apply(self._init_weights)

        # Logger for warnings and info
        self.logger = logging.getLogger(__name__)

        # Performance optimization settings
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient

    def _build_layers(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        """Build the transformer layers of the decoder."""
        layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = _BasicLayer(
                dim=int(embed_dims[i_layer]),
                out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                input_resolution=(self.patches_resolution[0] * (2**i_layer), self.patches_resolution[1] * (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                upsample=_PatchReverseMerging,
            )

            self.logger.info(f"Decoder layer {i_layer}: {layer.extra_repr()}")
            layers.append(layer)

        return layers

    def _build_adaptation_network(self):
        """Build an SNR adaptation network."""
        modulators = nn.ModuleList()
        linears = nn.ModuleList()

        # First linear projects from embedding dim to hidden dim
        linears.append(nn.Linear(self.embed_dims[0], self.hidden_dim))

        # Build layers for adaptation
        for i in range(self.layer_num):
            out_dim = self.embed_dims[0] if i == self.layer_num - 1 else self.hidden_dim
            modulators.append(_AdaptiveModulator(self.hidden_dim))
            linears.append(nn.Linear(self.hidden_dim, out_dim))

        return {"modulators": modulators, "linears": linears, "sigmoid": nn.Sigmoid()}

    def forward(self, x: torch.Tensor, snr: Optional[float] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the decoder.

        Args:
            x: Input features from encoder
            snr: Signal-to-noise ratio for SNR adaptation
            model_mode: Model variant to use:
                - 'SwinJSCC_w/o_SAandRA': Base model without adaptations
                - 'SwinJSCC_w/_SA': With SNR adaptation
                - 'SwinJSCC_w/_RA': With rate adaptation (no SNR adaptation needed)
                - 'SwinJSCC_w/_SAandRA': With both SNR and rate adaptation
            return_intermediate_features: Whether to return intermediate feature maps

        Returns:
            Output images [B, 3, H, W] and optionally intermediate features
        """
        # Use mixed precision if enabled
        with torch.cuda.amp.autocast() if self.use_mixed_precision and torch.cuda.is_available() else nullcontext():
            # Store intermediate features if requested
            intermediate_features = {} if return_intermediate_features else None

            if model_mode == "SwinJSCC_w/o_SAandRA":
                result = self._forward_base(x, intermediate_features)
            elif model_mode == "SwinJSCC_w/_RA":
                result = self._forward_base(x, intermediate_features)  # No SNR adaptation for rate-only mode
            elif model_mode == "SwinJSCC_w/_SA" or model_mode == "SwinJSCC_w/_SAandRA":
                if snr is None:
                    raise ValueError("SNR must be provided for SNR adaptation mode")
                result = self._forward_with_snr(x, snr, intermediate_features)
            else:
                raise ValueError(f"Unknown model mode: {model_mode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'")

        if return_intermediate_features:
            return result, intermediate_features
        return result

    def _forward_base(self, x: torch.Tensor, intermediate_features: Optional[Dict] = None) -> torch.Tensor:
        """Base forward pass without adaptation.

        Processes input features through the decoder layers without applying SNR adaptation.

        Args:
            x: Input features from encoder
            intermediate_features: Optional dictionary to store intermediate activations

        Returns:
            Decoded image tensor of shape [B, 3, H, W]
        """
        if self.head is not None:
            x = self.head(x)
            if intermediate_features is not None:
                intermediate_features["head"] = x.detach().clone()

        # Process through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if intermediate_features is not None:
                intermediate_features[f"layer_{i}"] = x.detach().clone()

        # Reshape to image format
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        if intermediate_features is not None:
            intermediate_features["output"] = x.detach().clone()

        return x

    def _forward_with_snr(self, x: torch.Tensor, snr: float, intermediate_features: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass with SNR adaptation.

        Applies SNR-dependent modulation before processing features through decoder layers.

        Args:
            x: Input features from encoder
            snr: Signal-to-noise ratio value in dB
            intermediate_features: Optional dictionary to store intermediate activations

        Returns:
            Decoded image tensor of shape [B, 3, H, W]
        """
        B, L, C = x.size()
        device = x.device

        # Apply head projection if available
        if self.head is not None:
            x = self.head(x)

        # Apply SNR adaptation
        snr_tensor = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)

        # Use the adaptation network
        network = self.snr_adaptation
        modulators = network["modulators"]
        linears = network["linears"]

        # Use detached features for adaptation path
        temp = x.detach()

        # First linear
        temp = linears[0](temp)

        # Process through layers
        for i in range(self.layer_num):
            if i > 0:
                temp = linears[i](temp)

            # Apply modulation
            bm = modulators[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm

        # Apply final modulation to features
        mod_val = network["sigmoid"](linears[-1](temp))
        x = x * mod_val

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x)

        # Reshape to image format
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        return x

    def _init_weights(self, m):
        """Initialize weights for network layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_resolution(self, H: int, W: int):
        """Update the resolution of internal components."""
        self.patches_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)

        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2**i_layer), W * (2**i_layer))

    def flops(self) -> int:
        """Calculate the floating point operations for the model."""
        flops = 0
        for layer in self.layers:
            flops += layer.flops()
        return flops

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return parameters that shouldn't use weight decay."""
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return parameter keywords that shouldn't use weight decay."""
        return {"relative_position_bias_table"}

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency."""
        if not self.memory_efficient:
            return

        def fn(module):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable

        self.apply(fn)

    def get_model_size(self) -> Dict[str, Union[int, float]]:
        """Return model size information."""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total_params": num_params, "trainable_params": num_trainable, "param_size_mb": num_params * 4 / 1024 / 1024, "flops_g": self.flops() / 1e9}  # Assuming float32


class nullcontext:
    """A context manager that does nothing.

    Used as a placeholder when no context manager is needed but the code structure expects one.
    This is particularly useful for conditional context management.
    """

    def __enter__(self):
        """Enter the context, returning None.

        This method is called when executing the `with` statement.
        Since this is a null context manager, it performs no action.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, always returning False to propagate exceptions.

        Args:
            exc_type: Exception type if an exception occurred, otherwise None
            exc_val: Exception value if an exception occurred, otherwise None
            exc_tb: Exception traceback if an exception occurred, otherwise None

        Returns:
            False implicitly (None), which means any exceptions will be propagated
        """
        pass


class SwinJSCCConfig:
    """Configuration object for SwinJSCC encoder and decoder.

    This class provides a centralized way to manage configuration parameters for both the encoder
    and decoder models, with methods to generate the appropriate keyword arguments for model
    initialization.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        bottleneck_dim: int = 16,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        patch_norm: bool = True,
        ape: bool = False,
        use_mixed_precision: bool = False,
        memory_efficient: bool = False,
        adaptation_hidden_factor: float = 1.5,
        adaptation_layers: int = 7,
    ):
        """Initialize configuration for SwinJSCC models.

        Args:
            img_size: Input/output image size
            patch_size: Patch size for embedding
            in_chans: Number of input channels
            embed_dims: List of embedding dimensions for each layer
            depths: List of depths for each layer
            num_heads: List of attention heads for each layer
            bottleneck_dim: Bottleneck dimension
            window_size: Size of attention windows
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QKV attention
            patch_norm: Whether to use normalization after patch embedding
            ape: Whether to use absolute position embedding
            use_mixed_precision: Whether to use mixed precision for performance
            memory_efficient: Whether to use memory-efficient operations
            adaptation_hidden_factor: Factor for hidden dimension in adaptation networks
            adaptation_layers: Number of layers in adaptation networks
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_norm = patch_norm
        self.ape = ape
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient
        self.adaptation_hidden_factor = adaptation_hidden_factor
        self.adaptation_layers = adaptation_layers

    def get_encoder_kwargs(self, C: int = None) -> Dict[str, Any]:
        """Get keyword arguments for encoder constructor."""
        return {
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dims": self.embed_dims,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "C": C,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "patch_norm": self.patch_norm,
            "bottleneck_dim": self.bottleneck_dim,
            "adaptation_hidden_factor": self.adaptation_hidden_factor,
            "adaptation_layers": self.adaptation_layers,
            "use_mixed_precision": self.use_mixed_precision,
            "memory_efficient": self.memory_efficient,
        }

    def get_decoder_kwargs(self, C: int = None) -> Dict[str, Any]:
        """Get keyword arguments for decoder constructor."""
        return {
            "img_size": self.img_size,
            "embed_dims": self.embed_dims,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "C": C,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "ape": self.ape,
            "patch_norm": self.patch_norm,
            "bottleneck_dim": self.bottleneck_dim,
            "adaptation_hidden_factor": self.adaptation_hidden_factor,
            "adaptation_layers": self.adaptation_layers,
            "use_mixed_precision": self.use_mixed_precision,
            "memory_efficient": self.memory_efficient,
        }

    @classmethod
    def from_preset(cls, preset: str = "tiny") -> "SwinJSCCConfig":
        """Create a configuration from a preset name."""
        presets = {
            "tiny": {"embed_dims": [96, 192, 384, 768], "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24], "window_size": 7},
            "small": {"embed_dims": [96, 192, 384, 768], "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24], "window_size": 7},
            "base": {"embed_dims": [128, 256, 512, 1024], "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32], "window_size": 7},
            "large": {"embed_dims": [192, 384, 768, 1536], "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48], "window_size": 7},
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available presets: {list(presets.keys())}")

        return cls(**presets[preset])


def create_swin_jscc_models(config: SwinJSCCConfig, channel_dim: int, device: Optional[torch.device] = None) -> Tuple[Yang2024DeepJSCCSwinEncoder, Yang2024DeepJSCCSwinDecoder]:
    """Create a pair of SwinJSCC encoder and decoder models.

    This is a convenience function that creates and initializes both encoder and decoder
    models from the same configuration, ensuring compatibility between them.

    Args:
        config: Configuration object for the models
        channel_dim: Dimension of the bottleneck/channel
        device: Device to create the models on (default: None)

    Returns:
        A tuple containing the encoder and decoder models
    """
    # Create encoder
    encoder = create_encoder(**config.get_encoder_kwargs(channel_dim))

    # Create decoder
    decoder = create_decoder(**config.get_decoder_kwargs(channel_dim))

    # Move to device if specified
    if device is not None:
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    return encoder, decoder


def create_encoder(**kwargs) -> Yang2024DeepJSCCSwinEncoder:
    """Create a Swin JSCC encoder model with the given parameters.

    This is a simple factory function to instantiate an encoder model.

    Args:
        **kwargs: Keyword arguments to pass to the encoder constructor

    Returns:
        An initialized Yang2024DeepJSCCSwinEncoder instance
    """
    return Yang2024DeepJSCCSwinEncoder(**kwargs)


def create_decoder(**kwargs) -> Yang2024DeepJSCCSwinDecoder:
    """Create a Swin JSCC decoder model with the given parameters.

    This is a simple factory function to instantiate a decoder model.

    Args:
        **kwargs: Keyword arguments to pass to the decoder constructor

    Returns:
        An initialized Yang2024DeepJSCCSwinDecoder instance
    """
    return Yang2024DeepJSCCSwinDecoder(**kwargs)


def build_model(config):
    """Build and initialize SwinJSCC models based on config.

    Validates the model by running a test input and reports statistics. This function
    handles the entire process of creating and initializing the model with error handling.

    Args:
        config: Configuration object with encoder_kwargs and device

    Returns:
        The built encoder model

    Raises:
        Exception: If model building fails
    """
    logger = logging.getLogger(__name__)

    try:
        device = getattr(config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        img_size = config.encoder_kwargs.get("img_size", 256)

        # Create test input
        if isinstance(img_size, (list, tuple)):
            test_input = torch.ones([1, 3, img_size[0], img_size[1]], device=device)
        else:
            test_input = torch.ones([1, 3, img_size, img_size], device=device)

        # Build encoder
        model = create_encoder(**config.encoder_kwargs).to(device)

        # Test forward pass
        with torch.no_grad():
            _ = model(test_input)

        # Report statistics
        size_info = model.get_model_size()
        logger.info(f"Created SwinJSCC encoder with {size_info['total_params']/1e6:.2f}M parameters")
        logger.info(f"Estimated FLOPs: {size_info['flops_g']:.2f}G")

        return model

    except Exception as e:
        logger.error(f"Failed to build model: {str(e)}")
        raise
