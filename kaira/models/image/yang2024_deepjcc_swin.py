import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features a nonlinearity and dropout in between.
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() input features
        self.fc2 = nn.Linear(hidden_features, out_features)s to input features)
        self.drop = nn.Dropout(drop)ut features (defaults to input features)
        act_layer: Activation layer (defaults to GELU)
    def forward(self, x):e (defaults to 0.0)
        x = self.fc1(x)
        x = self.act(x)in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        x = self.drop(x)()
        x = self.fc2(x)out_features or in_features
        x = self.drop(x)= hidden_features or in_features
        return x = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
def _window_partition(x, window_size):
    """
    Args:orward(self, x):
        x: (B, H, W, C)
        window_size (int): window size
    Returns:self.drop(x)
        windows: (num_windows*B, window_size, window_size, C)
    """ x = self.drop(x)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windowsion(x, window_size):
    """Partitions input tensor into non-overlapping windows.
    
def _window_reverse(windows, window_size, H, W):to windows of specified size
    """ efficient computation of window-based self-attention.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size, W, C)
        H (int): Height of imageh window (both height and width)
        W (int): Width of image
    Returns:
        x: (B, H, W, C)ioned windows of shape (num_windows*B, window_size, window_size, C)
    """     where num_windows = (H//window_size) * (W//window_size)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1) window_size, C)
    return x= x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class _WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    """Reverses the window partition operation.
    It supports both shifted and non-shifted window configurations.
    function merges windows back into the original tensor shape.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.s*B, window_size, window_size, C)
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0urns:
    """        x: Reconstructed tensor of shape (B, H, W, C)

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):0] / (H * W / window_size / window_size))
        super().__init__(), H // window_size, W // window_size, window_size, window_size, -1)
        self.dim = dim.view(B, H, W, -1)
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5class _WindowAttention(nn.Module):
e with relative position bias.
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH    It supports both of shifted and non-shifted window.

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])th of the window.
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Wwbias to query, key, value. Default: True
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1le=None, attn_drop=0.0, proj_drop=0.0):
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)5
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02) = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.softmax = nn.Softmax(dim=-1)
side the window
    def forward(self, x, add_token=True, token_num=0, mask=None):rds_h = torch.arange(self.window_size[0])
        """Compute self-attention within each window.s_w = torch.arange(self.window_size[1])
        )  # 2, Wh, Ww
        Args:
            x: Input features with shape (num_windows*B, N, C)ative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            add_token (bool): Whether to add special tokens in the attention maprelative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            token_num (int): Number of special tokens        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            mask: Optional attention mask of shape (num_windows, Wh*Ww, Wh*Ww) or None
            
        Returns:        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            Attention output of shape (num_windows*B, N, C)er("relative_position_index", relative_position_index)
        """
        B_, N, C = x.shape        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)        self.proj_drop = nn.Dropout(proj_drop)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (N+1)x(N+1)softmax = nn.Softmax(dim=-1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH    def forward(self, x, add_token=True, token_num=0, mask=None):
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        if add_token:ape of (num_windows*B, N, C)
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)ne
        else:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None: N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            if add_token: v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)        q = q * self.scale
            nW = mask.shape[0]-1)  # (N+1)x(N+1)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            attn = self.softmax(attn)n_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        else:
            attn = self.softmax(attn)oken:
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)None:
        x = self.proj_drop(x)
        return x padding mask matrix
d(mask, (token_num, 0, token_num, 0), "constant", 0)
    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}" self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)

    def flops(self, N):oftmax(attn)
        # calculate flops for 1 window with token length of N
        flops = 0ftmax(attn)
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dimattn_drop(attn)
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads))
        # x = self.proj(x)        return x
        flops += N * self.dim * self.dim
        return flops
={self.window_size}, num_heads={self.num_heads}"

class _PatchMerging(nn.Module): flops(self, N):
    r"""Patch Merging Layer.        # calculate flops for 1 window with token length of N
    
    This layer merges 2x2 neighboring patches together, reducing feature map resolution)
    while increasing the channel dimension.
nspose(-2, -1))
    Args:_heads * N * (self.dim // self.num_heads) * N
        input_resolution (tuple[int]): Resolution of input feature map (H, W). v)
        dim (int): Number of input channels.heads)
        out_dim (int, optional): Number of output channels. Default: None (2*dim)
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()hMerging(nn.Module):
        self.input_resolution = input_resolutiong Layer.
        if out_dim is None:
            out_dim = dim
        self.dim = dim(tuple[int]): Resolution of input feature.
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)r of input channels.
        self.norm = norm_layer(4 * dim)l): Normalization layer.  Default: nn.LayerNorm
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)
esolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
    def forward(self, x):
        """Forward function.n
        
        Args:
            x: Input tensor of shape (B, H*W, C)
            =False)
        Returns:_layer(4 * dim)
            Output tensor of shape (B, H*W/4, out_dim)2d(dim, out_dim, kernel_size=2, stride=2)
        """        # self.norm = nn.LayerNorm(out_dim)
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)W, C
        assert L == H * W, "input feature has wrong size"        """
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."tion
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C        # print(x.shape)
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C.input_resolution)
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 Cature has wrong size"
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C% 2 == 0, f"x size ({H}*{W}) are not even."
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H * W // 4, 4 * C)  # B H/2*W/2 4*C:2, 0::2, :]  # B H/2 W/2 C
        x = self.norm(x)        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x = self.reduction(x)        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
 # B H/2 W/2 C
        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)W // 4, 4 * C)  # B H/2*W/2 4*C
        # x = self.norm(x)
        return x

    def extra_repr(self) -> str:        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return f"input_resolution={self.input_resolution}, dim={self.dim}"nspose(1, 2)
rm(x)
    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim"input_resolution={self.input_resolution}, dim={self.dim}"
        return flops
    def flops(self):

class _PatchMerging4x(nn.Module):.dim
    """4x downsampling by applying patch merging twice.        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
    eturn flops
    This module applies the _PatchMerging operation twice to achieve 
    4x downsampling in spatial dimensions.
    
    Args: __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        input_resolution (tuple[int]): Resolution of input feature map (H, W).        super().__init__()
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm1 = _PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        use_conv (bool): Whether to use convolution for patch merging. Default: False/ 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
    """
e, W=None):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()ution
        H, W = input_resolution
        self.patch_merging1 = _PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv) 2, W // 2)
        self.patch_merging2 = _PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)        return x

    def forward(self, x, H=None, W=None):
        if H is None:erging(nn.Module):
            H, W = self.input_resolutionch Merging Layer.
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H // 2, W // 2)
        return xt feature.

odule, optional): Normalization layer.  Default: nn.LayerNorm
class _PatchReverseMerging(nn.Module):
    r"""Patch Reverse Merging Layer.
    lution, dim, out_dim, norm_layer=nn.LayerNorm):
    This layer performs the reverse of patch merging, effectively upsampling
    the feature map by 2x in spatial dimensions.ion = input_resolution

    Args:
        input_resolution (tuple[int]): Resolution of input feature map (H, W).rement = nn.Linear(dim, out_dim * 4, bias=False)
        dim (int): Number of input channels.        self.norm = norm_layer(dim)
        out_dim (int): Number of output channels.nspose2d(dim // 4, 3, 3, stride=1, padding=1)
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
, x):
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = diminput_resolution
        self.out_dim = out_dim        B, L, C = x.shape
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)        assert L == H * W, "input feature has wrong size"
        self.norm = norm_layer(dim), f"x size ({H}*{W}) are not even."
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)(x)
permute(0, 3, 1, 2)
    def forward(self, x):
        """j(x).flatten(2).transpose(1, 2)
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape        return x
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."lf) -> str:
        x = self.norm(x)input_resolution}, dim={self.dim}"
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)elf.input_resolution
        # x = self.proj(x).flatten(2).transpose(1, 2)        flops = H * 2 * W * 2 * self.dim // 4
        # x = self.norm(x)2) * self.dim // 4 * self.dim
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x
le):
    def extra_repr(self) -> str: norm_layer=nn.LayerNorm, use_conv=False):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"
v = use_conv
    def flops(self):        self.input_resolution = input_resolution
        H, W = self.input_resolution        self.dim = dim
        flops = H * 2 * W * 2 * self.dim // 4on
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dimm, use_conv=use_conv)
        return flops_merging2 = _PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)


class _PatchReverseMerging4x(nn.Module):
    """4x upsampling by applying patch reverse merging twice.solution
    ng1(x, H, W)
    This module applies the _PatchReverseMerging operation twice to achieve * 2)
    4x upsampling in spatial dimensions.
    
    Args:
        input_resolution (tuple[int]): Resolution of input feature map (H, W).{self.input_resolution}, dim={self.dim}"
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_conv (bool): Whether to use convolution. Default: Falseon
    """
 += (H * 2) * (W * 2) * self.dim // 4 * self.dim
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution:
        self.dim = dimd_dim=96, norm_layer=None):
        H, W = input_resolution
        self.patch_reverse_merging1 = _PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = _PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)tch_size)
 [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
    def forward(self, x, H=None, W=None):_size = img_size
        if H is None:        self.patch_size = patch_size
            H, W = self.input_resolution_resolution = patches_resolution
        x = self.patch_reverse_merging1(x, H, W)ution[0] * patches_resolution[1]
        x = self.patch_reverse_merging2(x, H * 2, W * 2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
    def flops(self):d_dim)
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4ne
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops

 and W == self.img_size[1], \
class _PatchEmbed(nn.Module):}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    """Image to Patch Embedding.).transpose(1, 2)  # B Ph*Pw C
    
    Splits an image into non-overlapping patches and embeds each patch as a token.
    
    Args:
        img_size (int | tuple(int)): Input image size. Default: 224.
        patch_size (int | tuple(int)): Patch token size. Default: 4.        Ho, Wo = self.patches_resolution
        in_chans (int): Number of input image channels. Default: 3._dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.            flops += Ho * Wo * self.embed_dim
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()class _SwinTransformerBlock(nn.Module):
        img_size = to_2tuple(img_size)ut_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]size = shift_size
atio
        self.in_chans = in_chanson) <= self.window_size:
        self.embed_dim = embed_dimput resolution, we don't partition windows
= 0
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)            self.window_size = min(self.input_resolution)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None_bias, qk_scale=qk_scale)

    def forward(self, x):yer(dim)
        B, C, H, W = x.shape        mlp_hidden_dim = int(dim * mlp_ratio)
        # assert H == self.img_size[0] and W == self.img_size[1], \ures=mlp_hidden_dim, act_layer=act_layer)
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw Ce > 0:
        if self.norm is not None:sk for SW-MSA
            x = self.norm(x)nput_resolution
        return x 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
    def flops(self): = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:                for w in w_slices:
            flops += Ho * Wo * self.embed_dimg_mask[:, h, w, :] = cnt
        return flops

ask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
class _SwinTransformerBlock(nn.Module):= mask_windows.view(-1, self.window_size * self.window_size)
    """Swin Transformer Block.            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    n_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    This is the basic building block of the Swin Transformer architecture, containing
    a window-based multi-head self-attention (W-MSA) module and a MLP block.
    
    Args:        self.register_buffer("attn_mask", attn_mask)
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        num_heads (int): Number of attention heads.
        window_size (int): Window size. Default: 7.
        shift_size (int): Shift size for SW-MSA. Default: 0.        assert L == H * W, "input feature has wrong size"
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module): Activation layer. Default: nn.GELU..view(B, H, W, C)
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """
        if self.shift_size > 0:
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):hifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_headsion windows
        self.window_size = window_size        x_windows = _window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        self.shift_size = shift_sizeew(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windowsows
            self.shift_size = 0ows = self.attn(x_windows, add_token=False, mask=self.attn_mask)
            self.window_size = min(self.input_resolution)view(-1, self.window_size, self.window_size, C)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"_x = _window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        self.norm1 = norm_layer(dim)ic shift
        self.attn = _WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
size), dims=(1, 2))
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)iew(B, H * W, C)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA        x = shortcut + x
            H, W = self.input_resolutionelf.norm2(x))
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:ndow_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt:
                    cnt += 1
on
            mask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)H * W
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)        # W-MSA/SW-MSA
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
 += self.dim * H * W
    def forward(self, x):lops
        H, W = self.input_resolution
        B, L, C = x.shape    def update_mask(self):
        assert L == H * W, "input feature has wrong size"0:

        shortcut = xput_resolution
        x = self.norm1(x) torch.zeros((1, H, W, 1))  # 1 H W 1
        x = x.view(B, H, W, C)ze), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
ice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        # cyclic shift
        if self.shift_size > 0:or h in h_slices:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x   cnt += 1

        # partition windowsask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        x_windows = _window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C   mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        B_, N, C = x_windows.shapemask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
mask.cuda()
        # merge windows
        attn_windows = self.attn(x_windows, add_token=False, mask=self.attn_mask)ass
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
le):
        # reverse cyclic shiftput_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, downsample=None):
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:olution = input_resolution
            x = shifted_xth = depth
        x = x.view(B, H * W, C)        self.blocks = nn.ModuleList(

        # FFN
        x = shortcut + x                    dim=out_dim, input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2), num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer
        x = x + self.mlp(self.norm2(x))
or i in range(depth)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
e is not None:
    def flops(self):            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)        for _, blk in enumerate(self.blocks):
        # mlp            x = blk(x)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W    def extra_repr(self) -> str:
        return flopseturn f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def update_mask(self): flops(self):
        if self.shift_size > 0:        flops = 0
            # calculate attention mask for SW-MSA
            H, W = self.input_resolutionlops()
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))            flops += self.downsample.flops()
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:    def update_resolution(self, H, W):
                for w in w_slices:, blk in enumerate(self.blocks):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1            blk.update_mask()
downsample is not None:
            mask_windows = _window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))class _AdaptiveModulator(nn.Module):
            self.attn_mask = attn_mask.cuda()    """Adaptive modulator for rate and SNR adaptation.
        else:
            pass

    """
class _BasicLayer(nn.Module):
    """A basic layer composed of multiple Swin Transformer blocks.
     super().__init__()
    This layer includes multiple Swin Transformer blocks and an optional downsampling layer.        self.fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    
    Args:rd(self, x: torch.Tensor) -> torch.Tensor:
        dim (int): Number of input channels.ator.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        depth (int): Number of blocks in this layer. shape [B, 1] representing SNR or rate values
        num_heads (int): Number of attention heads in each block.
        window_size (int): Window size in each block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.hidden_dim] with modulation values
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        downsample (nn.Module | None): Downsample layer at the end of the layer. Default: None.
    """
r(BaseModel):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, downsample=None):nt Source-Channel Coding Encoder :cite:`yang2024swinjscc`.
        super().__init__()
        self.dim = dimormer architecture with adaptive modulation capabilities for
        self.input_resolution = input_resolutiondaptation.
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                _SwinTransformerBlock(        self,
                    dim=out_dim, input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2), num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layerize: Union[int, Tuple[int, int]],
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
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:Initialize the SwinJSCC encoder.
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
        Args:
    def flops(self):ut image size (height, width)
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()            embed_dims: List of embedding dimensions for each layer
        if self.downsample is not None:r each layer
            flops += self.downsample.flops()tion heads for each layer
        return flopsount)
ttention windows
    def update_resolution(self, H, W):mension ratio
        for _, blk in enumerate(self.blocks): use bias in QKV projection
            blk.input_resolution = (H, W)or QKV attention
            blk.update_mask()            norm_layer: Normalization layer to use
        if self.downsample is not None:ter patch embedding
            self.downsample.input_resolution = (H * 2, W * 2)k dimension
r for hidden dimension in adaptation networks
umber of layers in adaptation networks
class _AdaptiveModulator(nn.Module):o use mixed precision for performance
    """Adaptive modulator for rate and SNR adaptation.efficient operations
    
    This module creates a modulation signal based on input parameters (SNR or rate)        super().__init__()
    to adaptively modify network features.

    Args:        if len(depths) != len(num_heads) or len(depths) != len(embed_dims) - 1:
        hidden_dim (int): Hidden dimension size for the modulatorths, num_heads, and embed_dims must match")
    """

    def __init__(self, hidden_dim: int):        self.patch_norm = patch_norm
        super().__init__()neck_dim
        self.fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.embed_dims = embed_dims
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the adaptive modulator.        self.patch_size = patch_size

        Args:
            x: Input tensor of shape [B, 1] representing SNR or rate values        if isinstance(img_size, int):
ize)
        Returns:
            Tensor of shape [B, hidden_dim] with modulation values
        """        self.H = img_size[0] // (2**self.num_layers)
        return self.fc(x) // (2**self.num_layers)

        # Patch embedding layer
@ModelRegistry.register_model()(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if patch_norm else None)
class Yang2024DeepJSCCSwinEncoder(BaseModel):
    """Swin Transformer-based Joint Source-Channel Coding Encoder :cite:`yang2024swinjscc`.        # Dimensions for adaptation networks
ms[-1] * adaptation_hidden_factor)
    This encoder uses a Swin Transformer architecture with adaptive modulation capabilities for
    signal-to-noise ratio (SNR) and rate adaptation.
    """        # Build transformer layers
 norm_layer)
    def __init__(
        self,mbed_dims[-1])
        img_size: Union[int, Tuple[int, int]],
        patch_size: int,ion
        in_chans: int,
        embed_dims: List[int],
        depths: List[int],
        num_heads: List[int],ild_adaptation_network()
        C: Optional[int] = None,aptation_network()
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,hts)
        qk_scale: Optional[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        bottleneck_dim: int = 16,
        adaptation_hidden_factor: float = 1.5,        # Performance optimization settings
        adaptation_layers: int = 7,
        use_mixed_precision: bool = False,memory_efficient
        memory_efficient: bool = False,
    ):(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        """Initialize the SwinJSCC encoder.        """Build the transformer layers of the encoder."""

        Args:
            img_size: Input image size (height, width)
            patch_size: Patch size for embeddings[i_layer - 1]) if i_layer != 0 else self.in_chans,
            in_chans: Number of input channels                out_dim=int(embed_dims[i_layer]),
            embed_dims: List of embedding dimensions for each layer(2**i_layer), self.patches_resolution[1] // (2**i_layer)),
            depths: List of depths for each layer
            num_heads: List of attention heads for each layer                num_heads=num_heads[i_layer],
            C: Output dimension (channel count)ze,
            window_size: Size of attention windowso,
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QKV attention
            norm_layer: Normalization layer to use                downsample=_PatchMerging if i_layer != 0 else None,
            patch_norm: Whether to use normalization after patch embedding
            bottleneck_dim: Bottleneck dimension
            adaptation_hidden_factor: Factor for hidden dimension in adaptation networks
            adaptation_layers: Number of layers in adaptation networks
            use_mixed_precision: Whether to use mixed precision for performance
            memory_efficient: Whether to use memory-efficient operationsn layers
        """
        super().__init__()
""
        # Validate inputs
        if len(depths) != len(num_heads) or len(depths) != len(embed_dims) - 1:
            raise ValueError("Lengths of depths, num_heads, and embed_dims must match")
dden dim
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim        # Build layers for adaptation
        self.mlp_ratio = mlp_ratio range(self.layer_num):
        self.embed_dims = embed_dims
        self.in_chans = in_chans modulators.append(_AdaptiveModulator(self.hidden_dim))
        self.patch_size = patch_size.Linear(self.hidden_dim, out_dim))

        # Convert scalar dimensions to tuples if needed, "linears": linears, "sigmoid": nn.Sigmoid()}
        if isinstance(img_size, int):
            img_size = (img_size, img_size)loat] = None, rate: Optional[int] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        self.img_size = img_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2**self.num_layers)
        self.W = img_size[1] // (2**self.num_layers)H, W]
NR adaptation
        # Patch embedding layer
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer if patch_norm else None)            model_mode: Model variant to use:
l without adaptations
        # Dimensions for adaptation networksw/_SA': With SNR adaptation
        self.hidden_dim = int(self.embed_dims[-1] * adaptation_hidden_factor)aptation
        self.layer_num = adaptation_layers
            return_intermediate_features: Whether to return intermediate feature maps
        # Build transformer layers
        self.layers = self._build_layers(embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer)
on and/or intermediate features
        self.norm = norm_layer(embed_dims[-1])        """

        # Channel output projection
        self.head = nn.Linear(embed_dims[-1], C) if C is not None else None        # Use mixed precision if enabled
ixed_precision and torch.cuda.is_available() else nullcontext():
        # Initialize adaptation modules if requested
        self.snr_adaptation = self._build_adaptation_network(){} if return_intermediate_features else None
        self.rate_adaptation = self._build_adaptation_network()
            # Patch embedding and feature extraction
        # Initialize weights
        self.apply(self._init_weights)ate_features:

        # Logger for warnings and info
        self.logger = logging.getLogger(__name__)            for i, layer in enumerate(self.layers):

        # Performance optimization settingsediate_features:
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient
            x = self.norm(x)
    def _build_layers(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        """Build the transformer layers of the encoder.""" x.detach().clone()
        layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = _BasicLayer(            result = None
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else self.in_chans,
                out_dim=int(embed_dims[i_layer]),
                input_resolution=(self.patches_resolution[0] // (2**i_layer), self.patches_resolution[1] // (2**i_layer)),                if self.head is not None:
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,f snr is None:
                qk_scale=qk_scale,ovided for SNR adaptation mode")
                norm_layer=norm_layer,                result = self._apply_snr_adaptation(x, snr, B, H, W)
                downsample=_PatchMerging if i_layer != 0 else None,
            )            elif model_mode == "SwinJSCC_w/_RA":

            self.logger.info(f"Encoder layer {i_layer}: {layer.extra_repr()}")ate adaptation mode")
            layers.append(layer)self._apply_rate_adaptation(x, rate, B, H, W)

        return layers
                if snr is None or rate is None:
    def _build_adaptation_network(self):rate must be provided for joint adaptation mode")
        """Build an adaptation network (SNR or rate)"""self._apply_snr_and_rate_adaptation(x, snr, rate, B, H, W)
        modulators = nn.ModuleList()
        linears = nn.ModuleList()
 model mode: {model_mode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'")
        # First linear projects from embedding dim to hidden dim
        linears.append(nn.Linear(self.embed_dims[-1], self.hidden_dim))        # Handle intermediate features
rmediate_features:
        # Build layers for adaptationt, tuple):
        for i in range(self.layer_num):                return (*result, intermediate_features)
            out_dim = self.embed_dims[-1] if i == self.layer_num - 1 else self.hidden_dim
            modulators.append(_AdaptiveModulator(self.hidden_dim))iate_features
            linears.append(nn.Linear(self.hidden_dim, out_dim))

        return {"modulators": modulators, "linears": linears, "sigmoid": nn.Sigmoid()}
, x: torch.Tensor, value: float, network: Dict, batch_size: int, spatial_dim: int) -> torch.Tensor:
    def forward(self, x: torch.Tensor, snr: Optional[float] = None, rate: Optional[int] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass through the encoder.
        value_tensor = torch.tensor(value, dtype=torch.float).to(device)
        Args:r.unsqueeze(0).expand(batch_size, -1)
            x: Input images [B, C, H, W]
            snr: Signal-to-noise ratio for SNR adaptation        # Use detached features for adaptation path
            rate: Target bit rate for rate adaptation
            model_mode: Model variant to use:
                - 'SwinJSCC_w/o_SAandRA': Base model without adaptations        # Apply adaptation network
                - 'SwinJSCC_w/_SA': With SNR adaptation
                - 'SwinJSCC_w/_RA': With rate adaptation
                - 'SwinJSCC_w/_SAandRA': With both SNR and rate adaptation
            return_intermediate_features: Whether to return intermediate feature maps
        temp = linears[0](temp)
        Returns:
            Output features, and optionally mask for rate adaptation and/or intermediate featuresyers
        """ range(self.layer_num):
        B, C, H, W = x.size()            if i > 0:

        # Use mixed precision if enabled
        with torch.cuda.amp.autocast() if self.use_mixed_precision and torch.cuda.is_available() else nullcontext():
            # Store intermediate features if requested
            intermediate_features = {} if return_intermediate_features else None            temp = temp * bm

            # Patch embedding and feature extraction
            x = self.patch_embed(x)        mod_val = network["sigmoid"](linears[-1](temp))
            if return_intermediate_features:
                intermediate_features["patch_embed"] = x.detach().clone()lation to features
_val, mod_val
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if return_intermediate_features:
                    intermediate_features[f"layer_{i}"] = x.detach().clone() W // (4**self.num_layers)
        x, _ = self._apply_adaptation(x, snr, self.snr_adaptation, batch_size, spatial_dim)
            x = self.norm(x)
            if return_intermediate_features:
                intermediate_features["norm"] = x.detach().clone()            x = self.head(x)

            # Apply the specified adaptation strategy
            result = Noneh.Tensor, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rate adaptation to features."""
            if model_mode == "SwinJSCC_w/o_SAandRA":
                if self.head is not None:
                    x = self.head(x)
                result = x        # Generate channel selection mask based on rate
mod_val, rate, batch_size, spatial_dim)
            elif model_mode == "SwinJSCC_w/_SA":
                if snr is None:
                    raise ValueError("SNR must be provided for SNR adaptation mode")        x = x * mask
                result = self._apply_snr_adaptation(x, snr, B, H, W)

            elif model_mode == "SwinJSCC_w/_RA":torch.Tensor, rate: int, batch_size: int, spatial_dim: int) -> torch.Tensor:
                if rate is None:        """Generate mask for rate adaptation based on modulation values."""
                    raise ValueError("Rate must be provided for rate adaptation mode")device
                result = self._apply_rate_adaptation(x, rate, B, H, W)

            elif model_mode == "SwinJSCC_w/_SAandRA":
                if snr is None or rate is None:
                    raise ValueError("Both SNR and rate must be provided for joint adaptation mode")        # Select top channels
                result = self._apply_snr_and_rate_adaptation(x, snr, rate, B, H, W)nce.sort(dim=1, descending=True)

            else:
                raise ValueError(f"Unknown model mode: {model_mode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'")lattened indexing
(1).repeat(1, rate)
        # Handle intermediate features        flattened_indices = selected_indices + batch_offsets.int()
        if return_intermediate_features:
            if isinstance(result, tuple):
                return (*result, intermediate_features)torch.zeros(channel_importance.size(), device=device).reshape(-1)
            else:        flat_mask[flattened_indices.reshape(-1)] = 1
                return result, intermediate_features
        # Reshape mask to match feature dimensions
        return resulte(batch_size, x.size(2))
dim, -1)
    def _apply_adaptation(self, x: torch.Tensor, value: float, network: Dict, batch_size: int, spatial_dim: int) -> torch.Tensor:
        """Common adaptation logic for SNR or rate adaptation."""
        device = x.device
        value_tensor = torch.tensor(value, dtype=torch.float).to(device) x: torch.Tensor, snr: float, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        value_batch = value_tensor.unsqueeze(0).expand(batch_size, -1)tion to features."""
num_layers)
        # Use detached features for adaptation path
        temp = x.detach()        # First apply SNR adaptation
lf.snr_adaptation, batch_size, spatial_dim)
        # Apply adaptation network
        modulators = network["modulators"]
        linears = network["linears"]self.rate_adaptation, batch_size, spatial_dim)

        # First linear        # Generate and apply mask
        temp = linears[0](temp)te_rate_mask(x, mod_val, rate, batch_size, spatial_dim)

        # Process through layers
        for i in range(self.layer_num):
            if i > 0:
                temp = linears[i](temp)

            # Apply modulatione(m, nn.Linear):
            bm = modulators[i](value_batch).unsqueeze(1).expand(-1, spatial_dim, -1)            trunc_normal_(m.weight, std=0.02)
            temp = temp * bm is not None:
nt_(m.bias, 0)
        # Final modulation values
        mod_val = network["sigmoid"](linears[-1](temp)) 0)
            nn.init.constant_(m.weight, 1.0)
        # Apply modulation to features
        return x * mod_val, mod_val W: int):

    def _apply_snr_adaptation(self, x: torch.Tensor, snr: float, batch_size: int, H: int, W: int) -> torch.Tensor:
        """Apply SNR adaptation to features."""        for i_layer, layer in enumerate(self.layers):
        spatial_dim = H * W // (4**self.num_layers) W // (2 ** (i_layer + 1)))
        x, _ = self._apply_adaptation(x, snr, self.snr_adaptation, batch_size, spatial_dim)

        if self.head is not None:te the floating point operations for the model."""
            x = self.head(x)        flops = 0
        return xatch_embed.flops()

    def _apply_rate_adaptation(self, x: torch.Tensor, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rate adaptation to features."""        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
        spatial_dim = H * W // (4**self.num_layers)
        x, mod_val = self._apply_adaptation(x, rate, self.rate_adaptation, batch_size, spatial_dim)

        # Generate channel selection mask based on rate
        mask = self._generate_rate_mask(x, mod_val, rate, batch_size, spatial_dim)""

        # Apply mask
        x = x * mask
        return x, mask    def no_weight_decay_keywords(self):
        """Return parameter keywords that shouldn't use weight decay."""
    def _generate_rate_mask(self, x: torch.Tensor, mod_val: torch.Tensor, rate: int, batch_size: int, spatial_dim: int) -> torch.Tensor:ion_bias_table"}
        """Generate mask for rate adaptation based on modulation values."""
        device = x.device
        """Enable gradient checkpointing for memory efficiency."""
        # Calculate importance of each channel
        channel_importance = torch.sum(mod_val, dim=1)

        # Select top channels def fn(module):
        _, indices = channel_importance.sort(dim=1, descending=True)            if hasattr(module, "gradient_checkpointing"):
        selected_indices = indices[:, :rate]odule.gradient_checkpointing = enable

        # Create batch offsets for flattened indexing
        batch_offsets = torch.arange(0, batch_size * x.size(2), x.size(2), device=device).unsqueeze(1).repeat(1, rate)
        flattened_indices = selected_indices + batch_offsets.int()f) -> Dict[str, Union[int, float]]:
information."""
        # Create mask from selected indices() for p in self.parameters())
        flat_mask = torch.zeros(channel_importance.size(), device=device).reshape(-1).numel() for p in self.parameters() if p.requires_grad)
        flat_mask[flattened_indices.reshape(-1)] = 1
: num_params, "trainable_params": num_trainable, "param_size_mb": num_params * 4 / 1024 / 1024, "flops_g": self.flops() / 1e9}  # Assuming float32
        # Reshape mask to match feature dimensions
        mask = flat_mask.reshape(batch_size, x.size(2))
        mask = mask.unsqueeze(1).expand(-1, spatial_dim, -1)del()
r(BaseModel):
        return masknt Source-Channel Coding Decoder :cite:`yang2024swinjscc`.

    def _apply_snr_and_rate_adaptation(self, x: torch.Tensor, snr: float, rate: int, batch_size: int, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:representation from Yang2024SwinJSCCEncoder and reconstructs the
        """Apply both SNR and rate adaptation to features."""tion for robust performance across varying channel
        spatial_dim = H * W // (4**self.num_layers)
"
        # First apply SNR adaptation
        x, _ = self._apply_adaptation(x, snr, self.snr_adaptation, batch_size, spatial_dim)    def __init__(

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
SwinJSCC decoder.
    def update_resolution(self, H: int, W: int):
        """Update the resolution of internal components."""
        self.patches_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)), W // (2 ** (i_layer + 1)))            depths: List of depths for each layer
 layer
    def flops(self) -> int:ncoder
        """Calculate the floating point operations for the model."""windows
        flops = 0            mlp_ratio: MLP hidden dimension ratio
        flops += self.patch_embed.flops()bias in QKV projection
        for layer in self.layers:Scale factor for QKV attention
            flops += layer.flops()n layer to use
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)lute position embedding
        return flopsmalization
neck dimension
    @torch.jit.ignorector: Factor for hidden dimension in adaptation networks
    def no_weight_decay(self):s: Number of layers in adaptation networks
        """Return parameters that shouldn't use weight decay."""ion: Whether to use mixed precision for performance
        return {"absolute_pos_embed"}
        """
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Return parameter keywords that shouldn't use weight decay."""
        return {"relative_position_bias_table"}
and embed_dims must match")
    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing for memory efficiency.""" needed
        if not self.memory_efficient:
            return            img_size = (img_size, img_size)

        def fn(module):
            if hasattr(module, "gradient_checkpointing"):        self.ape = ape
                module.gradient_checkpointing = enable

        self.apply(fn)

    def get_model_size(self) -> Dict[str, Union[int, float]]:        self.img_size = img_size
        """Return model size information."""
        num_params = sum(p.numel() for p in self.parameters())        self.W = img_size[1]
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))

        return {"total_params": num_params, "trainable_params": num_trainable, "param_size_mb": num_params * 4 / 1024 / 1024, "flops_g": self.flops() / 1e9}  # Assuming float32        # Positional embedding if enabled


@ModelRegistry.register_model()er(torch.zeros(1, num_patches, embed_dims[0]))
class Yang2024DeepJSCCSwinDecoder(BaseModel):            trunc_normal_(self.absolute_pos_embed, std=0.02)
    """Swin Transformer-based Joint Source-Channel Coding Decoder :cite:`yang2024swinjscc`.

    This decoder takes the encoded representation from Yang2024SwinJSCCEncoder and reconstructs the embed_dims[0]) if C is not None else None
    original image. It supports SNR adaptation for robust performance across varying channel
    conditions.
    """d_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer)

    def __init__(
        self,ms[0] * adaptation_hidden_factor)
        img_size: Union[int, Tuple[int, int]],
        embed_dims: List[int],d_adaptation_network()
        depths: List[int],
        num_heads: List[int],hts)
        C: Optional[int] = None,
        window_size: int = 4,
        mlp_ratio: float = 4.0,__)
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,        # Performance optimization settings
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,memory_efficient
        patch_norm: bool = True,
        bottleneck_dim: int = 16,(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        adaptation_hidden_factor: float = 1.5,        """Build the transformer layers of the decoder."""
        adaptation_layers: int = 7,
        use_mixed_precision: bool = False,
        memory_efficient: bool = False,
    ):s[i_layer]),
        """Initialize the SwinJSCC decoder.                out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
2**i_layer), self.patches_resolution[1] * (2**i_layer)),
        Args:
            img_size: Output image size (height, width)                num_heads=num_heads[i_layer],
            embed_dims: List of embedding dimensions for each layerze,
            depths: List of depths for each layero,
            num_heads: List of attention heads for each layer
            C: Input dimension from encoder
            window_size: Size of attention windows
            mlp_ratio: MLP hidden dimension ratio                upsample=_PatchReverseMerging,
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for QKV attention
            norm_layer: Normalization layer to use
            ape: Whether to use absolute position embedding
            patch_norm: Whether to use normalization
            bottleneck_dim: Bottleneck dimensionn layers
            adaptation_hidden_factor: Factor for hidden dimension in adaptation networks
            adaptation_layers: Number of layers in adaptation networks
            use_mixed_precision: Whether to use mixed precision for performance"
            memory_efficient: Whether to use memory-efficient operations
        """
        super().__init__()

        # Validate inputs
        if len(depths) != len(num_heads) or len(depths) != len(embed_dims) - 1:
            raise ValueError("Lengths of depths, num_heads, and embed_dims must match")layers for adaptation

        # Convert scalar dimensions to tuples if needed out_dim = self.embed_dims[0] if i == self.layer_num - 1 else self.hidden_dim
        if isinstance(img_size, int):odulator(self.hidden_dim))
            img_size = (img_size, img_size)

        self.num_layers = len(depths)oid()}
        self.ape = ape
        self.embed_dims = embed_dims[float] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.H = img_size[0]e ratio for SNR adaptation
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
 'SwinJSCC_w/_SA': With SNR adaptation
        # Positional embedding if enabled
        if self.ape:                - 'SwinJSCC_w/_SAandRA': With both SNR and rate adaptation
            num_patches = self.H // 4 * self.W // 4: Whether to return intermediate feature maps
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
            Output images [B, 3, H, W] and optionally intermediate features
        # Input projection from encoder channels
        self.head = nn.Linear(C, embed_dims[0]) if C is not None else None
ast() if self.use_mixed_precision and torch.cuda.is_available() else nullcontext():
        # Build layersiate features if requested
        self.layers = self._build_layers(embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer)intermediate_features else None

        # Initialize SNR adaptation modules            if model_mode == "SwinJSCC_w/o_SAandRA":
        self.hidden_dim = int(embed_dims[0] * adaptation_hidden_factor)orward_base(x, intermediate_features)
        self.layer_num = adaptation_layers:
        self.snr_adaptation = self._build_adaptation_network() self._forward_base(x, intermediate_features)  # No SNR adaptation for rate-only mode
r model_mode == "SwinJSCC_w/_SAandRA":
        self.apply(self._init_weights)
                    raise ValueError("SNR must be provided for SNR adaptation mode")
        # Logger for warnings and inforward_with_snr(x, snr, intermediate_features)
        self.logger = logging.getLogger(__name__)
ode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'")
        # Performance optimization settings
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient            return result, intermediate_features
esult
    def _build_layers(self, embed_dims, depths, num_heads, window_size, qkv_bias, qk_scale, norm_layer):
        """Build the transformer layers of the decoder."""
        layers = nn.ModuleList()."""
        for i_layer in range(self.num_layers):t None:
            layer = _BasicLayer((x)
                dim=int(embed_dims[i_layer]),            if intermediate_features is not None:
                out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,] = x.detach().clone()
                input_resolution=(self.patches_resolution[0] * (2**i_layer), self.patches_resolution[1] * (2**i_layer)),
                depth=depths[i_layer],yers
                num_heads=num_heads[i_layer],        for i, layer in enumerate(self.layers):
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,etach().clone()
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                upsample=_PatchReverseMerging,
            )).permute(0, 3, 1, 2)
not None:
            self.logger.info(f"Decoder layer {i_layer}: {layer.extra_repr()}")            intermediate_features["output"] = x.detach().clone()
            layers.append(layer)

        return layers
snr(self, x: torch.Tensor, snr: float, intermediate_features: Optional[Dict] = None) -> torch.Tensor:
    def _build_adaptation_network(self):R adaptation."""
        """Build an SNR adaptation network."""        B, L, C = x.size()
        modulators = nn.ModuleList()
        linears = nn.ModuleList()
projection if available
        # First linear projects from embedding dim to hidden dim
        linears.append(nn.Linear(self.embed_dims[0], self.hidden_dim))            x = self.head(x)

        # Build layers for adaptation
        for i in range(self.layer_num):ensor(snr, dtype=torch.float).to(device)
            out_dim = self.embed_dims[0] if i == self.layer_num - 1 else self.hidden_dim        snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)
            modulators.append(_AdaptiveModulator(self.hidden_dim))
            linears.append(nn.Linear(self.hidden_dim, out_dim))
snr_adaptation
        return {"modulators": modulators, "linears": linears, "sigmoid": nn.Sigmoid()}        modulators = network["modulators"]

    def forward(self, x: torch.Tensor, snr: Optional[float] = None, model_mode: str = "SwinJSCC_w/o_SAandRA", return_intermediate_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through the decoder.eatures for adaptation path
        temp = x.detach()
        Args:
            x: Input features from encoder
            snr: Signal-to-noise ratio for SNR adaptation
            model_mode: Model variant to use:
                - 'SwinJSCC_w/o_SAandRA': Base model without adaptations        # Process through layers
                - 'SwinJSCC_w/_SA': With SNR adaptationer_num):
                - 'SwinJSCC_w/_RA': With rate adaptation (no SNR adaptation needed)
                - 'SwinJSCC_w/_SAandRA': With both SNR and rate adaptationmp)
            return_intermediate_features: Whether to return intermediate feature maps

        Returns:squeeze(1).expand(-1, L, -1)
            Output images [B, 3, H, W] and optionally intermediate features
        """
        # Use mixed precision if enabled
        with torch.cuda.amp.autocast() if self.use_mixed_precision and torch.cuda.is_available() else nullcontext():        mod_val = network["sigmoid"](linears[-1](temp))
            # Store intermediate features if requested
            intermediate_features = {} if return_intermediate_features else None

            if model_mode == "SwinJSCC_w/o_SAandRA":
                result = self._forward_base(x, intermediate_features)
            elif model_mode == "SwinJSCC_w/_RA":
                result = self._forward_base(x, intermediate_features)  # No SNR adaptation for rate-only mode
            elif model_mode == "SwinJSCC_w/_SA" or model_mode == "SwinJSCC_w/_SAandRA":
                if snr is None:        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
                    raise ValueError("SNR must be provided for SNR adaptation mode")
                result = self._forward_with_snr(x, snr, intermediate_features)
            else:ghts(self, m):
                raise ValueError(f"Unknown model mode: {model_mode}. Valid modes are: " "'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'") network layers."""
):
        if return_intermediate_features:rmal_(m.weight, std=0.02)
            return result, intermediate_features            if m.bias is not None:
        return resultit.constant_(m.bias, 0)
LayerNorm):
    def _forward_base(self, x: torch.Tensor, intermediate_features: Optional[Dict] = None) -> torch.Tensor:
        """Base forward pass without adaptation."""t, 1.0)
        if self.head is not None:
            x = self.head(x)tion(self, H: int, W: int):
            if intermediate_features is not None:ernal components."""
                intermediate_features["head"] = x.detach().clone()

        # Process through layers        self.W = W * 2 ** len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if intermediate_features is not None: * (2**i_layer), W * (2**i_layer))
                intermediate_features[f"layer_{i}"] = x.detach().clone()
    def flops(self) -> int:
        # Reshape to image formate floating point operations for the model."""
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        if intermediate_features is not None:            flops += layer.flops()
            intermediate_features["output"] = x.detach().clone()

        return x

    def _forward_with_snr(self, x: torch.Tensor, snr: float, intermediate_features: Optional[Dict] = None) -> torch.Tensor:""
        """Forward pass with SNR adaptation."""
        B, L, C = x.size()
        device = x.device
    def no_weight_decay_keywords(self):
        # Apply head projection if available        """Return parameter keywords that shouldn't use weight decay."""
        if self.head is not None:elative_position_bias_table"}
            x = self.head(x)
    def set_gradient_checkpointing(self, enable: bool = True):
        # Apply SNR adaptation checkpointing for memory efficiency."""
        snr_tensor = torch.tensor(snr, dtype=torch.float).to(device) if not self.memory_efficient:
        snr_batch = snr_tensor.unsqueeze(0).expand(B, -1)            return

        # Use the adaptation networkfn(module):
        network = self.snr_adaptation            if hasattr(module, "gradient_checkpointing"):
        modulators = network["modulators"]able
        linears = network["linears"]
        self.apply(fn)
        # Use detached features for adaptation path
        temp = x.detach()e(self) -> Dict[str, Union[int, float]]:

        # First linear        num_params = sum(p.numel() for p in self.parameters())
        temp = linears[0](temp)able = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Process through layersle_params": num_trainable, "param_size_mb": num_params * 4 / 1024 / 1024, "flops_g": self.flops() / 1e9}  # Assuming float32
        for i in range(self.layer_num):
            if i > 0:
                temp = linears[i](temp)
g.
            # Apply modulation
            bm = modulators[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
            temp = temp * bm

        # Apply final modulation to features
        mod_val = network["sigmoid"](linears[-1](temp))
        x = x * mod_val
_type, exc_val, exc_tb):
        # Process through decoder layers
        for layer in self.layers:
            x = layer(x)

        # Reshape to image format"Configuration object for SwinJSCC encoder and decoder."""
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)    def __init__(
        return x
 = 224,
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
            layer.update_resolution(H * (2**i_layer), W * (2**i_layer))ion for SwinJSCC models.

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

    Args:
        config: Configuration object for the models
        channel_dim: Dimension of the bottleneck/channel
        device: Device to create the models on

    Returns:
        Encoder and decoder models
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
    """Create a Swin JSCC encoder model with the given parameters."""
    return Yang2024DeepJSCCSwinEncoder(**kwargs)


def create_decoder(**kwargs) -> Yang2024DeepJSCCSwinDecoder:
    """Create a Swin JSCC decoder model with the given parameters."""
    return Yang2024DeepJSCCSwinDecoder(**kwargs)


def build_model(config):
    """Build and initialize SwinJSCC models based on config. Validates the model by running a test
    input and reports stats.

    Args:
        config: Configuration object with encoder_kwargs and device

    Returns:
        The built encoder model
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
