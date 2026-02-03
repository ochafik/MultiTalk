# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of wan/modules/vae.py
import logging
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    'WanVAE',
]

CACHE_T = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ncdhw_to_ndhwc(x):
    """(B,C,D,H,W) -> (B,D,H,W,C)"""
    return mx.transpose(x, (0, 2, 3, 4, 1))


def _ndhwc_to_ncdhw(x):
    """(B,D,H,W,C) -> (B,C,D,H,W)"""
    return mx.transpose(x, (0, 4, 1, 2, 3))


def _nchw_to_nhwc(x):
    """(B,C,H,W) -> (B,H,W,C)"""
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x):
    """(B,H,W,C) -> (B,C,H,W)"""
    return mx.transpose(x, (0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """
    Causal 3d convolution.
    Data stays in NCDHW format externally; we transpose to NDHWC for the
    underlying MLX Conv3d, then transpose back.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        # Store the original padding for the causal padding logic
        # PyTorch padding order: (D, H, W)
        self._causal_pad_d_front = 2 * padding[0]  # causal: all padding goes to front
        self._causal_pad_d_back = 0
        self._causal_pad_h = padding[1]
        self._causal_pad_w = padding[2]

        # The actual conv uses no padding -- we pad manually
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, bias=bias)

    def __call__(self, x, cache_x=None):
        # x: (B, C, D, H, W)  -- NCDHW
        padding_d_front = self._causal_pad_d_front
        padding_d_back = self._causal_pad_d_back

        if cache_x is not None and self._causal_pad_d_front > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            padding_d_front -= cache_x.shape[2]
            padding_d_front = max(padding_d_front, 0)

        # Apply padding: mx.pad with pad_width per dim
        # x is NCDHW = dims (B, C, D, H, W)
        pad_width = [
            (0, 0),  # B
            (0, 0),  # C
            (padding_d_front, padding_d_back),  # D
            (self._causal_pad_h, self._causal_pad_h),  # H
            (self._causal_pad_w, self._causal_pad_w),  # W
        ]
        if any(p != (0, 0) for p in pad_width):
            x = mx.pad(x, pad_width)

        # Transpose to NDHWC for MLX Conv3d
        x = _ncdhw_to_ndhwc(x)
        x = self.conv(x)
        x = _ndhwc_to_ncdhw(x)
        return x


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = mx.ones(shape)
        self._has_bias = bias
        if bias:
            self.bias_param = mx.zeros(shape)

    def __call__(self, x):
        dim = 1 if self.channel_first else -1
        # F.normalize = x / ||x||_2  (L2 normalization along dim)
        norm = mx.sqrt(mx.sum(x * x, axis=dim, keepdims=True) + 1e-12)
        x_normed = x / norm
        bias = self.bias_param if self._has_bias else 0.0
        return x_normed * self.scale * self.gamma + bias


class Upsample2d(nn.Module):
    """Nearest-neighbor 2x upsampling for 2D (NCHW) tensors."""

    def __call__(self, x):
        # x: (B, C, H, W) -- NCHW
        # repeat along H and W
        # We use reshape tricks: (B, C, H, 1, W, 1) -> repeat -> (B, C, 2H, 2W)
        B, C, H, W = x.shape
        x = mx.expand_dims(x, (3, 5))          # (B, C, H, 1, W, 1)
        x = mx.repeat(x, repeats=2, axis=3)    # (B, C, H, 2, W, 1)
        x = mx.repeat(x, repeats=2, axis=5)    # (B, C, H, 2, W, 2)
        x = x.reshape(B, C, H * 2, W * 2)
        return x


class ZeroPad2d_0101(nn.Module):
    """Equivalent to nn.ZeroPad2d((0,1,0,1)) -- pad right=1, bottom=1 on NCHW."""

    def __call__(self, x):
        # x: (B, C, H, W)
        return mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1)])


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                         'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.upsample = Upsample2d()
            self.conv2d = nn.Conv2d(dim, dim // 2, 3, padding=1)
        elif mode == 'upsample3d':
            self.upsample = Upsample2d()
            self.conv2d = nn.Conv2d(dim, dim // 2, 3, padding=1)
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.zero_pad = ZeroPad2d_0101()
            self.conv2d = nn.Conv2d(dim, dim, 3, stride=(2, 2))
        elif mode == 'downsample3d':
            self.zero_pad = ZeroPad2d_0101()
            self.conv2d = nn.Conv2d(dim, dim, 3, stride=(2, 2))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        self._is_none = (mode == 'none')

    def _apply_resample_2d(self, x):
        """Apply the 2D resample path. x is NCHW."""
        if self.mode in ('upsample2d', 'upsample3d'):
            x = self.upsample(x)
            # Conv2d: NCHW -> NHWC -> conv -> NCHW
            x = _nchw_to_nhwc(x)
            x = self.conv2d(x)
            x = _nhwc_to_nchw(x)
        elif self.mode in ('downsample2d', 'downsample3d'):
            x = self.zero_pad(x)
            x = _nchw_to_nhwc(x)
            x = self.conv2d(x)
            x = _nhwc_to_nchw(x)
        return x

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.shape

        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = mx.concatenate([
                            mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                            cache_x
                        ], axis=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = mx.concatenate([
                            mx.zeros_like(cache_x),
                            cache_x
                        ], axis=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    # interleave: stack along time
                    x0 = x[:, 0, :, :, :, :]  # (b, c, t, h, w)
                    x1 = x[:, 1, :, :, :, :]
                    # Stack and interleave: (b, c, 2t, h, w)
                    x = mx.stack([x0, x1], axis=3)  # (b, c, t, 2, h, w)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        # rearrange: (b, c, t, h, w) -> (b*t, c, h, w)
        x = x.reshape(b * t, c, h, w)

        if self._is_none:
            pass  # identity
        else:
            x = self._apply_resample_2d(x)

        # rearrange back: (b*t, c', h', w') -> (b, c', t, h', w')
        c_new = x.shape[1]
        h_new = x.shape[2]
        w_new = x.shape[3]
        x = x.reshape(b, t, c_new, h_new, w_new)
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, c', t, h', w')

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(
                        mx.concatenate([feat_cache[idx][:, :, -1:, :, :], x], axis=2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Residual path layers (stored as list for iteration)
        self.norm1 = RMS_norm(in_dim, images=False)
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        self.act2 = nn.SiLU()
        # dropout removed for inference
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)

        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]

        if self.shortcut is not None:
            h = self.shortcut(x)
        else:
            h = x

        # Residual path - iterate through layers, handling CausalConv3d caching
        # Layer order: norm1, act1, conv1, norm2, act2, conv2
        residual_layers = [
            ('norm', self.norm1),
            ('act', self.act1),
            ('conv', self.conv1),
            ('norm', self.norm2),
            ('act', self.act2),
            ('conv', self.conv2),
        ]

        for kind, layer in residual_layers:
            if kind == 'conv' and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                        cache_x
                    ], axis=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        # Conv2d for qkv projection and output projection
        # These operate on NCHW data, we transpose for MLX
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # Note: proj weights should be zero-initialized when loading from checkpoint

    def __call__(self, x, feat_cache=None, feat_idx=None):
        # For compatibility with the iteration pattern in Encoder/Decoder
        if feat_idx is None:
            feat_idx = [0]

        identity = x
        b, c, t, h, w = x.shape
        # (b, c, t, h, w) -> (b*t, c, h, w)
        x = x.reshape(b * t, c, h, w)
        x = self.norm(x)

        # Conv2d: need NHWC for MLX
        x_nhwc = _nchw_to_nhwc(x)   # (b*t, h, w, c)
        qkv = self.to_qkv(x_nhwc)   # (b*t, h, w, 3c)
        qkv = _nhwc_to_nchw(qkv)    # (b*t, 3c, h, w)

        # Reshape to (b*t, 1, h*w, 3c) then split
        qkv = qkv.reshape(b * t, 1, c * 3, -1)          # (b*t, 1, 3c, h*w)
        qkv = mx.transpose(qkv, (0, 1, 3, 2))           # (b*t, 1, h*w, 3c)
        q, k, v = mx.split(qkv, 3, axis=-1)             # each (b*t, 1, h*w, c)

        # scaled_dot_product_attention: (B, n_heads, seq_len, head_dim)
        x = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / (c ** 0.5))

        # (b*t, 1, h*w, c) -> (b*t, c, h, w)
        x = x.squeeze(axis=1)                            # (b*t, h*w, c)
        x = mx.transpose(x, (0, 2, 1))                  # (b*t, c, h*w)
        x = x.reshape(b * t, c, h, w)

        # Output projection
        x_nhwc = _nchw_to_nhwc(x)
        x = self.proj(x_nhwc)
        x = _nhwc_to_nchw(x)

        # Back to 5D
        x = x.reshape(b, t, c, h, w)
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, c, t, h, w)
        return x + identity


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = downsamples

        # middle blocks
        self.mid_res1 = ResidualBlock(out_dim, out_dim, dropout)
        self.mid_attn = AttentionBlock(out_dim)
        self.mid_res2 = ResidualBlock(out_dim, out_dim, dropout)

        # output blocks
        self.head_norm = RMS_norm(out_dim, images=False)
        self.head_act = nn.SiLU()
        self.head_conv = CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                    cache_x
                ], axis=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in [self.mid_res1, self.mid_attn, self.mid_res2]:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for kind, layer in [('norm', self.head_norm), ('act', self.head_act), ('conv', self.head_conv)]:
            if kind == 'conv' and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                        cache_x
                    ], axis=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_res1 = ResidualBlock(dims[0], dims[0], dropout)
        self.mid_attn = AttentionBlock(dims[0])
        self.mid_res2 = ResidualBlock(dims[0], dims[0], dropout)

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = upsamples

        # output blocks
        self.head_norm = RMS_norm(out_dim, images=False)
        self.head_act = nn.SiLU()
        self.head_conv = CausalConv3d(out_dim, 3, 3, padding=1)

    def __call__(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]

        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate([
                    mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                    cache_x
                ], axis=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in [self.mid_res1, self.mid_attn, self.mid_res2]:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for kind, layer in [('norm', self.head_norm), ('act', self.head_act), ('conv', self.head_conv)]:
            if kind == 'conv' and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate([
                        mx.expand_dims(feat_cache[idx][:, :, -1, :, :], axis=2),
                        cache_x
                    ], axis=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Counting helper
# ---------------------------------------------------------------------------


def _count_conv3d_recursive(obj, visited=None):
    """Recursively count CausalConv3d instances in a module tree.

    Handles nn.Module (with children()), lists, and dicts as returned
    by MLX's children() method.
    Note: MLX nn.Module inherits from dict, so we must check nn.Module first.
    """
    if visited is None:
        visited = set()

    if isinstance(obj, nn.Module):
        obj_id = id(obj)
        if obj_id in visited:
            return 0
        visited.add(obj_id)
        count = 1 if isinstance(obj, CausalConv3d) else 0
        count += _count_conv3d_recursive(obj.children(), visited)
        return count
    elif isinstance(obj, dict):
        count = 0
        for v in obj.values():
            count += _count_conv3d_recursive(v, visited)
        return count
    elif isinstance(obj, list):
        count = 0
        for item in obj:
            count += _count_conv3d_recursive(item, visited)
        return count
    return 0


# ---------------------------------------------------------------------------
# Top-level VAE model
# ---------------------------------------------------------------------------

class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def __call__(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = mx.concatenate([out, out_], axis=2)

        mu, log_var = mx.split(self.conv1(out), 2, axis=1)

        if isinstance(scale[0], mx.array) and scale[0].ndim > 0:
            mu = (mu - scale[0].reshape(1, self.z_dim, 1, 1, 1)) * scale[1].reshape(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], mx.array) and scale[0].ndim > 0:
            z = z / scale[1].reshape(1, self.z_dim, 1, 1, 1) + scale[0].reshape(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        out = None
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = mx.concatenate([out, out_], axis=2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(std.shape)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = mx.exp(0.5 * mx.clip(log_var, -30.0, 20.0))
        return mu + std * mx.random.normal(std.shape)

    def clear_cache(self):
        self._conv_num = _count_conv3d_recursive(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = _count_conv3d_recursive(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


# ---------------------------------------------------------------------------
# Factory + wrapper
# ---------------------------------------------------------------------------

def _video_vae(pretrained_path=None, z_dim=16, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    model = WanVAE_(**cfg)

    if pretrained_path is not None:
        logging.info(f'loading {pretrained_path}')
        if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
            import torch
            raw = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if isinstance(raw, dict) and 'state_dict' in raw:
                raw = raw['state_dict']
            weights = {k: mx.array(v.float().numpy()) for k, v in raw.items()}
            del raw
        else:
            weights = mx.load(pretrained_path)
        # Convert PyTorch state dict keys/shapes to MLX format
        model = _load_torch_weights(model, weights)

    return model


def _load_torch_weights(model, state_dict):
    """Load a PyTorch state dict into the MLX VAE model.

    Handles key remapping (nn.Sequential numeric → named attrs),
    CausalConv3d .conv insertion, and Conv2d/Conv3d weight transposition.
    """
    import re

    # Pass 1: remap nn.Sequential numeric indices to named attributes
    remapped = {}
    for k, v in state_dict.items():
        # ResidualBlock.residual: Sequential(RMS_norm, SiLU, Conv3d, RMS_norm, SiLU, Dropout, Conv3d)
        k = re.sub(r'\.residual\.0\.', '.norm1.', k)
        k = re.sub(r'\.residual\.2\.', '.conv1.', k)
        k = re.sub(r'\.residual\.3\.', '.norm2.', k)
        k = re.sub(r'\.residual\.6\.', '.conv2.', k)

        # Resample: Sequential(Upsample/ZeroPad, Conv2d)
        k = re.sub(r'\.resample\.1\.', '.conv2d.', k)

        # Encoder3d/Decoder3d middle: Sequential(ResBlock, AttnBlock, ResBlock)
        k = re.sub(r'\.middle\.0\.', '.mid_res1.', k)
        k = re.sub(r'\.middle\.1\.', '.mid_attn.', k)
        k = re.sub(r'\.middle\.2\.', '.mid_res2.', k)

        # Encoder3d/Decoder3d head: Sequential(RMS_norm, SiLU, CausalConv3d)
        k = re.sub(r'\.head\.0\.', '.head_norm.', k)
        k = re.sub(r'\.head\.2\.', '.head_conv.', k)

        remapped[k] = v

    # Pass 2: identify CausalConv3d modules (those with 5D weights)
    # and insert .conv. before .weight/.bias for them
    conv3d_prefixes = set()
    for k, v in remapped.items():
        if k.endswith('.weight') and v.ndim == 5:
            conv3d_prefixes.add(k[:-len('.weight')])

    final = {}
    for k, v in remapped.items():
        # Insert .conv. for CausalConv3d parameters
        if k.endswith('.weight') or k.endswith('.bias'):
            suffix = '.weight' if k.endswith('.weight') else '.bias'
            prefix = k[:-len(suffix)]
            if prefix in conv3d_prefixes:
                k = prefix + '.conv' + suffix

        # Conv3d weight: (O, I, D, H, W) -> (O, D, H, W, I)
        if v.ndim == 5:
            v = mx.transpose(v, axes=(0, 2, 3, 4, 1))
        # Conv2d weight: (O, I, H, W) -> (O, H, W, I)
        elif v.ndim == 4:
            v = mx.transpose(v, axes=(0, 2, 3, 1))

        final[k] = v

    model.load_weights(list(final.items()))
    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=mx.float32):
        self.dtype = dtype

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = mx.array(mean, dtype=dtype)
        self.std = mx.array(std, dtype=dtype)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        )

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        return [
            self.model.encode(mx.expand_dims(u, axis=0), self.scale).squeeze(axis=0)
            for u in videos
        ]

    def decode(self, zs):
        return [
            mx.clip(
                self.model.decode(mx.expand_dims(u, axis=0), self.scale).squeeze(axis=0),
                -1, 1)
            for u in zs
        ]
