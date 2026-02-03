# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of attention.py - replaces flash_attn/xformers with MLX attention
import math
import mlx.core as mx
import mlx.nn as nn

from ..utils.multitalk_utils import (
    RotaryPositionalEmbedding1D,
    normalize_and_scale,
)

__all__ = ['flash_attention', 'attention', 'SingleStreamMutiAttention']


def flash_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None,
    q_scale=None, causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=mx.bfloat16,
    version=None,
):
    """MLX attention replacing flash_attn/xformers.

    Args:
        q: [B, Lq, Nq, C1]
        k: [B, Lk, Nk, C1]
        v: [B, Lk, Nk, C2]
        q_lens: [B] optional sequence lengths for queries
        k_lens: [B] optional sequence lengths for keys
        softmax_scale: float, scaling factor
        q_scale: float, additional query scaling
        causal: bool, causal attention mask
    """
    b, lq, nq, d = q.shape
    _, lk, nk, _ = k.shape

    if q_scale is not None:
        q = q * q_scale

    # Transpose to [B, N, L, D] for scaled_dot_product_attention
    q = mx.transpose(q, axes=(0, 2, 1, 3))
    k = mx.transpose(k, axes=(0, 2, 1, 3))
    v = mx.transpose(v, axes=(0, 2, 1, 3))

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(d))

    # Build attention mask for variable-length sequences
    mask = None
    if k_lens is not None:
        # Create a mask [B, 1, 1, Lk] where positions beyond k_lens are masked
        positions = mx.arange(lk)[None, :]  # [1, Lk]
        k_lens_expanded = mx.array(k_lens)[:, None] if not isinstance(k_lens, mx.array) else k_lens[:, None]
        mask = mx.where(positions < k_lens_expanded, 0.0, -1e9)
        mask = mask[:, None, None, :]  # [B, 1, 1, Lk]

    if causal:
        causal_mask = mx.where(
            mx.triu(mx.ones((lq, lk)), k=1) > 0,
            -1e9, 0.0
        )
        if mask is not None:
            mask = mask + causal_mask[None, None, :, :]
        else:
            mask = causal_mask[None, None, :, :]

    # Use MLX scaled dot product attention
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    # Transpose back to [B, L, N, D]
    x = mx.transpose(x, axes=(0, 2, 1, 3))
    return x


def attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None,
    q_scale=None, causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=mx.bfloat16,
    fa_version=None,
):
    """Unified attention interface for MLX."""
    return flash_attention(
        q=q, k=k, v=v,
        q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p, softmax_scale=softmax_scale,
        q_scale=q_scale, causal=causal,
        window_size=window_size, deterministic=deterministic,
        dtype=dtype, version=fa_version,
    )


class SingleStreamAttention(nn.Module):
    """Single-stream attention for audio cross-attention."""

    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else lambda x: x
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else lambda x: x

        self.proj = nn.Linear(dim, dim)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else lambda x: x
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else lambda x: x

    def __call__(self, x, encoder_hidden_states, shape=None, **kwargs):
        N_t, N_h, N_w = shape

        # Reshape: B (N_t S) C -> (B N_t) S C
        B_orig = x.shape[0]
        S = x.shape[1] // N_t
        x = mx.reshape(x, (B_orig * N_t, S, x.shape[-1]))

        B, N, C = x.shape

        # Query
        q = self.q_linear(x)
        q = mx.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = mx.transpose(q, axes=(0, 2, 1, 3))  # B H N K

        if self.qk_norm:
            q = self.q_norm(q)

        # KV from encoder
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = mx.reshape(encoder_kv, (B, N_a, 2, self.num_heads, self.head_dim))
        encoder_kv = mx.transpose(encoder_kv, axes=(2, 0, 3, 1, 4))  # 2 B H N_a K
        encoder_k = encoder_kv[0]
        encoder_v = encoder_kv[1]

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # Transpose for attention: B H M K -> B M H K
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        encoder_k = mx.transpose(encoder_k, axes=(0, 2, 1, 3))
        encoder_v = mx.transpose(encoder_v, axes=(0, 2, 1, 3))

        # Attention using MLX sdpa
        scale = 1.0 / math.sqrt(self.head_dim)
        q_t = mx.transpose(q, axes=(0, 2, 1, 3))  # B H M K
        k_t = mx.transpose(encoder_k, axes=(0, 2, 1, 3))
        v_t = mx.transpose(encoder_v, axes=(0, 2, 1, 3))
        x = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
        # B H M K -> B M H K
        x = mx.transpose(x, axes=(0, 2, 1, 3))

        # B M H K -> B H M K -> B N C
        x = mx.transpose(x, axes=(0, 2, 1, 3))
        x = mx.transpose(x, axes=(0, 2, 1, 3))
        x = mx.reshape(x, (B, N, C))

        x = self.proj(x)

        # Reshape back: (B N_t) S C -> B (N_t S) C
        x = mx.reshape(x, (B_orig, N_t * S, C))
        return x


class SingleStreamMutiAttention(SingleStreamAttention):
    """Multi-person audio attention with 1D RoPE for spatial control."""

    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__(
            dim=dim,
            encoder_hidden_states_dim=encoder_hidden_states_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eps=eps,
        )
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, self.class_interval)
        self.rope_h2 = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def __call__(self, x, encoder_hidden_states, shape=None,
                 x_ref_attn_map=None, human_num=None):
        encoder_hidden_states = mx.squeeze(encoder_hidden_states, axis=0)

        if human_num == 1:
            return super().__call__(x, encoder_hidden_states, shape)

        N_t = shape[0] if isinstance(shape, (list, tuple)) else shape[0].item()

        # Reshape: B (N_t S) C -> (B N_t) S C
        B_orig = x.shape[0]
        S = x.shape[1] // N_t
        x = mx.reshape(x, (B_orig * N_t, S, x.shape[-1]))

        B, N, C = x.shape

        # Query
        q = self.q_linear(x)
        q = mx.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = mx.transpose(q, axes=(0, 2, 1, 3))  # B H N K

        if self.qk_norm:
            q = self.q_norm(q)

        # Compute spatial positions from attention map
        max_vals = mx.max(x_ref_attn_map, axis=1, keepdims=True)[:, :, None]  # 2, 1, 1
        min_vals = mx.min(x_ref_attn_map, axis=1, keepdims=True)[:, :, None]  # 2, 1, 1
        max_min_vals = mx.concatenate([max_vals, min_vals], axis=2)

        h1_max = mx.max(max_min_vals[0, :, 0])
        h1_min = mx.min(max_min_vals[0, :, 1])
        h2_max = mx.max(max_min_vals[1, :, 0])
        h2_min = mx.min(max_min_vals[1, :, 1])

        human1 = normalize_and_scale(x_ref_attn_map[0], (h1_min, h1_max), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (h2_min, h2_max), (self.rope_h2[0], self.rope_h2[1]))
        back = mx.full((x_ref_attn_map.shape[1],), self.rope_bak, dtype=human1.dtype)

        max_indices = mx.argmax(x_ref_attn_map, axis=0)
        normalized_map = mx.stack([human1, human2, back], axis=1)
        # Gather: for each position, pick the class with max attention
        normalized_pos = mx.take_along_axis(
            normalized_map,
            max_indices[:, None],
            axis=1
        )[:, 0]

        # Apply 1D RoPE to queries
        q = mx.reshape(q, (B_orig, self.num_heads, N_t * S, self.head_dim))
        q = self.rope_1d(q, normalized_pos)
        q = mx.reshape(q, (B_orig * N_t, self.num_heads, S, self.head_dim))

        # KV from encoder
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = mx.reshape(encoder_kv, (B, N_a, 2, self.num_heads, self.head_dim))
        encoder_kv = mx.transpose(encoder_kv, axes=(2, 0, 3, 1, 4))
        encoder_k = encoder_kv[0]
        encoder_v = encoder_kv[1]

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # Position encode the keys
        per_frame = mx.zeros((N_a,), dtype=encoder_k.dtype)
        half = N_a // 2
        per_frame = mx.concatenate([
            mx.full((half,), (self.rope_h1[0] + self.rope_h1[1]) / 2, dtype=per_frame.dtype),
            mx.full((N_a - half,), (self.rope_h2[0] + self.rope_h2[1]) / 2, dtype=per_frame.dtype)
        ])
        encoder_pos = mx.tile(per_frame, (N_t,))
        encoder_k = mx.reshape(encoder_k, (B_orig, self.num_heads, N_t * N_a, self.head_dim))
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = mx.reshape(encoder_k, (B_orig * N_t, self.num_heads, N_a, self.head_dim))

        # Attention
        # q: B H S K, encoder_k: B H N_a K
        scale = 1.0 / math.sqrt(self.head_dim)
        x = mx.fast.scaled_dot_product_attention(q, encoder_k, encoder_v, scale=scale)
        # B H S K -> B S H K -> B S C
        x = mx.transpose(x, axes=(0, 2, 1, 3))
        x = mx.reshape(x, (B, N, C))

        x = self.proj(x)

        # Reshape back
        x = mx.reshape(x, (B_orig, N_t * S, C))
        return x
