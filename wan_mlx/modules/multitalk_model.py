# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of multitalk_model.py
import math
import numpy as np
import os
import mlx.core as mx
import mlx.nn as nn

from .attention import flash_attention, SingleStreamMutiAttention
from ..utils.multitalk_utils import get_attn_map_with_target
import logging

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)

    sinusoid = position[:, None] * mx.power(
        10000.0, -mx.arange(half).astype(mx.float32) / half
    )[None, :]
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = mx.arange(max_seq_len).astype(mx.float32)[:, None] * (
        1.0 / mx.power(
            theta,
            mx.arange(0, dim, 2).astype(mx.float32) / dim
        )
    )[None, :]
    # Return (cos, sin) tuple instead of complex polar
    cos_freqs = mx.cos(freqs)
    sin_freqs = mx.sin(freqs)
    return (cos_freqs, sin_freqs)


def rope_apply(x, grid_sizes, freqs):
    """Apply rotary positional embeddings.

    Args:
        x: [B, S, N, D]
        grid_sizes: [B, 3] tensor of (f, h, w)
        freqs: tuple of 3 (cos, sin) pairs, one per frequency band
    """
    s, n, d = x.shape[1], x.shape[2], x.shape[3]
    c = d // 2  # half-dim for RoPE

    # Split frequency bands
    c0 = c - 2 * (c // 3)
    c1 = c // 3
    c2 = c // 3

    # freqs is a tuple of 3 (cos, sin) pairs
    freqs_split = [
        (freqs[0][:, :c0], freqs[1][:, :c0]),
        (freqs[0][:, c0:c0+c1], freqs[1][:, c0:c0+c1]),
        (freqs[0][:, c0+c1:c0+c1+c2], freqs[1][:, c0+c1:c0+c1+c2]),
    ]

    output = []
    # Convert grid_sizes to a python list
    if isinstance(grid_sizes, mx.array):
        grid_list = np.array(grid_sizes).tolist()
    else:
        grid_list = grid_sizes.tolist()

    for i, (f, h, w) in enumerate(grid_list):
        f, h, w = int(f), int(h), int(w)
        seq_len = f * h * w

        # x_i: [S, N, D]
        x_i = x[i, :s].astype(mx.float32)

        # Split into even/odd for manual rotation
        x_even = x_i[..., 0::2]  # [S, N, D//2]
        x_odd = x_i[..., 1::2]

        # Build frequency grids: expand each freq band over (f,h,w)
        # freqs_split[0]: temporal, shape [max_len, c0]
        # freqs_split[1]: height, shape [max_len, c1]
        # freqs_split[2]: width, shape [max_len, c2]

        cos0 = freqs_split[0][0][:f]  # [f, c0]
        sin0 = freqs_split[0][1][:f]
        cos1 = freqs_split[1][0][:h]  # [h, c1]
        sin1 = freqs_split[1][1][:h]
        cos2 = freqs_split[2][0][:w]  # [w, c2]
        sin2 = freqs_split[2][1][:w]

        # Expand to [f, h, w, cx] and concatenate
        cos0_exp = mx.broadcast_to(
            mx.reshape(cos0, (f, 1, 1, -1)), (f, h, w, cos0.shape[-1])
        )
        sin0_exp = mx.broadcast_to(
            mx.reshape(sin0, (f, 1, 1, -1)), (f, h, w, sin0.shape[-1])
        )
        cos1_exp = mx.broadcast_to(
            mx.reshape(cos1, (1, h, 1, -1)), (f, h, w, cos1.shape[-1])
        )
        sin1_exp = mx.broadcast_to(
            mx.reshape(sin1, (1, h, 1, -1)), (f, h, w, sin1.shape[-1])
        )
        cos2_exp = mx.broadcast_to(
            mx.reshape(cos2, (1, 1, w, -1)), (f, h, w, cos2.shape[-1])
        )
        sin2_exp = mx.broadcast_to(
            mx.reshape(sin2, (1, 1, w, -1)), (f, h, w, sin2.shape[-1])
        )

        # Concatenate along last dim: [f, h, w, c]
        cos_f = mx.concatenate([cos0_exp, cos1_exp, cos2_exp], axis=-1)
        sin_f = mx.concatenate([sin0_exp, sin1_exp, sin2_exp], axis=-1)

        # Reshape to [seq_len, 1, c]
        cos_f = mx.reshape(cos_f, (seq_len, 1, -1))
        sin_f = mx.reshape(sin_f, (seq_len, 1, -1))

        # Apply rotation: only to the first seq_len tokens
        x_even_rot = x_even[:seq_len] * cos_f - x_odd[:seq_len] * sin_f
        x_odd_rot = x_even[:seq_len] * sin_f + x_odd[:seq_len] * cos_f

        # Interleave back using stack + reshape
        # x_even_rot, x_odd_rot: [seq_len, N, c]
        interleaved = mx.stack([x_even_rot, x_odd_rot], axis=-1)
        # [seq_len, N, c, 2] -> [seq_len, N, D]
        rotated = mx.reshape(interleaved, (seq_len, n, d))

        # For tokens beyond seq_len, keep original
        if seq_len < s:
            remainder = x[i, seq_len:]
            x_i_out = mx.concatenate([rotated, remainder], axis=0)
        else:
            x_i_out = rotated

        output.append(x_i_out)
    return mx.stack(output).astype(mx.float32)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(dim, eps=eps) if qk_norm else (lambda x: x)
        self.norm_k = nn.RMSNorm(dim, eps=eps) if qk_norm else (lambda x: x)

    def __call__(self, x, seq_lens, grid_sizes, freqs, ref_target_masks=None):
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # query, key, value
        q = mx.reshape(self.norm_q(self.q(x)), (b, s, n, d))
        k = mx.reshape(self.norm_k(self.k(x)), (b, s, n, d))
        v = mx.reshape(self.v(x), (b, s, n, d))

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x_out = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        ).astype(x.dtype)

        # output
        x_out = mx.reshape(x_out, (b, s, -1))
        x_out = self.o(x_out)

        x_ref_attn_map = get_attn_map_with_target(
            q.astype(x.dtype), k.astype(x.dtype), grid_sizes[0],
            ref_target_masks=ref_target_masks
        )

        return x_out, x_ref_attn_map


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = nn.RMSNorm(dim, eps=eps) if qk_norm else (lambda x: x)

    def __call__(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = mx.reshape(self.norm_q(self.q(x)), (b, -1, n, d))
        k = mx.reshape(self.norm_k(self.k(context)), (b, -1, n, d))
        v = mx.reshape(self.v(context), (b, -1, n, d))
        k_img = mx.reshape(self.norm_k_img(self.k_img(context_img)), (b, -1, n, d))
        v_img = mx.reshape(self.v_img(context_img), (b, -1, n, d))

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x_out = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x_out = mx.reshape(x_out, (b, -1, self.dim))
        img_x = mx.reshape(img_x, (b, -1, self.dim))
        x_out = x_out + img_x
        x_out = self.o(x_out)
        return x_out


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 norm_input_visual=True,
                 class_range=24,
                 class_interval=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = nn.LayerNorm(dim, eps=eps) if cross_attn_norm else (lambda x: x)
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=False)

        # ffn (inline Sequential replacement)
        self.ffn_linear1 = nn.Linear(dim, ffn_dim)
        self.ffn_act = nn.GELU(approx="precise")
        self.ffn_linear2 = nn.Linear(ffn_dim, dim)

        # modulation
        self.modulation = mx.random.normal(shape=(1, 6, dim)) / dim**0.5

        # audio cross attention
        self.audio_cross_attn = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=nn.RMSNorm,
            class_range=class_range,
            class_interval=class_interval
        )
        self.norm_x = nn.LayerNorm(dim, eps=eps) if norm_input_visual else (lambda x: x)

    def _ffn(self, x):
        return self.ffn_linear2(self.ffn_act(self.ffn_linear1(x)))

    def __call__(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
    ):
        dtype = x.dtype

        # modulation: compute 6 chunks
        e_mod = (self.modulation + e).astype(mx.float32)
        e0 = e_mod[:, 0:1, :]
        e1 = e_mod[:, 1:2, :]
        e2 = e_mod[:, 2:3, :]
        e3 = e_mod[:, 3:4, :]
        e4 = e_mod[:, 4:5, :]
        e5 = e_mod[:, 5:6, :]

        # self-attention
        normed = self.norm1(x).astype(mx.float32)
        sa_input = (normed * (1 + e1) + e0).astype(dtype)
        y, x_ref_attn_map = self.self_attn(
            sa_input, seq_lens, grid_sizes, freqs,
            ref_target_masks=ref_target_masks
        )
        x = (x + y * e2).astype(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # cross attn of audio
        shape_val = grid_sizes[0] if isinstance(grid_sizes, list) else grid_sizes[0]
        x_a = self.audio_cross_attn(
            self.norm_x(x), encoder_hidden_states=audio_embedding,
            shape=shape_val, x_ref_attn_map=x_ref_attn_map, human_num=human_num
        )
        x = x + x_a

        # ffn
        normed2 = self.norm2(x).astype(mx.float32)
        ffn_input = (normed2 * (1 + e4) + e3).astype(dtype)
        y = self._ffn(ffn_input)
        x = (x + y * e5).astype(dtype)

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        head_out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.head = nn.Linear(dim, head_out_dim)

        # modulation
        self.modulation = mx.random.normal(shape=(1, 2, dim)) / dim**0.5

    def __call__(self, x, e):
        e_mod = (self.modulation + mx.expand_dims(e, axis=1)).astype(mx.float32)
        e0 = e_mod[:, 0:1, :]
        e1 = e_mod[:, 1:2, :]
        x = self.head(self.norm(x) * (1 + e1) + e0)
        return x


class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.act = nn.GELU(approx="precise")
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def __call__(self, image_embeds):
        x = self.ln1(image_embeds)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.ln2(x)
        return x


class AudioProjModel(nn.Module):

    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else (lambda x: x)

    def __call__(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]

        # process audio of first frame: "bz f w b c -> (bz f) w b c" then flatten
        bf = audio_embeds.shape[0] * audio_embeds.shape[1]
        audio_embeds = mx.reshape(audio_embeds, (bf,) + audio_embeds.shape[2:])
        batch_size = audio_embeds.shape[0]
        audio_embeds = mx.reshape(audio_embeds, (batch_size, -1))

        # process audio of latter frame
        bf_vf = audio_embeds_vf.shape[0] * audio_embeds_vf.shape[1]
        audio_embeds_vf = mx.reshape(audio_embeds_vf, (bf_vf,) + audio_embeds_vf.shape[2:])
        batch_size_vf = audio_embeds_vf.shape[0]
        audio_embeds_vf = mx.reshape(audio_embeds_vf, (batch_size_vf, -1))

        # first projection
        audio_embeds = nn.relu(self.proj1(audio_embeds))
        audio_embeds_vf = nn.relu(self.proj1_vf(audio_embeds_vf))

        # reshape back: "(bz f) c -> bz f c"
        audio_embeds = mx.reshape(audio_embeds, (B, -1, audio_embeds.shape[-1]))
        audio_embeds_vf = mx.reshape(audio_embeds_vf, (B, -1, audio_embeds_vf.shape[-1]))

        audio_embeds_c = mx.concatenate([audio_embeds, audio_embeds_vf], axis=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = mx.reshape(audio_embeds_c, (batch_size_c * N_t, C_a))

        # second projection
        audio_embeds_c = nn.relu(self.proj2(audio_embeds_c))

        context_tokens = mx.reshape(
            self.proj3(audio_embeds_c),
            (batch_size_c * N_t, self.context_tokens, self.output_dim)
        )

        # normalization and reshape: "(bz f) m c -> bz f m c"
        context_tokens = self.norm(context_tokens)
        context_tokens = mx.reshape(
            context_tokens,
            (batch_size_c, video_length, self.context_tokens, self.output_dim)
        )

        return context_tokens


class WanModel(nn.Module):
    """Wan diffusion backbone supporting image-to-video (MultiTalk)."""

    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # audio params
                 audio_window=5,
                 intermediate_dim=512,
                 output_dim=768,
                 context_tokens=32,
                 vae_scale=4,
                 norm_input_visual=True,
                 norm_output_audio=True,
                 weight_init=False):
        super().__init__()

        assert model_type == 'i2v', 'MultiTalk model requires model_type is i2v.'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # text_embedding (inline Sequential)
        self.text_emb_linear1 = nn.Linear(text_dim, dim)
        self.text_emb_act = nn.GELU(approx="precise")
        self.text_emb_linear2 = nn.Linear(dim, dim)

        # time_embedding (inline Sequential)
        self.time_emb_linear1 = nn.Linear(freq_dim, dim)
        self.time_emb_act = nn.SiLU()
        self.time_emb_linear2 = nn.Linear(dim, dim)

        # time_projection (inline Sequential)
        self.time_proj_act = nn.SiLU()
        self.time_proj_linear = nn.Linear(dim, dim * 6)

        # blocks
        self.blocks = [
            WanAttentionBlock(
                'i2v_cross_attn', dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                output_dim=output_dim, norm_input_visual=norm_input_visual
            )
            for _ in range(num_layers)
        ]

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        f0 = rope_params(1024, d - 4 * (d // 6))
        f1 = rope_params(1024, 2 * (d // 6))
        f2 = rope_params(1024, 2 * (d // 6))
        # Store as concatenated (cos, sin) tuple
        self.freqs = (
            mx.concatenate([f0[0], f1[0], f2[0]], axis=1),
            mx.concatenate([f0[1], f1[1], f2[1]], axis=1),
        )

        self.img_emb = MLPProj(1280, dim)

        # audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # teacache state
        self.enable_teacache = False

    def _text_embedding(self, x):
        return self.text_emb_linear2(self.text_emb_act(self.text_emb_linear1(x)))

    def _time_embedding(self, x):
        return self.time_emb_linear2(self.time_emb_act(self.time_emb_linear1(x)))

    def _time_projection(self, x):
        return self.time_proj_linear(self.time_proj_act(x))

    def init_freqs(self):
        d = self.dim // self.num_heads
        f0 = rope_params(1024, d - 4 * (d // 6))
        f1 = rope_params(1024, 2 * (d // 6))
        f2 = rope_params(1024, 2 * (d // 6))
        self.freqs = (
            mx.concatenate([f0[0], f1[0], f2[0]], axis=1),
            mx.concatenate([f0[1], f1[1], f2[1]], axis=1),
        )

    def teacache_init(
        self,
        use_ret_steps=True,
        teacache_thresh=0.2,
        sample_steps=40,
        model_scale='multitalk-480',
    ):
        print("teacache_init")
        self.enable_teacache = True

        self.cnt = 0
        self.num_steps = sample_steps * 3
        self.teacache_thresh = teacache_thresh
        self.accumulated_rel_l1_distance_cond = 0
        self.accumulated_rel_l1_distance_drop_text = 0
        self.accumulated_rel_l1_distance_uncond = 0
        self.previous_e0_cond = None
        self.previous_e0_drop_text = None
        self.previous_e0_uncond = None
        self.previous_residual_cond = None
        self.previous_residual_drop_text = None
        self.previous_residual_uncond = None
        self.use_ret_steps = use_ret_steps

        if use_ret_steps:
            if model_scale == 'multitalk-480':
                self.coefficients = [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
            if model_scale == 'multitalk-720':
                self.coefficients = [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02]
            self.ret_steps = 5 * 3
            self.cutoff_steps = sample_steps * 3
        else:
            if model_scale == 'multitalk-480':
                self.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
            if model_scale == 'multitalk-720':
                self.coefficients = [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683]
            self.ret_steps = 1 * 3
            self.cutoff_steps = sample_steps * 3 - 3
        print("teacache_init done")

    def disable_teacache(self):
        self.enable_teacache = False

    def __call__(
            self,
            x,
            t,
            context,
            seq_len,
            clip_fea=None,
            y=None,
            audio=None,
            ref_target_masks=None,
    ):
        assert clip_fea is not None and y is not None

        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]
        x[0] = x[0].astype(context[0].dtype)

        # embeddings: patch_embedding expects [B, D_in, T, H, W] for Conv3d
        # MLX Conv3d: input [N, D1, D2, D3, C_in] -> need to transpose
        # PyTorch Conv3d: [N, C, D1, D2, D3]
        # MLX Conv3d: [N, D1, D2, D3, C]
        x_embedded = []
        for u in x:
            # u: [C, T, H, W] -> [T, H, W, C] for MLX Conv3d
            u_t = mx.transpose(mx.expand_dims(u, axis=0), axes=(0, 2, 3, 4, 1))
            emb = self.patch_embedding(u_t)
            x_embedded.append(emb)

        grid_sizes = mx.stack(
            [mx.array(list(u.shape[1:4]), dtype=mx.int32) for u in x_embedded]
        )
        # Flatten spatial dims: [1, Nt, Nh, Nw, C] -> [1, Nt*Nh*Nw, C]
        x_flat = [mx.reshape(u, (1, -1, u.shape[-1])) for u in x_embedded]
        seq_lens = mx.array([u.shape[1] for u in x_flat], dtype=mx.int32)
        assert int(mx.max(seq_lens).item()) <= seq_len

        # Pad and concatenate
        x_padded = mx.concatenate([
            mx.concatenate(
                [u, mx.zeros((1, seq_len - u.shape[1], u.shape[2]), dtype=u.dtype)],
                axis=1
            ) for u in x_flat
        ])

        # time embeddings
        e = self._time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).astype(mx.float32)
        )
        e0 = mx.reshape(self._time_projection(e), (e.shape[0], 6, self.dim))

        # text embedding
        context_lens = None
        context_padded = mx.stack([
            mx.concatenate(
                [u, mx.zeros((self.text_len - u.shape[0], u.shape[1]), dtype=u.dtype)],
                axis=0
            ) for u in context
        ])
        context_emb = self._text_embedding(context_padded)

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context_emb = mx.concatenate(
                [context_clip, context_emb], axis=1
            ).astype(x_padded.dtype)

        # audio processing
        audio_cond = audio.astype(x_padded.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]

        # rearrange: "b (n_t n) w s c -> b n_t n w s c", n=vae_scale
        b_a = latter_frame_audio_emb.shape[0]
        total_frames = latter_frame_audio_emb.shape[1]
        n_t_audio = total_frames // self.vae_scale
        latter_frame_audio_emb = mx.reshape(
            latter_frame_audio_emb,
            (b_a, n_t_audio, self.vae_scale) + latter_frame_audio_emb.shape[2:]
        )

        middle_index = self.audio_window // 2

        # latter_first_frame_audio_emb: [:, :, :1, :middle_index+1, ...]
        latter_first = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
        # "b n_t n w s c -> b n_t (n w) s c"
        latter_first = mx.reshape(
            latter_first,
            (latter_first.shape[0], latter_first.shape[1],
             latter_first.shape[2] * latter_first.shape[3]) + latter_first.shape[4:]
        )

        # latter_last_frame_audio_emb: [:, :, -1:, middle_index:, ...]
        latter_last = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last = mx.reshape(
            latter_last,
            (latter_last.shape[0], latter_last.shape[1],
             latter_last.shape[2] * latter_last.shape[3]) + latter_last.shape[4:]
        )

        # latter_middle_frame_audio_emb: [:, :, 1:-1, middle_index:middle_index+1, ...]
        latter_middle = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
        latter_middle = mx.reshape(
            latter_middle,
            (latter_middle.shape[0], latter_middle.shape[1],
             latter_middle.shape[2] * latter_middle.shape[3]) + latter_middle.shape[4:]
        )

        latter_frame_audio_emb_s = mx.concatenate(
            [latter_first, latter_middle, latter_last], axis=2
        )

        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        human_num = audio_embedding.shape[0]

        # Concat along token dim: split batch dim=0, then concat along dim=2
        # "split(1) along dim 0 -> concat along dim 2"
        parts = [audio_embedding[i:i+1] for i in range(human_num)]
        audio_embedding = mx.concatenate(parts, axis=2).astype(x_padded.dtype)

        # convert ref_target_masks to token_ref_target_masks
        token_ref_target_masks = None
        if ref_target_masks is not None:
            # ref_target_masks: [num_classes, H, W] -> interpolate to [num_classes, N_h, N_w]
            ref_target_masks_f = mx.expand_dims(ref_target_masks, axis=0).astype(mx.float32)
            # Nearest-neighbor interpolation via reshape trick
            # Input: [1, num_classes, H, W]
            # We need to downsample to [N_h, N_w]
            # Use stride-based slicing for nearest-neighbor
            nc = ref_target_masks_f.shape[1]
            h_orig = ref_target_masks_f.shape[2]
            w_orig = ref_target_masks_f.shape[3]
            h_stride = h_orig // N_h
            w_stride = w_orig // N_w
            token_ref_target_masks = ref_target_masks_f[:, :, ::h_stride, ::w_stride]
            token_ref_target_masks = token_ref_target_masks[:, :, :N_h, :N_w]
            token_ref_target_masks = mx.squeeze(token_ref_target_masks, axis=0)
            token_ref_target_masks = mx.where(token_ref_target_masks > 0, 1.0, 0.0)
            token_ref_target_masks = mx.reshape(
                token_ref_target_masks,
                (token_ref_target_masks.shape[0], -1)
            )
            token_ref_target_masks = token_ref_target_masks.astype(x_padded.dtype)

        x = x_padded

        # teacache logic
        if self.enable_teacache:
            modulated_inp = e0 if self.use_ret_steps else e

            if self.cnt % 3 == 0:  # cond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc = True
                    self.accumulated_rel_l1_distance_cond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    diff = mx.mean(mx.abs(modulated_inp - self.previous_e0_cond))
                    ref = mx.mean(mx.abs(self.previous_e0_cond))
                    self.accumulated_rel_l1_distance_cond += rescale_func(
                        float((diff / ref).item())
                    )
                    if self.accumulated_rel_l1_distance_cond < self.teacache_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance_cond = 0
                self.previous_e0_cond = modulated_inp

                if not should_calc:
                    x = x + self.previous_residual_cond
                else:
                    ori_x = x
                    for block in self.blocks:
                        x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                                  freqs=self.freqs, context=context_emb,
                                  context_lens=context_lens,
                                  audio_embedding=audio_embedding,
                                  ref_target_masks=token_ref_target_masks,
                                  human_num=human_num)
                    self.previous_residual_cond = x - ori_x

            elif self.cnt % 3 == 1:  # drop_text
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc = True
                    self.accumulated_rel_l1_distance_drop_text = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    diff = mx.mean(mx.abs(modulated_inp - self.previous_e0_drop_text))
                    ref = mx.mean(mx.abs(self.previous_e0_drop_text))
                    self.accumulated_rel_l1_distance_drop_text += rescale_func(
                        float((diff / ref).item())
                    )
                    if self.accumulated_rel_l1_distance_drop_text < self.teacache_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance_drop_text = 0
                self.previous_e0_drop_text = modulated_inp

                if not should_calc:
                    x = x + self.previous_residual_drop_text
                else:
                    ori_x = x
                    for block in self.blocks:
                        x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                                  freqs=self.freqs, context=context_emb,
                                  context_lens=context_lens,
                                  audio_embedding=audio_embedding,
                                  ref_target_masks=token_ref_target_masks,
                                  human_num=human_num)
                    self.previous_residual_drop_text = x - ori_x

            else:  # uncond
                if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc = True
                    self.accumulated_rel_l1_distance_uncond = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    diff = mx.mean(mx.abs(modulated_inp - self.previous_e0_uncond))
                    ref = mx.mean(mx.abs(self.previous_e0_uncond))
                    self.accumulated_rel_l1_distance_uncond += rescale_func(
                        float((diff / ref).item())
                    )
                    if self.accumulated_rel_l1_distance_uncond < self.teacache_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance_uncond = 0
                self.previous_e0_uncond = modulated_inp

                if not should_calc:
                    x = x + self.previous_residual_uncond
                else:
                    ori_x = x
                    for block in self.blocks:
                        x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                                  freqs=self.freqs, context=context_emb,
                                  context_lens=context_lens,
                                  audio_embedding=audio_embedding,
                                  ref_target_masks=token_ref_target_masks,
                                  human_num=human_num)
                    self.previous_residual_uncond = x - ori_x
        else:
            for block in self.blocks:
                x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                          freqs=self.freqs, context=context_emb,
                          context_lens=context_lens,
                          audio_embedding=audio_embedding,
                          ref_target_masks=token_ref_target_masks,
                          human_num=human_num)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        if self.enable_teacache:
            self.cnt += 1
            if self.cnt >= self.num_steps:
                self.cnt = 0

        return mx.stack(x).astype(mx.float32)

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings.

        Args:
            x: [B, L, C_out * prod(patch_size)]
            grid_sizes: [B, 3]

        Returns:
            List of tensors with shape [C_out, F, H, W]
        """
        c = self.out_dim
        out = []
        if isinstance(grid_sizes, mx.array):
            grid_list = np.array(grid_sizes).tolist()
        else:
            grid_list = grid_sizes.tolist()

        for idx, v in enumerate(grid_list):
            v = [int(vi) for vi in v]
            total = math.prod(v)
            u = x[idx, :total]
            u = mx.reshape(u, (*v, *self.patch_size, c))
            # einsum 'fhwpqrc->cfphqwr' is permutation (0,1,2,3,4,5,6)->(6,0,3,1,4,2,5)
            u = mx.transpose(u, axes=(6, 0, 3, 1, 4, 2, 5))
            u = mx.reshape(u, (c,) + tuple(i * j for i, j in zip(v, self.patch_size)))
            out.append(u)
        return out
