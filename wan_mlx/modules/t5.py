# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port
import logging
import math
import json
import os

import mlx.core as mx
import mlx.nn as nn

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    if x.dtype == mx.float16:
        clamp = 65504.0 - 1000  # float16 max approx
        x = mx.clip(x, -clamp, clamp)
    return x


class GELU(nn.Module):

    def __call__(self, x):
        return 0.5 * x * (1.0 + mx.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        variance = (x.astype(mx.float32) ** 2).mean(axis=-1, keepdims=True)
        x = x * (1.0 / mx.sqrt(variance + self.eps))
        if self.weight.dtype in [mx.float16, mx.bfloat16]:
            x = x.astype(self.weight.dtype)
        return self.weight * x


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        # dropout removed for inference

    def __call__(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        context = x if context is None else context
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, -1, n, c)
        k = self.k(context).reshape(b, -1, n, c)
        v = self.v(context).reshape(b, -1, n, c)

        # attention bias
        attn_bias = mx.zeros((b, n, q.shape[1], k.shape[1]))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            if mask.ndim == 2:
                mask = mask.reshape(b, 1, 1, -1)
            else:
                mask = mx.expand_dims(mask, axis=1)
            attn_bias = mx.where(mask == 0, -1e9, attn_bias)

        # compute attention (T5 does not use scaling)
        attn = mx.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(attn.dtype)
        x = mx.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers: gate = Linear + GELU
        self.gate_linear = nn.Linear(dim, dim_ffn, bias=False)
        self.gate_act = GELU()
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x):
        x = self.fc1(x) * self.gate_act(self.gate_linear(x))
        x = self.fc2(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def __call__(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def __call__(self,
                 x,
                 mask=None,
                 encoder_states=None,
                 encoder_mask=None,
                 pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def __call__(self, lq, lk):
        rel_pos = mx.arange(lk).reshape(1, -1) - mx.arange(lq).reshape(-1, 1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        # [Lq, Lk, N] -> [N, Lq, Lk] -> [1, N, Lq, Lk]
        rel_pos_embeds = mx.transpose(rel_pos_embeds, axes=(2, 0, 1))
        rel_pos_embeds = mx.expand_dims(rel_pos_embeds, axis=0)
        return rel_pos_embeds

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos, dtype=mx.int32)
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (mx.log(rel_pos.astype(mx.float32) / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).astype(mx.int32)
        rel_pos_large = mx.minimum(
            rel_pos_large,
            mx.full(rel_pos_large.shape, num_buckets - 1, dtype=mx.int32))
        rel_buckets = rel_buckets + mx.where(
            rel_pos < max_exact,
            rel_pos.astype(mx.int32),
            rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.blocks = [
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids, mask=None):
        x = self.token_embedding(ids)
        e = self.pos_embedding(x.shape[1],
                               x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        return x


class T5Decoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.blocks = [
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.shape

        # causal mask
        if mask is None:
            mask = mx.tril(mx.ones((1, s, s)))
        elif mask.ndim == 2:
            mask = mx.tril(mx.broadcast_to(
                mx.expand_dims(mask, axis=1), (mask.shape[0], s, mask.shape[1])))

        # layers
        x = self.token_embedding(ids)
        e = self.pos_embedding(x.shape[1],
                               x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        return x


class T5Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=mx.float32,
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # Remove device if passed
    kwargs.pop('device', None)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    model = model_cls(**kwargs)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=mx.bfloat16,
        checkpoint_path=None,
        tokenizer_path=None,
        **kwargs,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        logging.info(f'loading {checkpoint_path}')
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype)

        # load weights
        if checkpoint_path is not None:
            if checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.pt'):
                import torch
                raw = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if isinstance(raw, dict) and 'state_dict' in raw:
                    raw = raw['state_dict']
                weights = {k: mx.array(v.float().numpy()) for k, v in raw.items()}
                del raw
            else:
                weights = mx.load(checkpoint_path)
            # Remap nn.Sequential keys to MLX named attributes
            remapped = {}
            for k, v in weights.items():
                k = k.replace('.gate.0.', '.gate_linear.')
                remapped[k] = v
            model.load_weights(list(remapped.items()))

        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = mx.array(ids)
        mask = mx.array(mask)
        seq_lens = (mask > 0).sum(axis=1).astype(mx.int32)
        context = self.model(ids, mask)
        return [u[:v.item()] for u, v in zip(context, seq_lens)]
