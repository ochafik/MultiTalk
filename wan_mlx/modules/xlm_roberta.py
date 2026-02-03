# Modified from transformers.models.xlm_roberta.modeling_xlm_roberta
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port
import mlx.core as mx
import mlx.nn as nn

__all__ = ['XLMRoberta', 'xlm_roberta_large']


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # dropout removed for inference

    def __call__(self, x, mask):
        """
        x:   [B, L, C].
        """
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).transpose(0, 2, 1, 3)

        # compute attention using MLX scaled_dot_product_attention
        x = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, post_norm, dropout=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        # layers
        self.attn = SelfAttention(dim, num_heads, dropout, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.ffn_linear1 = nn.Linear(dim, dim * 4)
        self.ffn_act = nn.GELU()
        self.ffn_linear2 = nn.Linear(dim * 4, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x, mask):
        if self.post_norm:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self._ffn(x))
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self._ffn(self.norm2(x))
        return x

    def _ffn(self, x):
        x = self.ffn_linear1(x)
        x = self.ffn_act(x)
        x = self.ffn_linear2(x)
        return x


class XLMRoberta(nn.Module):
    """
    XLMRobertaModel with no pooler and no LM head.
    """

    def __init__(self,
                 vocab_size=250002,
                 max_seq_len=514,
                 type_size=1,
                 pad_id=1,
                 dim=1024,
                 num_heads=16,
                 num_layers=24,
                 post_norm=True,
                 dropout=0.1,
                 eps=1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        # embeddings (padding_idx not supported in MLX, ignored)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.type_embedding = nn.Embedding(type_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # blocks (plain list instead of nn.ModuleList)
        self.blocks = [
            AttentionBlock(dim, num_heads, post_norm, dropout, eps)
            for _ in range(num_layers)
        ]

        # norm layer
        self.norm = nn.LayerNorm(dim, eps=eps)

    def __call__(self, ids):
        """
        ids: [B, L] of int32.
        """
        b, s = ids.shape
        mask = (ids != self.pad_id).astype(mx.int32)

        # embeddings
        zeros = mx.zeros_like(ids)
        pos_ids = self.pad_id + mx.cumsum(mask, axis=1) * mask
        x = self.token_embedding(ids) + \
            self.type_embedding(zeros) + \
            self.pos_embedding(pos_ids)
        if self.post_norm:
            x = self.norm(x)

        # blocks
        # mask: [B, S] -> [B, 1, 1, S] attention mask
        attn_mask = mx.where(
            mask.reshape(b, 1, 1, s) > 0, 0.0, -1e9)
        for block in self.blocks:
            x = block(x, attn_mask)

        # output
        if not self.post_norm:
            x = self.norm(x)
        return x


def xlm_roberta_large(pretrained=False,
                      return_tokenizer=False,
                      **kwargs):
    """
    XLMRobertaLarge adapted from Huggingface.
    """
    # Remove device if passed
    kwargs.pop('device', None)

    # params
    cfg = dict(
        vocab_size=250002,
        max_seq_len=514,
        type_size=1,
        pad_id=1,
        dim=1024,
        num_heads=16,
        num_layers=24,
        post_norm=True,
        dropout=0.1,
        eps=1e-5)
    cfg.update(**kwargs)

    # init a model
    model = XLMRoberta(**cfg)
    return model
