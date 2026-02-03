# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port - INFERENCE ONLY
import logging
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .attention import flash_attention
from .tokenizers import HuggingfaceTokenizer
from .xlm_roberta import XLMRoberta

__all__ = [
    'XLMRobertaCLIP',
    'clip_xlm_roberta_vit_h_14',
    'CLIPModel',
]


def _bicubic_interpolate_2d(data, out_h, out_w):
    """Bicubic interpolation for 2D spatial data using numpy/scipy.

    Args:
        data: mx.array of shape [C, H_in, W_in]
        out_h: target height
        out_w: target width
    Returns:
        mx.array of shape [C, out_h, out_w]
    """
    data_np = np.array(data.astype(mx.float32))
    try:
        from scipy.ndimage import zoom
        c, h_in, w_in = data_np.shape
        zoom_h = out_h / h_in
        zoom_w = out_w / w_in
        result = zoom(data_np, (1, zoom_h, zoom_w), order=3)  # order=3 = bicubic
    except ImportError:
        # Bilinear fallback using PIL per-channel
        c, h_in, w_in = data_np.shape
        result = np.zeros((c, out_h, out_w), dtype=data_np.dtype)
        for ci in range(c):
            pil_img = Image.fromarray(data_np[ci].astype(np.float32), mode='F')
            pil_img = pil_img.resize((out_w, out_h), Image.BICUBIC)
            result[ci] = np.array(pil_img)
    return mx.array(result)


def pos_interpolate(pos, seq_len):
    if pos.shape[1] == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.shape[1]))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.shape[1] - src_grid * src_grid
        # prefix tokens (e.g. CLS)
        prefix = pos[:, :n]
        # spatial tokens: [1, src_grid*src_grid, dim] -> [dim, src_grid, src_grid]
        spatial = pos[0, n:]  # [src_grid*src_grid, dim]
        spatial = spatial.reshape(src_grid, src_grid, -1)
        spatial = mx.transpose(spatial, axes=(2, 0, 1))  # [dim, src_grid, src_grid]
        spatial_interp = _bicubic_interpolate_2d(spatial, tar_grid, tar_grid)
        # [dim, tar_grid, tar_grid] -> [1, tar_grid*tar_grid, dim]
        spatial_interp = mx.transpose(spatial_interp, axes=(1, 2, 0))
        spatial_interp = spatial_interp.reshape(1, tar_grid * tar_grid, -1)
        return mx.concatenate([prefix, spatial_interp], axis=1)


class QuickGELU(nn.Module):

    def __call__(self, x):
        return x * mx.sigmoid(1.702 * x)


class LayerNorm(nn.Module):
    """LayerNorm that casts to float32 for computation then back."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.inner = nn.LayerNorm(dim, eps=eps)

    def __call__(self, x):
        orig_dtype = x.dtype
        x = self.inner(x.astype(mx.float32))
        return x.astype(orig_dtype)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x):
        """
        x:   [B, L, C].
        """
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        qkv = self.to_qkv(x).reshape(b, s, 3, n, d)
        q = qkv[:, :, 0]  # [B, S, N, D]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # compute attention (dropout=0 for inference)
        x = flash_attention(q, k, v, dropout_p=0.0, causal=self.causal, version=2)
        x = x.reshape(b, s, c)

        # output (no dropout for inference)
        x = self.proj(x)
        return x


class SwiGLU(nn.Module):

    def __init__(self, dim, mid_dim):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def __call__(self, x):
        x = nn.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert activation in ['quick_gelu', 'gelu', 'swi_glu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == 'swi_glu':
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            # Sequential as explicit layers (no nn.Sequential / no Dropout)
            self.mlp_linear1 = nn.Linear(dim, int(dim * mlp_ratio))
            self.mlp_act = QuickGELU() if activation == 'quick_gelu' else nn.GELU()
            self.mlp_linear2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.mlp = None  # flag: use explicit layers

    def __call__(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self._mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self._mlp(self.norm2(x))
        return x

    def _mlp(self, x):
        if self.mlp is not None:
            return self.mlp(x)
        x = self.mlp_linear1(x)
        x = self.mlp_act(x)
        x = self.mlp_linear2(x)
        return x


class AttentionPool(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 activation='gelu',
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = mx.random.normal((1, 1, dim)) * gain
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        # MLP as explicit layers (no nn.Sequential / no Dropout)
        self.mlp_linear1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp_act = QuickGELU() if activation == 'quick_gelu' else nn.GELU()
        self.mlp_linear2 = nn.Linear(int(dim * mlp_ratio), dim)

    def __call__(self, x):
        """
        x:  [B, L, C].
        """
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim

        # compute query, key, value
        cls_emb = mx.broadcast_to(self.cls_embedding, (b, 1, c))
        q = self.to_q(cls_emb).reshape(b, 1, n, d)
        kv = self.to_kv(x).reshape(b, s, 2, n, d)
        k = kv[:, :, 0]  # [B, S, N, D]
        v = kv[:, :, 1]

        # compute attention
        x = flash_attention(q, k, v, version=2)
        x = x.reshape(b, 1, c)

        # output (no dropout for inference)
        x = self.proj(x)

        # mlp
        residual = x
        x = self.norm(x)
        x = self.mlp_linear1(x)
        x = self.mlp_act(x)
        x = self.mlp_linear2(x)
        x = residual + x
        return x[:, 0]


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pool_type='token',
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self._post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        # Conv2d for patch embedding
        # MLX Conv2d expects channels-last [B, H, W, C]
        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm)
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = mx.random.normal((1, 1, dim)) * gain
        else:
            self.cls_embedding = None
        num_pos = self.num_patches + (1 if pool_type in ('token', 'token_fc') else 0)
        self.pos_embedding = mx.random.normal((1, num_pos, dim)) * gain
        # embedding_dropout removed for inference

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = [
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ]
        self.post_norm_layer = LayerNorm(dim, eps=norm_eps)

        # head
        if pool_type == 'token':
            self.head = mx.random.normal((dim, out_dim)) * gain
        elif pool_type == 'token_fc':
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)
        else:
            self.head = None

    def __call__(self, x, interpolation=False, use_31_block=False):
        b = x.shape[0]

        # MLX Conv2d expects [B, H, W, C] (channels-last)
        # Input is [B, 3, H, W] (channels-first), so transpose
        x_nhwc = mx.transpose(x, axes=(0, 2, 3, 1))  # [B, H, W, 3]
        x = self.patch_embedding(x_nhwc)  # [B, H/P, W/P, dim]
        # Flatten spatial: [B, H/P * W/P, dim]
        x = x.reshape(b, -1, self.dim)

        if self.pool_type in ('token', 'token_fc'):
            cls = mx.broadcast_to(self.cls_embedding, (b, 1, self.dim))
            x = mx.concatenate([cls, x], axis=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.shape[1])
        else:
            e = self.pos_embedding
        x = x + e
        # no dropout for inference
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            for block in self.transformer[:-1]:
                x = block(x)
            return x
        else:
            for block in self.transformer:
                x = block(x)
            return x


class XLMRobertaWithHead(XLMRoberta):

    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop('out_dim')
        super().__init__(**kwargs)

        # head: Linear -> GELU -> Linear
        mid_dim = (self.dim + self.out_dim) // 2
        self.head_linear1 = nn.Linear(self.dim, mid_dim, bias=False)
        self.head_act = nn.GELU()
        self.head_linear2 = nn.Linear(mid_dim, self.out_dim, bias=False)

    def __call__(self, ids):
        # xlm-roberta
        x = super().__call__(ids)

        # average pooling
        mask = (ids != self.pad_id).astype(x.dtype)
        mask = mx.expand_dims(mask, axis=-1)  # [B, L, 1]
        x = (x * mask).sum(axis=1) / mask.sum(axis=1)

        # head
        x = self.head_linear1(x)
        x = self.head_act(x)
        x = self.head_linear2(x)
        return x


class XLMRobertaCLIP(nn.Module):

    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 activation='gelu',
                 vocab_size=250002,
                 max_text_len=514,
                 type_size=1,
                 pad_id=1,
                 text_dim=1024,
                 text_heads=16,
                 text_layers=24,
                 text_post_norm=True,
                 text_dropout=0.1,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            post_norm=text_post_norm,
            dropout=text_dropout)
        self.log_scale = mx.array(math.log(1 / 0.07))

    def __call__(self, imgs, txt_ids):
        """
        imgs:       [B, 3, H, W] of float32.
        txt_ids:    [B, L] of int32.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt


def _clip(pretrained=False,
          pretrained_name=None,
          model_cls=XLMRobertaCLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=mx.float32,
          **kwargs):
    # Remove device if passed (not used in MLX)
    kwargs.pop('device', None)

    # init a model
    model = model_cls(**kwargs)
    output = (model,)

    # init transforms (numpy/PIL-based, replacing torchvision.transforms)
    if return_transforms:
        # mean and std
        if pretrained_name is not None and 'siglip' in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        image_size = model.image_size

        class _Normalize:
            """Replaces torchvision.transforms.Normalize for use in CLIPModel.visual()."""
            def __init__(self, mean, std):
                self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
                self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)

            def __call__(self, x):
                """
                x: mx.array [B, C, H, W] in [0, 1] range
                Returns: mx.array [B, C, H, W] normalized
                """
                mean_mx = mx.array(self.mean)
                std_mx = mx.array(self.std)
                return (x - mean_mx) / std_mx

        class _Transforms:
            """Mimics torchvision.transforms.Compose with .transforms attribute."""
            def __init__(self, normalize, image_size):
                self.image_size = image_size
                self.transforms = [normalize]  # transforms[-1] is Normalize

            def __call__(self, img):
                """Transform a PIL Image or numpy array to normalized mx.array.

                Args:
                    img: PIL Image or numpy array [H, W, 3] uint8
                Returns:
                    mx.array [3, H, W] normalized float32
                """
                if isinstance(img, Image.Image):
                    img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                    img = np.array(img).astype(np.float32) / 255.0
                else:
                    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(
                        (self.image_size, self.image_size), Image.BICUBIC)).astype(np.float32) / 255.0
                # Normalize: (img - mean) / std
                mean_arr = np.array(mean).reshape(1, 1, 3)
                std_arr = np.array(std).reshape(1, 1, 3)
                img = (img - mean_arr) / std_arr
                # HWC -> CHW
                img = img.transpose(2, 0, 1)
                return mx.array(img)

        normalize = _Normalize(mean, std)
        transforms = _Transforms(normalize, image_size)
        output += (transforms,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(
        pretrained=False,
        pretrained_name='open-clip-xlm-roberta-large-vit-huge-14',
        **kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool='token',
        activation='gelu',
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0)
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class CLIPModel:

    def __init__(self, dtype, checkpoint_path, tokenizer_path, **kwargs):
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype)

        # load weights
        logging.info(f'loading {checkpoint_path}')
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
                # Visual AttentionBlock MLP: Sequential(Linear, act, Linear, Dropout)
                k = k.replace('.mlp.0.', '.mlp_linear1.')
                k = k.replace('.mlp.2.', '.mlp_linear2.')
                # Textual XLMRoberta FFN: Sequential(Linear, GELU, Linear)
                k = k.replace('.ffn.0.', '.ffn_linear1.')
                k = k.replace('.ffn.2.', '.ffn_linear2.')
                # XLMRobertaWithHead head: Sequential(Linear, GELU, Linear)
                if k.startswith('textual.head.0.'):
                    k = k.replace('textual.head.0.', 'textual.head_linear1.')
                elif k.startswith('textual.head.2.'):
                    k = k.replace('textual.head.2.', 'textual.head_linear2.')
                # VisionTransformer post_norm -> post_norm_layer
                k = k.replace('visual.post_norm.', 'visual.post_norm_layer.inner.')
                # LayerNorm wrapper: insert .inner. for visual norms
                k = k.replace('visual.pre_norm.', 'visual.pre_norm.inner.')
                import re
                k = re.sub(r'(visual\.transformer\.\d+\.norm[12])\.',
                           r'\1.inner.', k)
                # Conv2d weight transposition: (O, I, H, W) -> (O, H, W, I)
                if v.ndim == 4:
                    v = mx.transpose(v, axes=(0, 2, 3, 1))
                remapped[k] = v
            self.model.load_weights(list(remapped.items()))

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=self.model.max_text_len - 2,
            clean='whitespace')

    def visual(self, videos):
        """Process video frames through the visual encoder.

        Mimics the original PyTorch CLIPModel.visual() which:
        1. For each video u in videos: transpose(0,1) to [C,T,H,W],
           then F.interpolate to resize H,W to image_size.
        2. torch.cat all results -> [B, C, H_new, W_new]
        3. Scale from [-1,1] to [0,1] via .mul_(0.5).add_(0.5)
        4. Apply Normalize transform (the last transform)
        5. Forward through model.visual with use_31_block=True

        Args:
            videos: list of mx.array (or iterable of mx.array),
                    each of shape [C, T, H, W] or [T, C, H, W] depending on caller.
                    In practice, callers pass [C, 1, H, W] per element.
                    The original code does u.transpose(0,1) making [T, C, H, W].
                    Then F.interpolate treats it as [N=T, C=C, H, W] and resizes H,W.
        """
        size = (self.model.image_size,) * 2

        # Process each video: resize spatial dims and concatenate
        processed = []
        for u in videos:
            # u is [C, T, H, W] in original; u.transpose(0,1) -> [T, C, H, W]
            # F.interpolate on [T, C, H, W] resizes last 2 dims (H, W)
            u_np = np.array(u.astype(mx.float32))
            # Transpose: [C, T, H, W] -> [T, C, H, W]
            u_np = u_np.transpose(1, 0, 2, 3)
            t, c, h, w = u_np.shape
            # Resize each frame spatially using PIL bicubic
            frames = []
            for ti in range(t):
                frame = u_np[ti]  # [C, H, W]
                frame_hwc = frame.transpose(1, 2, 0)  # [H, W, C]
                # Use PIL for bicubic resize
                # Clip to valid range for uint8 conversion isn't needed here;
                # we work in float space
                resized_channels = []
                for ci in range(c):
                    pil_ch = Image.fromarray(frame[:, :, :][ci].astype(np.float32), mode='F')
                    pil_ch = pil_ch.resize((size[1], size[0]), Image.BICUBIC)
                    resized_channels.append(np.array(pil_ch))
                resized = np.stack(resized_channels, axis=0)  # [C, H_new, W_new]
                frames.append(resized)
            processed.append(np.stack(frames, axis=0))  # [T, C, H_new, W_new]

        # Concatenate along batch dim: [total_T, C, H_new, W_new]
        all_frames = np.concatenate(processed, axis=0)

        # Scale from [-1,1] to [0,1]: .mul_(0.5).add_(0.5)
        all_frames = all_frames * 0.5 + 0.5

        # Apply Normalize transform (the last transform in self.transforms)
        videos_mx = mx.array(all_frames)
        normalize = self.transforms.transforms[-1]
        videos_mx = normalize(videos_mx)

        # forward
        out = self.model.visual(videos_mx, use_31_block=True)
        return out
