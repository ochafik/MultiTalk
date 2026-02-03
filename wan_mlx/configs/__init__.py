# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX-native configs - no PyTorch dependency
import copy
import os
import mlx.core as mx
from easydict import EasyDict

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ----------------------- Shared config -----------------------
wan_shared_cfg = EasyDict()
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = mx.bfloat16
wan_shared_cfg.text_len = 512
wan_shared_cfg.param_dtype = mx.bfloat16
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0cJPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70'

# ----------------------- MultiTalk 14B -----------------------
multitalk_14B = EasyDict(__name__='Config: Wan MultiTalk AI2V 14B')
multitalk_14B.update(wan_shared_cfg)
multitalk_14B.sample_neg_prompt = 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'

multitalk_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
multitalk_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
multitalk_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
multitalk_14B.clip_dtype = mx.float16
multitalk_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
multitalk_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
multitalk_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
multitalk_14B.vae_stride = (4, 8, 8)

# transformer
multitalk_14B.patch_size = (1, 2, 2)
multitalk_14B.dim = 5120
multitalk_14B.ffn_dim = 13824
multitalk_14B.freq_dim = 256
multitalk_14B.num_heads = 40
multitalk_14B.num_layers = 40
multitalk_14B.window_size = (-1, -1)
multitalk_14B.qk_norm = True
multitalk_14B.cross_attn_norm = True
multitalk_14B.eps = 1e-6

# ----------------------- All configs -----------------------
WAN_CONFIGS = {
    'multitalk-14B': multitalk_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
    'multitalk-480': (640, 640),
    'multitalk-720': (960, 960),
}

SUPPORTED_SIZES = {
    'multitalk-14B': ('multitalk-480', 'multitalk-720'),
}
