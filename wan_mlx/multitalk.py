# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of MultiTalk pipeline
import gc
import importlib
import json
import logging
import math
import os
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from functools import partial
from PIL import Image
from tqdm import tqdm

from .modules.clip import CLIPModel
from .modules.multitalk_model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.multitalk_utils import (
    MomentumBuffer,
    adaptive_projected_guidance,
    match_and_blend_colors,
    ASPECT_RATIO_627,
    ASPECT_RATIO_960,
)

__all__ = ['MLXMultiTalkPipeline']


def _remap_dit_key(key):
    """Remap PyTorch nn.Sequential weight keys to MLX attribute names.

    PyTorch uses nn.Sequential with numeric indices (e.g. text_embedding.0.weight),
    but MLX uses explicit attribute names (e.g. text_emb_linear1.weight).
    """
    # text_embedding: Sequential(Linear, SiLU, Linear)
    # text_embedding.0 -> text_emb_linear1, text_embedding.2 -> text_emb_linear2
    key = key.replace('text_embedding.0.', 'text_emb_linear1.')
    key = key.replace('text_embedding.2.', 'text_emb_linear2.')

    # time_embedding: Sequential(Linear, SiLU, Linear)
    key = key.replace('time_embedding.0.', 'time_emb_linear1.')
    key = key.replace('time_embedding.2.', 'time_emb_linear2.')

    # time_projection: Sequential(SiLU, Linear) - index 1 is the Linear
    key = key.replace('time_projection.1.', 'time_proj_linear.')

    # blocks.N.ffn: Sequential(Linear, GELU, Linear)
    # ffn.0 -> ffn_linear1, ffn.2 -> ffn_linear2
    if '.ffn.0.' in key:
        key = key.replace('.ffn.0.', '.ffn_linear1.')
    if '.ffn.2.' in key:
        key = key.replace('.ffn.2.', '.ffn_linear2.')

    # img_emb.proj: Sequential(LayerNorm, Linear, GELU, Linear, LayerNorm)
    # proj.0 -> ln1, proj.1 -> linear1, proj.3 -> linear2, proj.4 -> ln2
    key = key.replace('img_emb.proj.0.', 'img_emb.ln1.')
    key = key.replace('img_emb.proj.1.', 'img_emb.linear1.')
    key = key.replace('img_emb.proj.3.', 'img_emb.linear2.')
    key = key.replace('img_emb.proj.4.', 'img_emb.ln2.')

    return key


def _remap_dit_weights(weights):
    """Remap and transpose DiT weights from PyTorch to MLX format."""
    remapped = {}
    for key, value in weights.items():
        new_key = _remap_dit_key(key)

        # Conv3d weight transposition: (O, I, D, H, W) -> (O, D, H, W, I)
        if 'patch_embedding.weight' in new_key and value.ndim == 5:
            value = mx.transpose(value, axes=(0, 2, 3, 4, 1))

        remapped[new_key] = value
    return remapped


def resize_and_centercrop(cond_image, target_size):
    """Resize image or array to target size with center crop.

    Args:
        cond_image: PIL Image or mx.array/numpy array.
        target_size: (target_h, target_w)

    Returns:
        mx.array in the format expected by the pipeline.
    """
    target_h, target_w = target_size

    if isinstance(cond_image, Image.Image):
        orig_h, orig_w = cond_image.height, cond_image.width
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_np = np.array(resized_image).astype(np.float32)

        # Center crop
        crop_top = (final_h - target_h) // 2
        crop_left = (final_w - target_w) // 2
        cropped = resized_np[crop_top:crop_top + target_h, crop_left:crop_left + target_w, :]

        # HWC -> 1, C, 1, H, W  (for conditioning frame)
        cropped = cropped.transpose(2, 0, 1)  # C, H, W
        cropped = cropped[np.newaxis, :, np.newaxis, :, :]  # 1, C, 1, H, W
        return mx.array(cropped)

    elif isinstance(cond_image, mx.array):
        cond_np = np.array(cond_image)
    elif isinstance(cond_image, np.ndarray):
        cond_np = cond_image
    else:
        raise TypeError(f"Unsupported type: {type(cond_image)}")

    # Handle mx.array / numpy (tensor path) - shape: (num_classes, H, W) or (C, H, W)
    if cond_np.ndim == 3:
        num_ch, orig_h, orig_w = cond_np.shape
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        # Nearest-neighbor resize via repeat
        result = np.zeros((num_ch, final_h, final_w), dtype=cond_np.dtype)
        for ch in range(num_ch):
            pil_img = Image.fromarray(
                (cond_np[ch] * 255).clip(0, 255).astype(np.uint8)
                if cond_np.max() <= 1.0
                else cond_np[ch].astype(np.uint8)
            )
            resized = pil_img.resize((final_w, final_h), resample=Image.NEAREST)
            result[ch] = np.array(resized).astype(np.float32)
            if cond_np.max() <= 1.0:
                result[ch] /= 255.0

        # Center crop
        crop_top = (final_h - target_h) // 2
        crop_left = (final_w - target_w) // 2
        cropped = result[:, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
        return mx.array(cropped)

    raise ValueError(f"Unexpected ndim={cond_np.ndim}")


def timestep_transform(t, shift=5.0, num_timesteps=1000):
    """Shift timestep schedule for flow matching."""
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    return new_t * num_timesteps


class MLXMultiTalkPipeline:

    def __init__(
        self,
        config,
        checkpoint_dir,
        num_timesteps=1000,
        use_timestep_transform=True,
        quantize_bits=None,
    ):
        """Initialize the MLX MultiTalk pipeline.

        Args:
            config: EasyDict with model parameters from wan.configs.
            checkpoint_dir: Path to model checkpoint directory.
            num_timesteps: Number of diffusion timesteps.
            use_timestep_transform: Whether to apply timestep shift.
            quantize_bits: None, 4, or 8 for MLX quantization.
        """
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = mx.bfloat16  # MLX standard dtype

        # T5 text encoder
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=mx.bfloat16,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        # VAE
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        )

        # CLIP
        self.clip = CLIPModel(
            dtype=mx.bfloat16,
            checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer),
        )

        # DiT model
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        wan_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(wan_config_path):
            with open(wan_config_path) as f:
                wan_config = json.load(f)
            # Filter out diffusers metadata keys
            wan_config = {k: v for k, v in wan_config.items()
                         if not k.startswith('_')}
            self.model = WanModel(**wan_config)
        else:
            # Fall back to config-based construction
            self.model = WanModel()

        # Load weights from safetensors (use index.json if available)
        index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        dit_weights_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(index_path):
            # Sharded model: read index to find all shard files
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            all_weights = {}
            for sf_name in shard_files:
                sf_path = os.path.join(checkpoint_dir, sf_name)
                logging.info(f"Loading DiT shard: {sf_path}")
                shard_weights = mx.load(sf_path)
                all_weights.update(shard_weights)
            remapped = _remap_dit_weights(all_weights)
            self.model.load_weights(list(remapped.items()))
        elif os.path.exists(dit_weights_path):
            logging.info(f"Loading DiT weights from {dit_weights_path}")
            weights = mx.load(dit_weights_path)
            remapped = _remap_dit_weights(weights)
            self.model.load_weights(list(remapped.items()))
        else:
            logging.warning(
                f"No safetensors weights found in {checkpoint_dir}. "
                "Model will use random weights."
            )

        # Re-initialize RoPE frequencies (they are not saved in weights)
        self.model.init_freqs()

        # Apply quantization if requested
        if quantize_bits is not None:
            logging.info(f"Quantizing model to {quantize_bits} bits")
            nn.quantize(self.model, bits=quantize_bits)

        mx.eval(self.model.parameters())

        self.sample_neg_prompt = config.sample_neg_prompt
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform

    def add_noise(self, original_samples, noise, timesteps):
        """Add noise to samples at given timestep (flow matching schedule).

        Args:
            original_samples: mx.array - clean latents
            noise: mx.array - noise tensor
            timesteps: mx.array - scalar timestep value

        Returns:
            mx.array - noisy samples
        """
        t = timesteps.astype(mx.float32) / self.num_timesteps
        # Reshape for broadcasting: add dims for C, T, H, W
        while t.ndim < noise.ndim:
            t = mx.expand_dims(t, axis=-1)
        return (1 - t) * original_samples + t * noise

    def generate(
        self,
        input_data,
        size_buckget='multitalk-480',
        motion_frame=25,
        frame_num=81,
        shift=5.0,
        sampling_steps=40,
        text_guide_scale=5.0,
        audio_guide_scale=4.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        max_frames_num=1000,
        face_scale=0.05,
        progress=True,
        color_correction_strength=0.0,
        extra_args=None,
    ):
        """Generate video frames from input image and audio.

        Args:
            input_data: dict with 'prompt', 'cond_image', 'cond_audio' keys.
            size_buckget: Size bucket name ('multitalk-480' or 'multitalk-720').
            motion_frame: Number of motion frames for streaming mode.
            frame_num: Frames per clip (should be 4n+1).
            shift: Noise schedule shift parameter.
            sampling_steps: Number of denoising steps.
            text_guide_scale: Text CFG scale.
            audio_guide_scale: Audio CFG scale.
            n_prompt: Negative prompt.
            seed: Random seed (-1 for random).
            offload_model: Unused in MLX (unified memory).
            max_frames_num: Maximum total frames.
            face_scale: Scale for face region detection.
            progress: Show progress bar.
            color_correction_strength: Color correction strength [0, 1].
            extra_args: Additional arguments (teacache, APG settings).

        Returns:
            mx.array of shape [C, T, H, W] in [-1, 1].
        """
        # Init TeaCache
        if extra_args is not None and extra_args.use_teacache:
            self.model.teacache_init(
                sample_steps=sampling_steps,
                teacache_thresh=extra_args.teacache_thresh,
                model_scale=extra_args.size,
            )
        else:
            self.model.disable_teacache()

        input_prompt = input_data['prompt']
        cond_file_path = input_data['cond_image']
        cond_image = Image.open(cond_file_path).convert('RGB')

        # Decide proper size from aspect ratio bucket
        if size_buckget == 'multitalk-480':
            bucket_config = ASPECT_RATIO_627
        elif size_buckget == 'multitalk-720':
            bucket_config = ASPECT_RATIO_960
        else:
            raise NotImplementedError(f'Unsupported size bucket: {size_buckget}')

        src_h, src_w = cond_image.height, cond_image.width
        ratio = src_h / src_w
        closest_bucket = sorted(
            list(bucket_config.keys()),
            key=lambda x: abs(float(x) - ratio)
        )[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        cond_image = resize_and_centercrop(cond_image, (target_h, target_w))

        cond_image = cond_image / 255.0
        cond_image = (cond_image - 0.5) * 2  # normalize to [-1, 1]

        # Store original for color correction
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image

        # Read audio embeddings (saved as .pt from Wav2Vec2)
        audio_embedding_path_1 = input_data['cond_audio']['person1']
        if len(input_data['cond_audio']) == 1:
            HUMAN_NUMBER = 1
            audio_embedding_path_2 = None
        else:
            HUMAN_NUMBER = 2
            audio_embedding_path_2 = input_data['cond_audio']['person2']

        full_audio_embs = []
        audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
        for human_idx in range(HUMAN_NUMBER):
            audio_embedding_path = audio_embedding_paths[human_idx]
            if not os.path.exists(audio_embedding_path):
                continue

            # Load audio embedding - support both .pt and safetensors
            if audio_embedding_path.endswith('.safetensors'):
                emb_dict = mx.load(audio_embedding_path)
                full_audio_emb = list(emb_dict.values())[0]
            else:
                # .pt file: load via torch then convert
                import torch
                emb = torch.load(audio_embedding_path, map_location='cpu')
                full_audio_emb = mx.array(emb.numpy())

            if mx.any(mx.isnan(full_audio_emb)):
                continue
            if full_audio_emb.shape[0] <= frame_num:
                continue
            full_audio_embs.append(full_audio_emb)

        assert len(full_audio_embs) == HUMAN_NUMBER, \
            "Audio file not found or length does not satisfy frame_num."

        # Preprocess text embedding
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        mx.eval(context, context_null)

        # Prepare params for iterative generation
        indices = (mx.arange(2 * 2 + 1) - 2) * 1
        clip_length = frame_num
        is_first_clip = True
        arrive_last_frame = False
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []

        # Set random seed
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        mx.random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Start iterative video generation
        while True:
            audio_embs = []
            for human_idx in range(HUMAN_NUMBER):
                center_indices_np = np.arange(audio_start_idx, audio_end_idx, 1)
                indices_np = np.array(indices)
                # center_indices: [frame_num, window_size]
                center_idx = center_indices_np[:, None] + indices_np[None, :]
                center_idx = np.clip(
                    center_idx, 0, full_audio_embs[human_idx].shape[0] - 1
                )
                center_idx_mx = mx.array(center_idx.astype(np.int32))
                # Gather audio embeddings
                audio_emb = full_audio_embs[human_idx][center_idx_mx]
                audio_emb = mx.expand_dims(audio_emb, axis=0)  # 1, T, W, S, C
                audio_embs.append(audio_emb)

            audio_embs = mx.concatenate(audio_embs, axis=0).astype(self.param_dtype)
            mx.eval(audio_embs)

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h = h // self.vae_stride[1]
            lat_w = w // self.vae_stride[2]
            max_seq_len = (
                ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w
                // (self.patch_size[1] * self.patch_size[2])
            )

            noise = mx.random.normal(
                shape=(16, (frame_num - 1) // 4 + 1, lat_h, lat_w),
            ).astype(mx.float32)

            # Build mask
            msk = mx.ones((1, frame_num, lat_h, lat_w))
            # Zero out frames beyond the motion frames
            msk_np = np.ones((1, frame_num, lat_h, lat_w), dtype=np.float32)
            msk_np[:, cur_motion_frames_num:] = 0
            msk = mx.array(msk_np)
            # Reshape: first frame gets 4 repeats
            first_frame_repeat = mx.repeat(msk[:, 0:1], repeats=4, axis=1)
            msk = mx.concatenate([first_frame_repeat, msk[:, 1:]], axis=1)
            msk = mx.reshape(msk, (1, msk.shape[1] // 4, 4, lat_h, lat_w))
            msk = mx.transpose(msk, axes=(0, 2, 1, 3, 4)).astype(self.param_dtype)

            # CLIP embedding
            clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).astype(
                self.param_dtype
            )
            mx.eval(clip_context)

            # VAE encode: zero-pad video frames
            video_frames = mx.zeros(
                (1, cond_image.shape[1], frame_num - cond_image.shape[2], target_h, target_w)
            )
            padding_frames_pixels_values = mx.concatenate(
                [cond_image, video_frames], axis=2
            )
            y = self.vae.encode(padding_frames_pixels_values)
            y = mx.stack(y).astype(self.param_dtype)  # B, C, T, H, W
            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
            latent_motion_frames = y[:, :, :cur_motion_frames_latent_num][0]  # C, T, H, W
            y = mx.concatenate([msk, y], axis=1)  # B, 4+C, T, H, W
            mx.eval(y, latent_motion_frames)

            # Construct human masks
            human_masks = []
            if HUMAN_NUMBER == 1:
                background_mask = np.ones((src_h, src_w), dtype=np.float32)
                human_mask1 = np.ones((src_h, src_w), dtype=np.float32)
                human_mask2 = np.ones((src_h, src_w), dtype=np.float32)
                human_masks = [human_mask1, human_mask2, background_mask]
            elif HUMAN_NUMBER == 2:
                if 'bbox' in input_data:
                    assert len(input_data['bbox']) == len(input_data['cond_audio']), \
                        "Number of bboxes must match cond_audio entries."
                    background_mask = np.zeros((src_h, src_w), dtype=np.float32)
                    for _, person_bbox in input_data['bbox'].items():
                        x_min, y_min, x_max, y_max = person_bbox
                        human_mask = np.zeros((src_h, src_w), dtype=np.float32)
                        human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                        background_mask += human_mask
                        human_masks.append(human_mask)
                else:
                    x_min = int(src_h * face_scale)
                    x_max = int(src_h * (1 - face_scale))
                    human_mask1 = np.zeros((src_h, src_w), dtype=np.float32)
                    human_mask2 = np.zeros((src_h, src_w), dtype=np.float32)
                    lefty_min = int((src_w // 2) * face_scale)
                    lefty_max = int((src_w // 2) * (1 - face_scale))
                    righty_min = int((src_w // 2) * face_scale + (src_w // 2))
                    righty_max = int((src_w // 2) * (1 - face_scale) + (src_w // 2))
                    human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                    human_mask2[x_min:x_max, righty_min:righty_max] = 1
                    background_mask = human_mask1 + human_mask2
                    human_masks = [human_mask1, human_mask2]
                background_mask = np.where(background_mask > 0, 0.0, 1.0)
                human_masks.append(background_mask)

            ref_target_masks = mx.array(np.stack(human_masks, axis=0))
            ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))

            # Downsample masks to latent resolution (nearest)
            _, _, _, _lat_h, _lat_w = y.shape
            ref_masks_np = np.array(ref_target_masks)
            # ref_masks_np: (num_classes, H, W) -> downsample to (_lat_h, _lat_w)
            from PIL import Image as PILImage
            downsampled = []
            for ch in range(ref_masks_np.shape[0]):
                pil_mask = PILImage.fromarray(
                    (ref_masks_np[ch] * 255).clip(0, 255).astype(np.uint8)
                )
                pil_mask = pil_mask.resize((_lat_w, _lat_h), resample=PILImage.NEAREST)
                downsampled.append(np.array(pil_mask).astype(np.float32) / 255.0)
            ref_target_masks = mx.array(np.stack(downsampled, axis=0))
            ref_target_masks = mx.where(ref_target_masks > 0, 1.0, 0.0)
            mx.eval(ref_target_masks)

            # Prepare timesteps
            timesteps = list(
                np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32)
            )
            timesteps.append(0.0)
            timesteps = [mx.array([float(t)]) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [
                    timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps)
                    for t in timesteps
                ]

            # Sample videos
            latent = noise

            # Condition configs
            arg_c = {
                'context': context,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': y,
                'audio': audio_embs,
                'ref_target_masks': ref_target_masks,
            }

            arg_null_text = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': y,
                'audio': audio_embs,
                'ref_target_masks': ref_target_masks,
            }

            arg_null_audio = {
                'context': context,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': y,
                'audio': mx.zeros_like(audio_embs)[-1:],
                'ref_target_masks': ref_target_masks,
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': y,
                'audio': mx.zeros_like(audio_embs)[-1:],
                'ref_target_masks': ref_target_masks,
            }

            # Inject motion frames before denoising
            if not is_first_clip:
                latent_motion_frames = latent_motion_frames.astype(latent.dtype)
                motion_add_noise = mx.random.normal(latent_motion_frames.shape)
                add_latent = self.add_noise(
                    latent_motion_frames, motion_add_noise, timesteps[0]
                )
                T_m = add_latent.shape[1]
                latent_np = np.array(latent)
                latent_np[:, :T_m] = np.array(add_latent)
                latent = mx.array(latent_np)

            # APG buffers
            if extra_args is not None and extra_args.use_apg:
                text_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)
                audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)

            progress_wrap = partial(tqdm, total=len(timesteps) - 1) if progress else (lambda x: x)
            for i in progress_wrap(range(len(timesteps) - 1)):
                timestep = timesteps[i]
                latent_model_input = [latent]

                # Forward: conditioned
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c
                )[0]
                mx.eval(noise_pred_cond)

                if math.isclose(text_guide_scale, 1.0):
                    noise_pred_drop_audio = self.model(
                        latent_model_input, t=timestep, **arg_null_audio
                    )[0]
                    mx.eval(noise_pred_drop_audio)
                else:
                    noise_pred_drop_text = self.model(
                        latent_model_input, t=timestep, **arg_null_text
                    )[0]
                    mx.eval(noise_pred_drop_text)

                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null
                    )[0]
                    mx.eval(noise_pred_uncond)

                # Guidance
                if extra_args is not None and extra_args.use_apg:
                    if math.isclose(text_guide_scale, 1.0):
                        diff_uncond_audio = noise_pred_cond - noise_pred_drop_audio
                        noise_pred = noise_pred_cond + (audio_guide_scale - 1) * adaptive_projected_guidance(
                            diff_uncond_audio,
                            noise_pred_cond,
                            momentum_buffer=audio_momentumbuffer,
                            norm_threshold=extra_args.apg_norm_threshold,
                        )
                    else:
                        diff_uncond_text = noise_pred_cond - noise_pred_drop_text
                        diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                        noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(
                            diff_uncond_text,
                            noise_pred_cond,
                            momentum_buffer=text_momentumbuffer,
                            norm_threshold=extra_args.apg_norm_threshold,
                        ) + (audio_guide_scale - 1) * adaptive_projected_guidance(
                            diff_uncond_audio,
                            noise_pred_cond,
                            momentum_buffer=audio_momentumbuffer,
                            norm_threshold=extra_args.apg_norm_threshold,
                        )
                else:
                    # Vanilla CFG
                    if math.isclose(text_guide_scale, 1.0):
                        noise_pred = (
                            noise_pred_drop_audio
                            + audio_guide_scale * (noise_pred_cond - noise_pred_drop_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guide_scale * (noise_pred_cond - noise_pred_drop_text)
                            + audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                        )

                noise_pred = -noise_pred

                # Update latent (Euler step)
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps
                # Broadcast dt for element-wise multiply
                while dt.ndim < noise_pred.ndim:
                    dt = mx.expand_dims(dt, axis=-1)
                latent = latent + noise_pred * dt

                # Re-inject motion frames
                if not is_first_clip:
                    latent_motion_frames_cur = latent_motion_frames.astype(latent.dtype)
                    motion_add_noise = mx.random.normal(latent_motion_frames_cur.shape)
                    add_latent = self.add_noise(
                        latent_motion_frames_cur, motion_add_noise, timesteps[i + 1]
                    )
                    T_m = add_latent.shape[1]
                    latent_np = np.array(latent)
                    latent_np[:, :T_m] = np.array(add_latent)
                    latent = mx.array(latent_np)

                mx.eval(latent)

            # VAE decode
            x0 = [latent]
            videos = self.vae.decode(x0)
            videos = mx.stack(videos)  # B, C, T, H, W
            mx.eval(videos)

            # Color correction
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(
                    videos, original_color_reference, color_correction_strength
                )

            # Cache generated clip
            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # Check if done
            if arrive_last_frame:
                break

            # Update for next clip
            is_first_clip = False
            cur_motion_frames_num = motion_frame
            cond_image = videos[:, :, -cur_motion_frames_num:].astype(mx.float32)
            audio_start_idx += (frame_num - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # Handle end of audio
            if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                for human_idx in range(HUMAN_NUMBER):
                    source_frame = len(full_audio_embs[human_idx])
                    if audio_end_idx >= len(full_audio_embs[human_idx]):
                        miss_length = audio_end_idx - len(full_audio_embs[human_idx]) + 3
                        # Flip last miss_length frames and append
                        tail = full_audio_embs[human_idx][-miss_length:]
                        add_audio_emb = tail[::-1]
                        full_audio_embs[human_idx] = mx.concatenate(
                            [full_audio_embs[human_idx], add_audio_emb], axis=0
                        )
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            if max_frames_num <= frame_num:
                break

            mx.metal.clear_cache()

        # Concatenate all clips
        gen_video_samples = mx.concatenate(gen_video_list, axis=2)[:, :, :int(max_frames_num)]
        gen_video_samples = gen_video_samples.astype(mx.float32)

        if max_frames_num > frame_num and sum(miss_lengths) > 0:
            gen_video_samples = gen_video_samples[:, :, :-miss_lengths[0]]

        mx.eval(gen_video_samples)
        mx.metal.clear_cache()

        return gen_video_samples[0]
