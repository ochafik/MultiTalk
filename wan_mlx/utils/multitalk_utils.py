# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of multitalk_utils.py
import os
import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import imageio
import uuid
from tqdm import tqdm
import subprocess
import soundfile as sf
import binascii
import os.path as osp
from skimage import color

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ASPECT_RATIO_627 = {
    '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1),
    '0.82': ([576, 704], 1), '1.00': ([640, 640], 1), '1.22': ([704, 576], 1), '1.50': ([768, 512], 1),
    '1.86': ([832, 448], 1), '2.00': ([896, 448], 1), '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1),
    '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)}

ASPECT_RATIO_960 = {
    '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1),
    '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1),
    '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1),
    '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1),
    '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1),
    '3.75': ([1920, 512], 1)}


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):
    """Calculate attention map between visual query and reference key.

    Args:
        visual_q: [B, M, H, K]
        ref_k: [B, M, H, K]
        ref_target_masks: [num_classes, seq_len]
    """
    ref_k = ref_k.astype(visual_q.dtype)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    # B, M, H, K -> B, H, M, K
    visual_q = mx.transpose(visual_q, axes=(0, 2, 1, 3))
    ref_k = mx.transpose(ref_k, axes=(0, 2, 1, 3))
    # B, H, M, K @ B, H, K, M -> B, H, M, M
    attn = visual_q @ mx.transpose(ref_k, axes=(0, 1, 3, 2))

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = mx.softmax(attn, axis=-1)

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.astype(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.astype(visual_q.dtype)

    for class_idx in range(ref_target_masks.shape[0]):
        ref_target_mask = ref_target_masks[class_idx]
        ref_target_mask = ref_target_mask[None, None, None, :]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = mx.sum(x_ref_attnmap, axis=-1) / mx.sum(ref_target_mask)
        # B, H, M -> B, M, H
        x_ref_attnmap = mx.transpose(x_ref_attnmap, axes=(0, 2, 1))

        if mode == 'mean':
            x_ref_attnmap = mx.mean(x_ref_attnmap, axis=-1)
        elif mode == 'max':
            x_ref_attnmap = mx.max(x_ref_attnmap, axis=-1)

        x_ref_attn_maps.append(x_ref_attnmap)

    return mx.concatenate(x_ref_attn_maps, axis=0)


def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2):
    """Get attention map with target masks.

    Args:
        visual_q: [B, M, H, K]
        ref_k: [B, M, H, K]
        shape: (N_t, N_h, N_w)
        ref_target_masks: [num_classes, N_h * N_w]
    """
    N_t, N_h, N_w = shape
    x_seqlens = int(N_h) * int(N_w)
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = mx.zeros((class_num, seq_lens), dtype=visual_q.dtype)

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i * split_chunk:(i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk:(i + 1) * split_chunk, :],
            ref_target_masks)
        x_ref_attn_maps = x_ref_attn_maps + x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


def rotate_half(x):
    """Rotate half of the dimensions for RoPE."""
    # x: [..., d] -> split into [..., d/2, 2]
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    # Interleave: [-x2, x1] -> [..., d]
    # For the rotate_half pattern used in 1D RoPE:
    # rearrange(x, "... (d r) -> ... d r", r=2) then unbind and stack(-x2, x1)
    x_reshaped = mx.reshape(x, x.shape[:-1] + (d // 2, 2))
    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]
    rotated = mx.stack([-x2, x1], axis=-1)
    return mx.reshape(rotated, x.shape)


class RotaryPositionalEmbedding1D(nn.Module):
    """1D Rotary Positional Embedding for audio attention."""

    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def precompute_freqs_cis_1d(self, pos_indices):
        freqs = 1.0 / (self.base ** (mx.arange(0, self.head_dim, 2)[:self.head_dim // 2].astype(mx.float32) / self.head_dim))
        freqs = mx.einsum("..., f -> ... f", pos_indices.astype(mx.float32), freqs)
        # repeat: ... n -> ... (n r), r=2
        freqs = mx.repeat(freqs, repeats=2, axis=-1)
        return freqs

    def __call__(self, x, pos_indices):
        """Apply 1D RoPE.

        Args:
            x: [B, head, seq, head_dim]
            pos_indices: [seq,]
        Returns:
            x with same shape, rotated by positional embeddings.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)
        x_ = x.astype(mx.float32)
        freqs_cis = freqs_cis.astype(mx.float32)
        cos_f = mx.cos(freqs_cis)
        sin_f = mx.sin(freqs_cis)
        # Reshape for broadcasting: [seq, dim] -> [1, 1, seq, dim]
        cos_f = mx.reshape(cos_f, (1, 1) + cos_f.shape)
        sin_f = mx.reshape(sin_f, (1, 1) + sin_f.shape)
        x_ = (x_ * cos_f) + (rotate_half(x_) * sin_f)
        return x_.astype(x.dtype)


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0, v1):
    """Project v0 onto v1 and return parallel and orthogonal components."""
    dtype = v0.dtype
    v0 = v0.astype(mx.float32)
    v1 = v1.astype(mx.float32)
    # Normalize v1
    norm = mx.sqrt(mx.sum(v1 * v1, axis=(-1, -2, -3, -4), keepdims=True) + 1e-8)
    v1 = v1 / norm
    v0_parallel = mx.sum(v0 * v1, axis=(-1, -2, -3, -4), keepdims=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.astype(dtype), v0_orthogonal.astype(dtype)


def adaptive_projected_guidance(diff, pred_cond, momentum_buffer=None, eta=0.0, norm_threshold=55):
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = mx.ones_like(diff)
        diff_norm = mx.sqrt(mx.sum(diff * diff, axis=(-1, -2, -3, -4), keepdims=True))
        print(f"diff_norm: {diff_norm}")
        scale_factor = mx.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return normalized_update


def match_and_blend_colors(source_chunk, reference_image, strength):
    """Match colors of source video chunk to reference image using Lab color space.

    Args:
        source_chunk: mx.array [B, C, T, H, W] in [-1, 1]
        reference_image: mx.array [B, C, 1, H, W] in [-1, 1]
        strength: float 0-1

    Returns:
        Color-corrected mx.array
    """
    if strength == 0.0:
        return source_chunk

    # Convert to numpy for skimage operations
    # [1, C, T, H, W] -> [T, H, W, C]
    source_np = np.array(source_chunk[0]).transpose(1, 2, 3, 0)
    # [1, C, 1, H, W] -> [H, W, C]
    ref_np = np.array(reference_image[0, :, 0]).transpose(1, 2, 0)

    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError:
        return source_chunk

    corrected_frames = []
    for i in range(source_np_01.shape[0]):
        frame = source_np_01[i]
        try:
            source_lab = color.rgb2lab(frame)
        except ValueError:
            corrected_frames.append(frame)
            continue

        corrected = source_lab.copy()
        for j in range(3):
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()
            if std_src == 0:
                corrected[:, :, j] = mean_ref
            else:
                corrected[:, :, j] = (corrected[:, :, j] - mean_src) * (std_ref / std_src) + mean_ref

        try:
            corrected_rgb = color.lab2rgb(corrected)
        except ValueError:
            corrected_frames.append(frame)
            continue

        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
        blended = (1 - strength) * frame + strength * corrected_rgb
        corrected_frames.append(blended)

    result = np.stack(corrected_frames, axis=0)
    result = (result * 2.0) - 1.0
    # [T, H, W, C] -> [C, T, H, W]
    result = result.transpose(3, 0, 1, 2)
    result = mx.array(result[np.newaxis], dtype=source_chunk.dtype)
    return result


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5):
    """Save generated video with audio using ffmpeg.

    Args:
        gen_video_samples: mx.array [C, T, H, W] in [-1, 1]
        save_path: output path (without extension)
        vocal_audio_list: list of audio file paths
        fps: frames per second
        quality: video quality
    """
    video_np = np.array(gen_video_samples)
    video_np = (video_np + 1) / 2
    # C, T, H, W -> T, H, W, C
    video_np = video_np.transpose(1, 2, 3, 0)
    video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)

    save_path_tmp = save_path + "-temp.mp4"
    writer = imageio.get_writer(save_path_tmp, fps=fps, quality=quality)
    for frame in tqdm(video_np, desc="Saving video"):
        writer.append_data(frame)
    writer.close()

    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    subprocess.run([
        "ffmpeg", "-i", vocal_audio_list[0],
        "-t", f'{duration}', save_path_crop_audio
    ], check=True)

    save_path_final = save_path + ".mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", save_path_tmp,
        "-i", save_path_crop_audio,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest", save_path_final
    ], check=True)
    os.remove(save_path_tmp)
    os.remove(save_path_crop_audio)
