# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX port of generate_multitalk.py
import argparse
import logging
import os
import sys
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import numpy as np
import mlx.core as mx
from PIL import Image
import subprocess
import re

import librosa
import pyloudnorm as pyln
import soundfile as sf

from wan_mlx.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan_mlx.utils.multitalk_utils import save_video_ffmpeg

# Wav2Vec2 stays on PyTorch (HuggingFace model)
import torch
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model


def _validate_args(args):
    """Validate CLI arguments."""
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        if args.size == 'multitalk-480':
            args.sample_shift = 7
        elif args.size == 'multitalk-720':
            args.sample_shift = 11
        else:
            raise NotImplementedError(f'Not supported size: {args.size}')

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, 99999999)
    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupported size {args.size} for task {args.task}, " \
        f"supported sizes: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from image and audio using MultiTalk (MLX)"
    )
    parser.add_argument(
        "--task", type=str, default="multitalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size", type=str, default="multitalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="Size bucket for generated video.")
    parser.add_argument(
        "--frame_num", type=int, default=81,
        help="Frames per clip (should be 4n+1).")
    parser.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="Path to model checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir", type=str, default=None,
        help="Path to wav2vec checkpoint directory.")
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output file path (without extension).")
    parser.add_argument(
        "--audio_save_dir", type=str, default='save_audio',
        help="Directory to save audio embeddings.")
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument(
        "--input_json", type=str, default='examples.json',
        help="JSON file with input conditions.")
    parser.add_argument(
        "--motion_frame", type=int, default=25,
        help="Motion frames for streaming mode.")
    parser.add_argument(
        "--mode", type=str, default="clip",
        choices=['clip', 'streaming'],
        help="clip: single chunk, streaming: long video.")
    parser.add_argument(
        "--sample_steps", type=int, default=None,
        help="Number of sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None,
        help="Sampling shift factor.")
    parser.add_argument(
        "--sample_text_guide_scale", type=float, default=5.0,
        help="Text CFG scale.")
    parser.add_argument(
        "--sample_audio_guide_scale", type=float, default=4.0,
        help="Audio CFG scale.")
    parser.add_argument(
        "--audio_mode", type=str, default="localfile",
        choices=['localfile', 'tts'],
        help="Audio source: local file or TTS.")
    parser.add_argument(
        "--use_teacache", action="store_true", default=False,
        help="Enable TeaCache acceleration.")
    parser.add_argument(
        "--teacache_thresh", type=float, default=0.2,
        help="TeaCache threshold.")
    parser.add_argument(
        "--use_apg", action="store_true", default=False,
        help="Enable adaptive projected guidance (APG).")
    parser.add_argument(
        "--apg_momentum", type=float, default=-0.75,
        help="APG momentum.")
    parser.add_argument(
        "--apg_norm_threshold", type=float, default=55,
        help="APG norm threshold.")
    parser.add_argument(
        "--color_correction_strength", type=float, default=1.0,
        help="Color correction strength [0.0-1.0].")
    parser.add_argument(
        "--quantize", type=int, default=None, choices=[4, 8],
        help="MLX quantization bits (4 or 8). Default: no quantization.")

    args = parser.parse_args()
    _validate_args(args)
    return args


# ---------------------------------------------------------------------------
# Audio utilities (framework-agnostic: librosa, soundfile, pyloudnorm)
# ---------------------------------------------------------------------------

def custom_init(wav2vec_dir):
    """Initialize Wav2Vec2 model and feature extractor (PyTorch, on CPU)."""
    device = torch.device('cpu')
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir, local_files_only=True
    )
    return wav2vec_feature_extractor, audio_encoder


def loudness_norm(audio_array, sr=16000, lufs=-23):
    """Normalize audio loudness."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def audio_prepare_single(audio_path, sample_rate=16000):
    """Load and normalize a single audio file."""
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        return extract_audio_from_video(audio_path, sample_rate)
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array


def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    """Prepare multi-person audio."""
    if not (left_path == 'None' or right_path == 'None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path == 'None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path == 'None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type == 'para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type == 'add':
        new_human_speech1 = np.concatenate([
            human_speech_array1[:human_speech_array1.shape[0]],
            np.zeros(human_speech_array2.shape[0])
        ])
        new_human_speech2 = np.concatenate([
            np.zeros(human_speech_array1.shape[0]),
            human_speech_array2[:human_speech_array2.shape[0]]
        ])

    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs


def extract_audio_from_video(filename, sample_rate):
    """Extract audio from video file using ffmpeg."""
    raw_audio_path = filename.split('/')[-1].split('.')[0] + '.wav'
    subprocess.run([
        "ffmpeg", "-y", "-i", str(filename),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2",
        str(raw_audio_path),
    ], check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)
    return human_speech_array


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000):
    """Extract audio embedding using Wav2Vec2 (PyTorch on CPU).

    Args:
        speech_array: numpy audio array
        wav2vec_feature_extractor: HuggingFace feature extractor
        audio_encoder: Wav2Vec2 model
        sr: sample rate

    Returns:
        mx.array of audio embeddings [T, S, D]
    """
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # 25 fps

    # Feature extraction
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float()
    audio_feature = audio_feature.unsqueeze(0)

    # Encode with Wav2Vec2
    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )

    if len(embeddings) == 0:
        print("Failed to extract audio embedding")
        return None

    # Stack hidden states: [layers, B, seq, dim]
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    # Rearrange: "b s d -> s b d" using transpose
    audio_emb = audio_emb.permute(1, 0, 2)  # [seq, layers, dim]

    # Convert to MLX
    audio_emb_np = audio_emb.cpu().detach().numpy()
    return mx.array(audio_emb_np)


# ---------------------------------------------------------------------------
# TTS utilities (using Kokoro - stays on PyTorch as it is small)
# ---------------------------------------------------------------------------

def process_tts_single(text, save_dir, voice1):
    """Generate TTS for single speaker."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(text, voice=voice_tensor, speed=1, split_pattern=r'\n+')

    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)

    save_path1 = f'{save_dir}/s1.wav'
    sf.write(save_path1, audios.numpy(), 24000)
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1


def process_tts_multi(text, save_dir, voice1, voice2):
    """Generate TTS for multi-speaker dialogue."""
    from kokoro import KPipeline

    pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    for idx, (speaker, content) in enumerate(matches):
        if speaker == '1':
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == '2':
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))

    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences

    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences.numpy(), 24000)
    sf.write(save_path2, s2_sentences.numpy(), 24000)
    sf.write(save_path_sum, sum_sentences.numpy(), 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    return s1, s2, save_path_sum


# ---------------------------------------------------------------------------
# Embedding save/load (MLX safetensors)
# ---------------------------------------------------------------------------

def save_embedding(emb, path):
    """Save an mx.array embedding as safetensors."""
    mx.save_safetensors(path, {"embedding": emb})


def load_embedding(path):
    """Load an embedding from safetensors."""
    data = mx.load(path)
    return data["embedding"]


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def generate(args):
    _init_logging()

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    assert args.task == "multitalk-14B", "You should choose multitalk-14B as args.task."

    # Read input JSON and prepare audio embeddings
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    wav2vec_feature_extractor, audio_encoder = custom_init(args.wav2vec_dir)
    args.audio_save_dir = os.path.join(
        args.audio_save_dir,
        input_data['cond_image'].split('/')[-1].split('.')[0]
    )
    os.makedirs(args.audio_save_dir, exist_ok=True)

    if args.audio_mode == 'localfile':
        if len(input_data['cond_audio']) == 2:
            new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(
                input_data['cond_audio']['person1'],
                input_data['cond_audio']['person2'],
                input_data['audio_type'],
            )
            audio_embedding_1 = get_embedding(
                new_human_speech1, wav2vec_feature_extractor, audio_encoder
            )
            audio_embedding_2 = get_embedding(
                new_human_speech2, wav2vec_feature_extractor, audio_encoder
            )
            emb1_path = os.path.join(args.audio_save_dir, '1.safetensors')
            emb2_path = os.path.join(args.audio_save_dir, '2.safetensors')
            sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
            sf.write(sum_audio, sum_human_speechs, 16000)
            save_embedding(audio_embedding_1, emb1_path)
            save_embedding(audio_embedding_2, emb2_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio
        elif len(input_data['cond_audio']) == 1:
            human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
            audio_embedding = get_embedding(
                human_speech, wav2vec_feature_extractor, audio_encoder
            )
            emb_path = os.path.join(args.audio_save_dir, '1.safetensors')
            sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
            sf.write(sum_audio, human_speech, 16000)
            save_embedding(audio_embedding, emb_path)
            input_data['cond_audio']['person1'] = emb_path
            input_data['video_audio'] = sum_audio

    elif args.audio_mode == 'tts':
        if 'human2_voice' not in input_data['tts_audio'].keys():
            new_human_speech1, sum_audio = process_tts_single(
                input_data['tts_audio']['text'],
                args.audio_save_dir,
                input_data['tts_audio']['human1_voice'],
            )
            audio_embedding_1 = get_embedding(
                new_human_speech1, wav2vec_feature_extractor, audio_encoder
            )
            emb1_path = os.path.join(args.audio_save_dir, '1.safetensors')
            save_embedding(audio_embedding_1, emb1_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['video_audio'] = sum_audio
        else:
            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                input_data['tts_audio']['text'],
                args.audio_save_dir,
                input_data['tts_audio']['human1_voice'],
                input_data['tts_audio']['human2_voice'],
            )
            audio_embedding_1 = get_embedding(
                new_human_speech1, wav2vec_feature_extractor, audio_encoder
            )
            audio_embedding_2 = get_embedding(
                new_human_speech2, wav2vec_feature_extractor, audio_encoder
            )
            emb1_path = os.path.join(args.audio_save_dir, '1.safetensors')
            emb2_path = os.path.join(args.audio_save_dir, '2.safetensors')
            save_embedding(audio_embedding_1, emb1_path)
            save_embedding(audio_embedding_2, emb2_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio

    # Free Wav2Vec2 model (PyTorch, no longer needed)
    del audio_encoder, wav2vec_feature_extractor
    gc.collect()

    # Create MLX pipeline
    logging.info("Creating MLX MultiTalk pipeline.")
    from wan_mlx import MLXMultiTalkPipeline

    pipeline = MLXMultiTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quantize_bits=args.quantize,
    )

    logging.info("Generating video ...")
    video = pipeline.generate(
        input_data,
        size_buckget=args.size,
        motion_frame=args.motion_frame,
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sample_steps,
        text_guide_scale=args.sample_text_guide_scale,
        audio_guide_scale=args.sample_audio_guide_scale,
        seed=args.base_seed,
        max_frames_num=args.frame_num if args.mode == 'clip' else 1000,
        color_correction_strength=args.color_correction_strength,
        extra_args=args,
    )

    if args.save_file is None:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
        args.save_file = f"{args.task}_{args.size}_{formatted_prompt}_{formatted_time}"

    logging.info(f"Saving generated video to {args.save_file}.mp4")

    # save_video_ffmpeg from wan_mlx accepts mx.array or numpy
    save_video_ffmpeg(video, args.save_file, [input_data['video_audio']])

    logging.info("Finished.")


if __name__ == "__main__":
    import gc
    args = _parse_args()
    generate(args)
