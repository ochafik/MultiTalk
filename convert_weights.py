#!/usr/bin/env python3
"""Convert PyTorch MultiTalk weights to MLX format.

Usage:
    python convert_weights.py --model-type dit --input weights/Wan2.1-I2V-14B-480P --output weights/mlx
    python convert_weights.py --model-type vae --input weights/Wan2.1-I2V-14B-480P/vae --output weights/mlx
    python convert_weights.py --model-type t5 --input weights/umt5-xxl --output weights/mlx
    python convert_weights.py --model-type clip --input weights/open-clip --output weights/mlx
    python convert_weights.py --model-type wav2vec2 --input weights/chinese-wav2vec2-base --output weights/mlx
    python convert_weights.py --model-type all --input weights --output weights/mlx
    python convert_weights.py --model-type dit --input weights/... --output weights/mlx --quantize 4
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_torch_weights(path):
    """Load weights from safetensors or .pt/.pth files."""
    import mlx.core as mx

    path = Path(path)
    if path.is_dir():
        # Look for safetensors files first
        safetensor_files = sorted(path.glob("*.safetensors"))
        pt_files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))

        if safetensor_files:
            weights = {}
            for f in safetensor_files:
                logger.info(f"Loading {f}")
                w = mx.load(str(f))
                weights.update(w)
            return weights
        elif pt_files:
            # For .pt files, we need torch to load them
            import torch
            weights = {}
            for f in pt_files:
                logger.info(f"Loading {f}")
                w = torch.load(str(f), map_location="cpu", weights_only=True)
                if isinstance(w, dict) and "state_dict" in w:
                    w = w["state_dict"]
                for k, v in w.items():
                    weights[k] = mx.array(v.numpy())
            return weights
        else:
            raise FileNotFoundError(f"No weight files found in {path}")
    elif path.suffix == ".safetensors":
        return mx.load(str(path))
    elif path.suffix in (".pt", ".pth"):
        import torch
        w = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(w, dict) and "state_dict" in w:
            w = w["state_dict"]
        return {k: mx.array(v.numpy()) for k, v in w.items()}
    else:
        raise ValueError(f"Unknown file format: {path}")


def remap_dit_weights(weights):
    """Remap PyTorch DiT weight names to MLX convention.

    MLX nn.Module uses attribute names directly, so we need to convert
    PyTorch's state_dict keys to match the MLX module structure.

    Key differences:
    - nn.ModuleList -> Python list, so "blocks.0.xxx" stays the same
    - nn.Sequential -> inline, so "text_embedding.0.weight" -> same
    - nn.Parameter named 'modulation' -> stays as 'modulation'
    - Conv3d weight layout: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
    """
    import mlx.core as mx

    remapped = {}
    for key, value in weights.items():
        new_key = key

        # Handle Conv3d weight transposition
        # PyTorch: (out_channels, in_channels, kD, kH, kW)
        # MLX:     (out_channels, kD, kH, kW, in_channels)
        if "patch_embedding.weight" in key and value.ndim == 5:
            value = mx.transpose(value, axes=(0, 2, 3, 4, 1))

        remapped[new_key] = value

    return remapped


def remap_vae_weights(weights):
    """Remap VAE weights. Conv3d and Conv2d need transposition."""
    import mlx.core as mx

    remapped = {}
    for key, value in weights.items():
        # Conv3d: (O, I, D, H, W) -> (O, D, H, W, I)
        if value.ndim == 5:
            value = mx.transpose(value, axes=(0, 2, 3, 4, 1))
        # Conv2d: (O, I, H, W) -> (O, H, W, I)
        elif value.ndim == 4:
            value = mx.transpose(value, axes=(0, 2, 3, 1))

        remapped[key] = value

    return remapped


def remap_t5_weights(weights):
    """Remap T5 encoder weights."""
    return weights  # T5 is all Linear layers, no transposition needed


def remap_clip_weights(weights):
    """Remap CLIP weights. Conv2d patch embedding needs transposition."""
    import mlx.core as mx

    remapped = {}
    for key, value in weights.items():
        # Conv2d: (O, I, H, W) -> (O, H, W, I)
        if value.ndim == 4:
            value = mx.transpose(value, axes=(0, 2, 3, 1))
        remapped[key] = value

    return remapped


def remap_wav2vec2_weights(weights):
    """Remap Wav2Vec2 weights. Conv1d needs transposition."""
    import mlx.core as mx

    remapped = {}
    for key, value in weights.items():
        # Conv1d: (O, I, L) -> (O, L, I)
        if value.ndim == 3:
            value = mx.transpose(value, axes=(0, 2, 1))
        remapped[key] = value

    return remapped


def quantize_weights(weights, bits=4, group_size=64):
    """Quantize linear layer weights using MLX quantization."""
    import mlx.core as mx
    import mlx.nn as nn

    quantized = {}
    for key, value in weights.items():
        if value.ndim == 2 and "weight" in key and value.shape[0] > 32 and value.shape[1] > 32:
            # Quantize this linear weight
            q, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
            base_key = key.rsplit(".", 1)[0] if "." in key else key
            quantized[key] = q
            quantized[base_key + ".scales"] = scales
            quantized[base_key + ".biases"] = biases
        else:
            quantized[key] = value

    return quantized


def convert_model(model_type, input_path, output_dir, quantize_bits=None, group_size=64):
    """Convert a single model type."""
    import mlx.core as mx

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading {model_type} weights from {input_path}")
    weights = load_torch_weights(input_path)
    logger.info(f"Loaded {len(weights)} weight tensors")

    # Remap weights based on model type
    remap_fn = {
        "dit": remap_dit_weights,
        "vae": remap_vae_weights,
        "t5": remap_t5_weights,
        "clip": remap_clip_weights,
        "wav2vec2": remap_wav2vec2_weights,
    }.get(model_type)

    if remap_fn:
        weights = remap_fn(weights)
        logger.info(f"Remapped weight keys for {model_type}")

    # Optional quantization
    if quantize_bits:
        logger.info(f"Quantizing to {quantize_bits}-bit with group_size={group_size}")
        weights = quantize_weights(weights, bits=quantize_bits, group_size=group_size)
        logger.info(f"Quantized to {len(weights)} tensors")

    # Save
    output_path = os.path.join(output_dir, f"{model_type}.safetensors")
    logger.info(f"Saving to {output_path}")
    mx.save_safetensors(output_path, weights)

    # Print summary
    total_params = sum(v.size for v in weights.values())
    total_bytes = sum(v.nbytes for v in weights.values())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Total size: {total_bytes / 1e9:.2f} GB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch MultiTalk weights to MLX format")
    parser.add_argument("--model-type", required=True,
                        choices=["dit", "vae", "t5", "clip", "wav2vec2", "kokoro", "all"],
                        help="Type of model to convert")
    parser.add_argument("--input", required=True, help="Input weights path (directory or file)")
    parser.add_argument("--output", required=True, help="Output directory for MLX weights")
    parser.add_argument("--quantize", type=int, default=None, choices=[4, 8],
                        help="Quantize to N bits (4 or 8)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Group size for quantization")

    args = parser.parse_args()

    if args.model_type == "all":
        # Convert all models from a base directory
        base = Path(args.input)
        for mtype in ["dit", "vae", "t5", "clip", "wav2vec2"]:
            candidate = base / mtype
            if candidate.exists():
                convert_model(mtype, str(candidate), args.output, args.quantize, args.group_size)
            else:
                logger.warning(f"Skipping {mtype}: {candidate} not found")
    else:
        convert_model(args.model_type, args.input, args.output, args.quantize, args.group_size)


if __name__ == "__main__":
    main()
