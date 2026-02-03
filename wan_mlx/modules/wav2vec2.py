# Wav2Vec2 audio encoder ported to MLX.
# Standalone implementation (no HuggingFace PyTorch dependency).
# Weight names match HuggingFace Wav2Vec2 naming for loading from safetensors.

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ['Wav2Vec2Model']


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def linear_interpolation(features: mx.array, seq_len: int) -> mx.array:
    """Interpolate features to match video frame count.

    Args:
        features: [B, T, D] tensor of features.
        seq_len: target sequence length (number of video frames).

    Returns:
        Interpolated features of shape [B, seq_len, D].
    """
    # Transpose to [B, D, T] for per-channel interpolation
    features_t = mx.transpose(features, axes=(0, 2, 1))
    # Use numpy for interpolation (one-time operation, not in hot path)
    features_np = np.array(features_t)
    B, D, T = features_np.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, seq_len)
    result = np.zeros((B, D, seq_len))
    for b in range(B):
        for d in range(D):
            result[b, d] = np.interp(x_new, x_old, features_np[b, d])
    result = mx.array(result)
    return mx.transpose(result, axes=(0, 2, 1))  # [B, seq_len, D]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class Wav2Vec2Output:
    """Mirrors transformers.modeling_outputs.BaseModelOutput."""
    last_hidden_state: mx.array
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Wav2Vec2Config:
    # Feature extractor (7 conv layers)
    conv_dim: List[int] = field(
        default_factory=lambda: [512, 512, 512, 512, 512, 512, 512]
    )
    conv_kernel: List[int] = field(
        default_factory=lambda: [10, 3, 3, 3, 3, 2, 2]
    )
    conv_stride: List[int] = field(
        default_factory=lambda: [5, 2, 2, 2, 2, 2, 2]
    )
    num_feat_extract_layers: int = 7

    # Feature projection
    hidden_size: int = 768
    feat_extract_norm: str = "group"  # "group" for first layer, rest use no norm

    # Transformer encoder
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Output
    output_hidden_states: bool = False
    output_attentions: bool = True


# ---------------------------------------------------------------------------
# Feature extractor conv layers
# ---------------------------------------------------------------------------

class Wav2Vec2GroupNormConvLayer(nn.Module):
    """First conv layer with GroupNorm."""

    def __init__(self, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        # Input is raw waveform: 1 channel
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.layer_norm = nn.GroupNorm(
            num_groups=out_channels, dims=out_channels
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, 1, T]  (channels-first for Conv1d but MLX Conv1d expects [B, T, C])
        # MLX Conv1d: input [B, T, C], output [B, T', C']
        x = self.conv(x)
        # GroupNorm expects [B, T, C] which is what MLX Conv1d outputs
        x = self.layer_norm(x)
        x = nn.gelu(x)
        return x


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    """Conv layers 1-6 without normalization."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = nn.gelu(x)
        return x


class Wav2Vec2FeatureExtractor(nn.Module):
    """Stack of 7 1D conv layers for raw waveform feature extraction."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        conv_layers = []
        for i in range(config.num_feat_extract_layers):
            in_c = 1 if i == 0 else config.conv_dim[i - 1]
            out_c = config.conv_dim[i]
            k = config.conv_kernel[i]
            s = config.conv_stride[i]
            if i == 0 and config.feat_extract_norm == "group":
                conv_layers.append(
                    Wav2Vec2GroupNormConvLayer(out_c, k, s)
                )
            else:
                conv_layers.append(
                    Wav2Vec2NoLayerNormConvLayer(in_c, out_c, k, s)
                )
        self.conv_layers = conv_layers

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: [B, T_raw] raw waveform or [B, 1, T_raw]
        Returns:
            [B, T_feat, conv_dim[-1]]
        """
        if x.ndim == 2:
            # [B, T] -> [B, T, 1]  (MLX Conv1d wants channels last)
            x = mx.expand_dims(x, axis=-1)
        elif x.ndim == 3 and x.shape[1] > x.shape[2]:
            # Likely [B, T, 1] already; keep as-is
            pass
        elif x.ndim == 3 and x.shape[1] == 1:
            # [B, 1, T] channels-first -> [B, T, 1] channels-last
            x = mx.transpose(x, axes=(0, 2, 1))

        for layer in self.conv_layers:
            x = layer(x)
        return x  # [B, T_feat, D]


# ---------------------------------------------------------------------------
# Feature projection
# ---------------------------------------------------------------------------

class Wav2Vec2FeatureProjection(nn.Module):
    """LayerNorm + Linear projection from conv features to hidden size."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        last_conv_dim = config.conv_dim[-1]
        self.layer_norm = nn.LayerNorm(last_conv_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(last_conv_dim, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            x: [B, T, conv_dim]
        Returns:
            (hidden_states [B, T, hidden_size], normed_features [B, T, conv_dim])
        """
        normed = self.layer_norm(x)
        projected = self.projection(normed)
        projected = self.dropout(projected)
        return projected, normed


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------

class Wav2Vec2Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            hidden_states: [B, T, D]
            attention_mask: optional [B, 1, 1, T] additive mask
        Returns:
            (output [B, T, D], attn_weights [B, H, T, T])
        """
        B, T, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(hidden_states).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden_states).reshape(B, T, H, d).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden_states).reshape(B, T, H, d).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        # Compute attention weights separately for output (if needed)
        attn_weights = (q @ mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Reshape attn_output back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class Wav2Vec2FeedForward(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output_dense = nn.Linear(
            config.intermediate_size, config.hidden_size
        )
        self.intermediate_dropout = nn.Dropout(p=config.hidden_dropout)
        self.output_dropout = nn.Dropout(p=config.hidden_dropout)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------

class Wav2Vec2EncoderLayer(nn.Module):
    """Single transformer encoder layer: LN + Attn + residual + LN + FFN + residual."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights


# ---------------------------------------------------------------------------
# Encoder (stack of layers)
# ---------------------------------------------------------------------------

class Wav2Vec2Encoder(nn.Module):
    """Transformer encoder: stack of Wav2Vec2EncoderLayer."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layers = [
            Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attn_weights = layer(
                hidden_states, attention_mask=attention_mask
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_attentions


# ---------------------------------------------------------------------------
# Full Wav2Vec2 Model
# ---------------------------------------------------------------------------

class Wav2Vec2Model(nn.Module):
    """
    Standalone MLX Wav2Vec2 model for audio feature extraction.

    Matches the HuggingFace Wav2Vec2Model architecture so that converted
    safetensors weights can be loaded directly. Weight names follow the
    HuggingFace naming convention:
        feature_extractor.conv_layers.{i}.conv.weight
        feature_extractor.conv_layers.{i}.layer_norm.weight / .bias
        feature_projection.layer_norm.weight / .bias
        feature_projection.projection.weight / .bias
        encoder.layers.{i}.attention.{q,k,v}_proj.weight / .bias
        encoder.layers.{i}.attention.out_proj.weight / .bias
        encoder.layers.{i}.layer_norm.weight / .bias
        encoder.layers.{i}.feed_forward.intermediate_dense.weight / .bias
        encoder.layers.{i}.feed_forward.output_dense.weight / .bias
        encoder.layers.{i}.final_layer_norm.weight / .bias
        encoder.layer_norm.weight / .bias
    """

    def __init__(self, config: Optional[Wav2Vec2Config] = None):
        super().__init__()
        if config is None:
            config = Wav2Vec2Config()
        self.config = config

        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        self.encoder = Wav2Vec2Encoder(config)

    def __call__(
        self,
        input_values: mx.array,
        seq_len: int,
        attention_mask: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Wav2Vec2Output:
        """
        Full forward pass: feature extraction -> interpolation -> projection -> encoder.

        Args:
            input_values: [B, T_raw] raw audio waveform.
            seq_len: target number of frames (video frame count).
            attention_mask: optional [B, T] mask.
            output_attentions: whether to return attention weights.
            output_hidden_states: whether to return all hidden states.

        Returns:
            Wav2Vec2Output with last_hidden_state, hidden_states, attentions.
        """
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        # Feature extraction (conv layers)
        extract_features = self.feature_extractor(input_values)
        # extract_features: [B, T_feat, conv_dim]

        # Interpolate to match video frame count
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        # Feature projection
        hidden_states, _normed = self.feature_projection(extract_features)

        # Transformer encoder
        encoder_out = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state, all_hidden_states, all_attentions = encoder_out

        return Wav2Vec2Output(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def feature_extract(
        self,
        input_values: mx.array,
        seq_len: int,
    ) -> mx.array:
        """Extract and interpolate conv features (no encoder).

        Args:
            input_values: [B, T_raw] raw audio waveform.
            seq_len: target number of frames.

        Returns:
            [B, seq_len, conv_dim] interpolated features.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        return extract_features

    def encode(
        self,
        extract_features: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Wav2Vec2Output:
        """Encode pre-extracted features through projection + transformer.

        Args:
            extract_features: [B, T, conv_dim] from feature_extract().
            attention_mask: optional mask.
            output_attentions: whether to return attention weights.
            output_hidden_states: whether to return all hidden states.

        Returns:
            Wav2Vec2Output.
        """
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        hidden_states, _normed = self.feature_projection(extract_features)

        encoder_out = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state, all_hidden_states, all_attentions = encoder_out

        return Wav2Vec2Output(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
