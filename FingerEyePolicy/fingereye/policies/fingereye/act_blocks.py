# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(
        self,
        n_encoder_layers=4,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.1,
        pre_norm=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACTEncoderLayer(
                    dim_model=dim_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    feedforward_activation=feedforward_activation,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(n_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

class ACTEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        dropout=0.1,
        pre_norm=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        q = k = x if pos_embed is None else x + pos_embed
        attn_out = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        # attn_out: (S, B, dim)
        
        x = skip + self.dropout1(attn_out)

        # ===== Feedforward =====
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)

        return x

class ACTDecoder(nn.Module):
    def __init__(
        self,
        n_decoder_layers: int,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        dropout: float,
        pre_norm: bool,
    ):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.n_decoder_layers = n_decoder_layers
        self.layers = nn.ModuleList(
            [
                ACTDecoderLayer(
                    dim_model=dim_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    feedforward_activation=feedforward_activation,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        x,
        encoder_out,
        decoder_pos_embed=None,
        encoder_pos_embed=None,
    ):
        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
            attention_weights_list.append(attention_weights)
        if self.norm is not None:
            x = self.norm(x)
        return x, attention_weights_list

class ACTDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim_model = dim_model
        self.head_dim = dim_model // n_heads

        # ===== Standard transformer parts =====
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(feedforward_activation)
        self.pre_norm = pre_norm

    def maybe_add_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x,
        encoder_out,
        decoder_pos_embed=None,
        encoder_pos_embed=None,
    ):
        """
        x: (DS, B, C)
        encoder_out: (ES, B, C)
        """

        # ===== Self-attention =====
        skip = x
        if self.pre_norm:
            x = self.norm1(x)

        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # ===== Cross-attention =====
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        q = self.maybe_add_pos_embed(x, decoder_pos_embed)
        k = self.maybe_add_pos_embed(encoder_out, encoder_pos_embed)

        attn_out, attn_weights = self.multihead_attn(q, k, value=encoder_out)
        # attn_out shape: (DS, B, dim_model)

        x = skip + self.dropout2(attn_out)

        # ===== Feedforward =====
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)

        return x, attn_weights


class StructuredACTDecoder(nn.Module):
    """ACT decoder with parallel group-wise cross-attention over FE, wrist, and joint tokens."""

    def __init__(
        self,
        n_decoder_layers: int,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                StructuredACTDecoderLayer(
                    dim_model=dim_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    feedforward_activation=feedforward_activation,
                    dropout=dropout,
                    pre_norm=pre_norm,
                )
                for _ in range(n_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        if encoder_pos_embed is not None:
            raise ValueError("StructuredACTDecoder does not support encoder_pos_embed.")

        n_tokens = encoder_out.shape[0]
        if n_tokens == 6:
            fe_tokens = encoder_out[:4]
            wrist_token = encoder_out[4:5]
            joint_token = encoder_out[5:6]
        elif n_tokens == 4:
            fe_tokens = encoder_out[:2]
            wrist_token = encoder_out[2:3]
            joint_token = encoder_out[3:4]
        elif n_tokens == 3:
            fe_tokens = encoder_out[:1]
            wrist_token = encoder_out[1:2]
            joint_token = encoder_out[2:3]
        elif n_tokens == 2:
            fe_tokens = encoder_out[:0]
            wrist_token = encoder_out[0:1]
            joint_token = encoder_out[1:2]
        else:
            raise ValueError(
                "StructuredACTDecoder expects condition tokens ordered as "
                "4 FE + 1 wrist + 1 joint, 2 FE + 1 wrist + 1 joint, "
                "1 FE + 1 wrist + 1 joint, or 1 wrist + 1 joint; "
                f"got {n_tokens} tokens."
            )

        attention_weights_list = []
        for layer in self.layers:
            x, attention_weights = layer(
                x,
                joint_token=joint_token,
                wrist_token=wrist_token,
                fe_tokens=fe_tokens,
                decoder_pos_embed=decoder_pos_embed,
            )
            attention_weights_list.append(attention_weights)
        if self.norm is not None:
            x = self.norm(x)
        return x, attention_weights_list


class StructuredACTDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        feedforward_activation: str,
        dropout: float,
        pre_norm: bool,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.cross_attn_joint = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.cross_attn_wrist = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.cross_attn_fe = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.activation = get_activation_fn(feedforward_activation)

        self.norm_self = nn.LayerNorm(dim_model)
        self.norm_joint = nn.LayerNorm(dim_model)
        self.norm_wrist = nn.LayerNorm(dim_model)
        self.norm_fe = nn.LayerNorm(dim_model)
        self.norm_cross = nn.LayerNorm(dim_model)
        self.norm_ffn = nn.LayerNorm(dim_model)
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def maybe_add_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def _self_attention(self, x, decoder_pos_embed):
        skip = x
        if self.pre_norm:
            x = self.norm_self(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout_self(x)
        if not self.pre_norm:
            x = self.norm_self(x)
        return x

    def _cross_delta(self, x, memory, norm, attn, decoder_pos_embed):
        if memory.shape[0] == 0:
            return None, None
        q_input = norm(x) if self.pre_norm else x
        q = self.maybe_add_pos_embed(q_input, decoder_pos_embed)
        attn_out, attn_weights = attn(q, memory, value=memory)
        return attn_out, attn_weights

    def _ffn(self, x):
        skip = x
        if self.pre_norm:
            x = self.norm_ffn(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout_ffn(x)
        if not self.pre_norm:
            x = self.norm_ffn(x)
        return x

    def forward(self, x, joint_token, wrist_token, fe_tokens, decoder_pos_embed=None):
        x = self._self_attention(x, decoder_pos_embed)
        attention_weights = {}
        deltas = []

        delta_joint, attention_weights["joint"] = self._cross_delta(
            x, joint_token, self.norm_joint, self.cross_attn_joint, decoder_pos_embed
        )
        deltas.append(delta_joint)

        delta_wrist, attention_weights["wrist"] = self._cross_delta(
            x, wrist_token, self.norm_wrist, self.cross_attn_wrist, decoder_pos_embed
        )
        deltas.append(delta_wrist)

        delta_fe, attention_weights["fe"] = self._cross_delta(
            x, fe_tokens, self.norm_fe, self.cross_attn_fe, decoder_pos_embed
        )
        if delta_fe is not None:
            deltas.append(delta_fe)

        x = x + self.dropout_cross(torch.stack(deltas, dim=0).mean(dim=0))
        if not self.pre_norm:
            x = self.norm_cross(x)
        x = self._ffn(x)
        return x, attention_weights


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x):
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed

def get_activation_fn(activation: str):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int):
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()
