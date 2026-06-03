# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma 1 2B transformer blocks - PyTorch Reference Implementation.

Gemma 2B style transformer layers with:
    - RMSNorm (pre-normalization, Gemma (weight + 1) convention)
    - Multi-Query Attention (MQA) with num_kv_heads=1
    - GeGLU MLP (gated tanh-GELU activation)
    - Rotary Position Embeddings (RoPE)

Gemma 2B (VLM backbone): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1, head_dim=256

This is the standalone Gemma-1-2B backbone only -- the pi0.5 action-expert
(adaRMS / gated-residual) additions are intentionally omitted, since the
tt_metal/CUDA port targets this backbone first.

Self-contained: GemmaConfig is defined here rather than imported from the
surrounding pi0_5 package.
"""

import dataclasses
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class GemmaConfig:
    """Minimal Gemma config (the fields the reference blocks need + model dims)."""

    width: int = 2048           # hidden size / d_model
    depth: int = 18             # number of transformer blocks
    mlp_dim: int = 16384        # FFN hidden dim
    num_heads: int = 8          # query heads
    num_kv_heads: int = 1       # key/value heads (MQA)
    head_dim: int = 256
    vocab_size: int = 257_152   # PaliGemma vocab
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0


# ============================================================================
# RMSNorm
# ============================================================================


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with Gemma-style weight offset: uses (weight + 1) instead of weight."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * (weight + 1.0)


# ============================================================================
# Rotary Position Embeddings
# ============================================================================


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin), each (max_seq_len, head_dim // 2)."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs_outer = torch.outer(t, freqs)
    return torch.cos(freqs_outer), torch.sin(freqs_outer)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q (B,Nh,T,Dh) and K (B,Nkv,T,Dh)."""
    seq_len = q.shape[2]
    head_dim = q.shape[-1]

    if position_ids is not None:
        cos = cos[position_ids]              # (B, T, Dh//2)
        sin = sin[position_ids]
        cos = cos.unsqueeze(1)               # (B, 1, T, Dh//2)
        sin = sin.unsqueeze(1)
    else:
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # keep rotation in q/k precision so bf16 stays bf16
    q_dtype, k_dtype = q.dtype, k.dtype
    cos_q, sin_q = cos.to(q_dtype), sin.to(q_dtype)
    cos_k, sin_k = cos.to(k_dtype), sin.to(k_dtype)

    q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
    k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]

    q_rotated = torch.cat(
        [
            q1 * cos_q[..., : head_dim // 2] - q2 * sin_q[..., : head_dim // 2],
            q1 * sin_q[..., head_dim // 2 :] + q2 * cos_q[..., head_dim // 2 :],
        ],
        dim=-1,
    )
    k_rotated = torch.cat(
        [
            k1 * cos_k[..., : head_dim // 2] - k2 * sin_k[..., : head_dim // 2],
            k1 * sin_k[..., head_dim // 2 :] + k2 * cos_k[..., head_dim // 2 :],
        ],
        dim=-1,
    )
    return q_rotated, k_rotated


# ============================================================================
# Multi-Query Attention
# ============================================================================


class GemmaAttention:
    """Gemma Multi-Query Attention: 8 query heads, 1 KV head broadcast across them."""

    def __init__(self, config: GemmaConfig, weights: Dict[str, torch.Tensor], layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.q_proj = weights["self_attn.q_proj.weight"]
        self.k_proj = weights["self_attn.k_proj.weight"]
        self.v_proj = weights["self_attn.v_proj.weight"]
        self.o_proj = weights["self_attn.o_proj.weight"]
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj.to(hidden_states.dtype)
        k_proj = self.k_proj.to(hidden_states.dtype)
        v_proj = self.v_proj.to(hidden_states.dtype)
        q = F.linear(hidden_states, q_proj)
        k = F.linear(hidden_states, k_proj)
        v = F.linear(hidden_states, v_proj)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        kv_len = k.shape[2]
        k_expanded = k.expand(batch_size, self.num_heads, kv_len, self.head_dim)
        v_expanded = v.expand(batch_size, self.num_heads, kv_len, self.head_dim)

        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        o_proj = self.o_proj.to(attn_output.dtype)
        output = F.linear(attn_output, o_proj)
        return output, new_cache


# ============================================================================
# GeGLU MLP
# ============================================================================


class GemmaMLP:
    """Gemma MLP: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: GemmaConfig, weights: Dict[str, torch.Tensor]):
        self.config = config
        self.gate_proj = weights["mlp.gate_proj.weight"]
        self.up_proj = weights["mlp.up_proj.weight"]
        self.down_proj = weights["mlp.down_proj.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj.to(x.dtype)
        up_proj = self.up_proj.to(x.dtype)
        down_proj = self.down_proj.to(x.dtype)
        gate = F.linear(x, gate_proj)
        up = F.linear(x, up_proj)
        hidden = F.gelu(gate, approximate="tanh") * up
        return F.linear(hidden, down_proj)


# ============================================================================
# Full Transformer Block (Pre-LN, residual)
# ============================================================================


class GemmaBlock:
    """x -> RMSNorm -> Attention -> +res -> RMSNorm -> MLP -> +res."""

    def __init__(self, config: GemmaConfig, weights: Dict[str, torch.Tensor], layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]
        self.attention = GemmaAttention(config, weights, layer_idx)
        self.mlp = GemmaMLP(config, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed = rms_norm(hidden_states, self.input_layernorm_weight, self.config.rms_norm_eps)
        attn_output, new_cache = self.attention.forward(
            normed, cos, sin, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = hidden_states + attn_output

        normed = rms_norm(hidden_states, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        mlp_output = self.mlp.forward(normed)
        hidden_states = hidden_states + mlp_output
        return hidden_states, new_cache
